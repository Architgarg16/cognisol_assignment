"""Explainable content-based career recommender.

The model uses cosine-distance nearest neighbors for candidate generation and an
evidence-aware reranker for final recommendations. It intentionally excludes
demographic and protected attributes.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors


class CatalogValidationError(ValueError):
    """Raised when the career catalog is not internally consistent."""


class ProfileValidationError(ValueError):
    """Raised when a student feature profile is malformed."""


class InsufficientEvidenceError(ValueError):
    """Raised when the profile cannot support a responsible recommendation."""


class CareerRecommender:
    """Fit and serve evidence-aware content recommendations.

    A career catalog defines the importance of skills and subjects for each
    career. Student mastery values become a vector in the same feature space.
    Nearest neighbors supplies candidate careers; a deterministic reranker then
    combines directional alignment, absolute proficiency, evidence coverage,
    and evidence reliability.
    """

    MODEL_FAMILY = "content-knn-cosine"
    FEATURE_SCHEMA_VERSION = "1.0"

    def __init__(
        self,
        *,
        min_evidence_count: int = 3,
        reliability_saturation: int = 10,
        min_coverage: float = 0.35,
    ) -> None:
        if min_evidence_count < 1:
            raise ValueError("min_evidence_count must be positive")
        if reliability_saturation < min_evidence_count:
            raise ValueError(
                "reliability_saturation must be at least min_evidence_count"
            )
        if not 0 < min_coverage <= 1:
            raise ValueError("min_coverage must be in (0, 1]")

        self.min_evidence_count = min_evidence_count
        self.reliability_saturation = reliability_saturation
        self.min_coverage = min_coverage
        self.catalog: dict[str, Any] | None = None
        self.feature_names: list[str] = []
        self.career_ids: list[str] = []
        self.career_matrix: np.ndarray | None = None
        self.model_version: str | None = None
        self._nn: NearestNeighbors | None = None
        self._career_lookup: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_catalog_file(
        cls,
        catalog_path: str | Path,
        **kwargs: Any,
    ) -> "CareerRecommender":
        """Create and fit a recommender from a JSON catalog."""

        with Path(catalog_path).open("r", encoding="utf-8") as handle:
            catalog = json.load(handle)
        return cls(**kwargs).fit(catalog)

    def fit(self, catalog: Mapping[str, Any]) -> "CareerRecommender":
        """Validate the catalog and fit the nearest-neighbor candidate model."""

        validated = self._validate_catalog(catalog)
        self.catalog = validated
        skills = validated["feature_schema"]["skills"]
        subjects = validated["feature_schema"]["subjects"]
        self.feature_names = [
            *(f"skill:{name}" for name in skills),
            *(f"subject:{name}" for name in subjects),
        ]

        careers = validated["careers"]
        self.career_ids = [career["id"] for career in careers]
        self._career_lookup = {career["id"]: career for career in careers}
        self.career_matrix = np.vstack(
            [self._career_vector(career) for career in careers]
        )

        self._nn = NearestNeighbors(metric="cosine", algorithm="brute")
        self._nn.fit(self.career_matrix)

        canonical = json.dumps(validated, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        digest = hashlib.sha256(canonical).hexdigest()[:12]
        self.model_version = f"{self.MODEL_FAMILY}-{digest}"
        return self

    def recommend(
        self,
        student_profile: Mapping[str, Any],
        *,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Return ranked, explainable career recommendations.

        The expected profile fields are:

        - `student_id`
        - `skill_mastery`: mapping from skill to score in [0, 100]
        - `skill_evidence_counts`: mapping from skill to non-negative count
        - `subject_scores`: mapping from subject to score in [0, 100]
        - `subject_evidence_counts`: mapping from subject to non-negative count
        """

        self._ensure_fitted()
        if top_k < 1:
            raise ValueError("top_k must be positive")

        (
            student_vector,
            observed_mask,
            reliability_vector,
            ignored_features,
        ) = self._student_vector(student_profile)

        observed_count = int(observed_mask.sum())
        if observed_count < 2:
            raise InsufficientEvidenceError(
                "At least two eligible skill or subject features are required"
            )

        assert self._nn is not None
        assert self.career_matrix is not None
        query = student_vector.reshape(1, -1)
        neighbor_count = len(self.career_ids)
        distances, indices = self._nn.kneighbors(query, n_neighbors=neighbor_count)

        candidates: list[dict[str, Any]] = []
        for raw_distance, career_index in zip(distances[0], indices[0], strict=True):
            career_id = self.career_ids[int(career_index)]
            career = self._career_lookup[career_id]
            career_vector = self.career_matrix[int(career_index)]
            scored = self._score_candidate(
                career=career,
                career_vector=career_vector,
                student_vector=student_vector,
                observed_mask=observed_mask,
                reliability_vector=reliability_vector,
                raw_knn_similarity=max(0.0, 1.0 - float(raw_distance)),
            )
            if scored is not None:
                candidates.append(scored)

        if not candidates:
            raise InsufficientEvidenceError(
                "Available evidence does not cover enough of any career profile"
            )

        candidates.sort(
            key=lambda item: (
                -item["match_score"],
                -item["confidence_score"],
                item["career_id"],
            )
        )
        selected = candidates[: min(top_k, len(candidates))]
        for rank, recommendation in enumerate(selected, start=1):
            recommendation["rank"] = rank

        student_id = str(student_profile.get("student_id", "")).strip()
        return {
            "student_id": student_id,
            "model_version": self.model_version,
            "feature_schema_version": self.FEATURE_SCHEMA_VERSION,
            "generated_at": datetime.now(UTC).isoformat(),
            "eligible_feature_count": observed_count,
            "total_feature_count": len(self.feature_names),
            "ignored_or_low_evidence_features": ignored_features,
            "recommendations": selected,
            "disclaimer": (
                "Exploratory educational guidance only. This model does not "
                "determine aptitude, admission, or employment suitability."
            ),
        }

    def save(self, model_path: str | Path) -> Path:
        """Persist the fitted model.

        Joblib uses pickle internally. Only load artifacts produced by a trusted
        source.
        """

        self._ensure_fitted()
        destination = Path(model_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return destination

    @classmethod
    def load(cls, model_path: str | Path) -> "CareerRecommender":
        """Load a trusted fitted model artifact."""

        loaded = joblib.load(Path(model_path))
        if not isinstance(loaded, cls):
            raise TypeError("Artifact does not contain a CareerRecommender")
        loaded._ensure_fitted()
        return loaded

    def model_card(self) -> dict[str, Any]:
        """Return machine-readable model documentation."""

        self._ensure_fitted()
        assert self.catalog is not None
        return {
            "model_version": self.model_version,
            "model_family": self.MODEL_FAMILY,
            "feature_schema_version": self.FEATURE_SCHEMA_VERSION,
            "algorithm": (
                "Cosine-distance k-nearest-neighbor candidate generation with "
                "evidence-aware deterministic reranking"
            ),
            "career_count": len(self.career_ids),
            "feature_count": len(self.feature_names),
            "features": self.feature_names,
            "minimum_evidence_count_per_feature": self.min_evidence_count,
            "minimum_career_coverage": self.min_coverage,
            "ranking_formula": {
                "alignment_weight": 0.65,
                "proficiency_weight": 0.35,
                "coverage_penalty": "0.70 + 0.30 * coverage",
                "confidence": "coverage * evidence_reliability",
            },
            "intended_use": (
                "Rank exploratory career paths from demonstrated skill mastery "
                "and subject performance in a learning analytics application."
            ),
            "excluded_features": [
                "name",
                "age",
                "gender",
                "ethnicity",
                "disability",
                "parent identity",
                "institute identity",
                "economic status",
            ],
            "limitations": [
                "Career definitions are curated, not learned from labor outcomes.",
                "Offline benchmark profiles are synthetic sanity checks.",
                "Scores are not probabilities of career success.",
                "Sparse evidence lowers confidence and can suppress ranking.",
                "The catalog and model require expert review before real use.",
            ],
        }

    def _score_candidate(
        self,
        *,
        career: Mapping[str, Any],
        career_vector: np.ndarray,
        student_vector: np.ndarray,
        observed_mask: np.ndarray,
        reliability_vector: np.ndarray,
        raw_knn_similarity: float,
    ) -> dict[str, Any] | None:
        total_importance = float(career_vector.sum())
        observed_importance = float(career_vector[observed_mask].sum())
        if total_importance <= 0 or observed_importance <= 0:
            return None

        coverage = observed_importance / total_importance
        if coverage < self.min_coverage:
            return None

        active_mask = observed_mask & (career_vector > 0)
        student_active = student_vector[active_mask]
        career_active = career_vector[active_mask]
        if student_active.size == 0:
            return None

        denominator = float(
            np.linalg.norm(student_active) * np.linalg.norm(career_active)
        )
        alignment = (
            float(np.dot(student_active, career_active) / denominator)
            if denominator > 0
            else 0.0
        )
        proficiency = float(np.average(student_active, weights=career_active))
        reliability = float(
            np.average(
                reliability_vector[active_mask],
                weights=career_active,
            )
        )

        coverage_penalty = 0.70 + 0.30 * coverage
        match_score = (0.65 * alignment + 0.35 * proficiency) * coverage_penalty * 100
        confidence_score = coverage * reliability * 100

        reasons = self._top_evidence(
            career_vector=career_vector,
            student_vector=student_vector,
            observed_mask=observed_mask,
            mode="strength",
        )
        gaps = self._top_evidence(
            career_vector=career_vector,
            student_vector=student_vector,
            observed_mask=observed_mask,
            mode="gap",
        )
        missing = [
            self._feature_label(self.feature_names[index])
            for index in np.where((career_vector >= 0.60) & ~observed_mask)[0]
        ]

        return {
            "career_id": career["id"],
            "title": career["title"],
            "description": career["description"],
            "match_score": round(match_score, 2),
            "confidence_score": round(confidence_score, 2),
            "alignment_score": round(alignment * 100, 2),
            "proficiency_score": round(proficiency * 100, 2),
            "evidence_coverage": round(coverage * 100, 2),
            "candidate_similarity": round(raw_knn_similarity * 100, 2),
            "reasons": reasons,
            "development_gaps": gaps,
            "missing_critical_evidence": missing,
            "explanation": (
                f"{career['title']} matches the student's demonstrated "
                f"strength pattern with {coverage * 100:.0f}% of the career "
                "profile supported by eligible evidence."
            ),
        }

    def _top_evidence(
        self,
        *,
        career_vector: np.ndarray,
        student_vector: np.ndarray,
        observed_mask: np.ndarray,
        mode: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for index, feature_name in enumerate(self.feature_names):
            importance = float(career_vector[index])
            if importance <= 0 or not observed_mask[index]:
                continue
            score = float(student_vector[index])
            if mode == "strength":
                contribution = importance * score
            elif mode == "gap":
                contribution = importance * (1.0 - score)
            else:
                raise ValueError(f"Unsupported evidence mode: {mode}")
            items.append(
                {
                    "feature": self._feature_label(feature_name),
                    "feature_type": feature_name.split(":", maxsplit=1)[0],
                    "student_score": round(score * 100, 2),
                    "career_importance": round(importance * 100, 2),
                    "contribution": round(contribution * 100, 2),
                }
            )

        items.sort(
            key=lambda item: (
                -item["contribution"],
                item["feature"],
            )
        )
        return items[:limit]

    def _student_vector(
        self, profile: Mapping[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        if not isinstance(profile, Mapping):
            raise ProfileValidationError("student_profile must be a mapping")

        student_id = str(profile.get("student_id", "")).strip()
        if not student_id:
            raise ProfileValidationError("student_id is required")

        skill_scores = self._validated_number_map(
            profile.get("skill_mastery", {}),
            "skill_mastery",
            maximum=100,
        )
        skill_counts = self._validated_number_map(
            profile.get("skill_evidence_counts", {}),
            "skill_evidence_counts",
            maximum=None,
            integer_only=True,
        )
        subject_scores = self._validated_number_map(
            profile.get("subject_scores", {}),
            "subject_scores",
            maximum=100,
        )
        subject_counts = self._validated_number_map(
            profile.get("subject_evidence_counts", {}),
            "subject_evidence_counts",
            maximum=None,
            integer_only=True,
        )

        vector = np.zeros(len(self.feature_names), dtype=float)
        observed = np.zeros(len(self.feature_names), dtype=bool)
        reliability = np.zeros(len(self.feature_names), dtype=float)
        ignored: list[str] = []

        for index, feature_name in enumerate(self.feature_names):
            feature_type, raw_name = feature_name.split(":", maxsplit=1)
            if feature_type == "skill":
                scores, counts = skill_scores, skill_counts
            else:
                scores, counts = subject_scores, subject_counts

            if raw_name not in scores:
                continue
            count = int(counts.get(raw_name, 0))
            if count < self.min_evidence_count:
                ignored.append(
                    f"{feature_name} (evidence={count}, "
                    f"minimum={self.min_evidence_count})"
                )
                continue

            vector[index] = float(scores[raw_name]) / 100.0
            observed[index] = True
            reliability[index] = min(count / self.reliability_saturation, 1.0)

        known_skills = {
            name.split(":", maxsplit=1)[1]
            for name in self.feature_names
            if name.startswith("skill:")
        }
        known_subjects = {
            name.split(":", maxsplit=1)[1]
            for name in self.feature_names
            if name.startswith("subject:")
        }
        ignored.extend(
            f"skill:{name} (unknown feature)"
            for name in sorted(set(skill_scores) - known_skills)
        )
        ignored.extend(
            f"subject:{name} (unknown feature)"
            for name in sorted(set(subject_scores) - known_subjects)
        )
        return vector, observed, reliability, ignored

    @staticmethod
    def _validated_number_map(
        value: Any,
        field_name: str,
        *,
        maximum: float | None,
        integer_only: bool = False,
    ) -> dict[str, float]:
        if not isinstance(value, Mapping):
            raise ProfileValidationError(f"{field_name} must be a mapping")

        validated: dict[str, float] = {}
        for raw_key, raw_number in value.items():
            key = str(raw_key).strip()
            if not key:
                raise ProfileValidationError(f"{field_name} contains an empty key")
            if isinstance(raw_number, bool) or not isinstance(raw_number, (int, float)):
                raise ProfileValidationError(f"{field_name}.{key} must be numeric")
            number = float(raw_number)
            if not np.isfinite(number) or number < 0:
                raise ProfileValidationError(
                    f"{field_name}.{key} must be non-negative and finite"
                )
            if maximum is not None and number > maximum:
                raise ProfileValidationError(f"{field_name}.{key} must be <= {maximum}")
            if integer_only and not number.is_integer():
                raise ProfileValidationError(f"{field_name}.{key} must be an integer")
            validated[key] = number
        return validated

    def _career_vector(self, career: Mapping[str, Any]) -> np.ndarray:
        vector = np.zeros(len(self.feature_names), dtype=float)
        skills = career["skill_weights"]
        subjects = career["subject_weights"]
        for index, feature_name in enumerate(self.feature_names):
            feature_type, raw_name = feature_name.split(":", maxsplit=1)
            source = skills if feature_type == "skill" else subjects
            vector[index] = float(source.get(raw_name, 0.0))
        return vector

    @classmethod
    def _validate_catalog(cls, catalog: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(catalog, Mapping):
            raise CatalogValidationError("catalog must be a mapping")
        try:
            schema = catalog["feature_schema"]
            skills = list(schema["skills"])
            subjects = list(schema["subjects"])
            careers = list(catalog["careers"])
        except (KeyError, TypeError) as exc:
            raise CatalogValidationError(
                "catalog requires feature_schema.skills, "
                "feature_schema.subjects, and careers"
            ) from exc

        if not skills or not subjects:
            raise CatalogValidationError("skills and subjects must both be non-empty")
        if len(skills) != len(set(skills)):
            raise CatalogValidationError("skill names must be unique")
        if len(subjects) != len(set(subjects)):
            raise CatalogValidationError("subject names must be unique")
        if not careers:
            raise CatalogValidationError("at least one career is required")

        career_ids: set[str] = set()
        validated_careers: list[dict[str, Any]] = []
        for career in careers:
            if not isinstance(career, Mapping):
                raise CatalogValidationError("each career must be a mapping")
            career_id = str(career.get("id", "")).strip()
            title = str(career.get("title", "")).strip()
            description = str(career.get("description", "")).strip()
            if not career_id or not title or not description:
                raise CatalogValidationError(
                    "each career requires id, title, and description"
                )
            if career_id in career_ids:
                raise CatalogValidationError(f"duplicate career id: {career_id}")
            career_ids.add(career_id)

            skill_weights = cls._validate_weight_map(
                career.get("skill_weights", {}),
                allowed=set(skills),
                field=f"{career_id}.skill_weights",
            )
            subject_weights = cls._validate_weight_map(
                career.get("subject_weights", {}),
                allowed=set(subjects),
                field=f"{career_id}.subject_weights",
            )
            if sum(skill_weights.values()) + sum(subject_weights.values()) <= 0:
                raise CatalogValidationError(
                    f"{career_id} must have at least one positive weight"
                )
            validated_careers.append(
                {
                    "id": career_id,
                    "title": title,
                    "description": description,
                    "skill_weights": skill_weights,
                    "subject_weights": subject_weights,
                }
            )

        return {
            "catalog_version": str(catalog.get("catalog_version", "1.0")),
            "feature_schema": {
                "skills": [str(item) for item in skills],
                "subjects": [str(item) for item in subjects],
            },
            "careers": validated_careers,
        }

    @staticmethod
    def _validate_weight_map(
        raw_weights: Any,
        *,
        allowed: set[str],
        field: str,
    ) -> dict[str, float]:
        if not isinstance(raw_weights, Mapping):
            raise CatalogValidationError(f"{field} must be a mapping")
        validated: dict[str, float] = {}
        for raw_key, raw_value in raw_weights.items():
            key = str(raw_key)
            if key not in allowed:
                raise CatalogValidationError(f"{field} contains unknown feature: {key}")
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise CatalogValidationError(f"{field}.{key} must be numeric")
            weight = float(raw_value)
            if not np.isfinite(weight) or not 0 <= weight <= 1:
                raise CatalogValidationError(f"{field}.{key} must be in [0, 1]")
            if weight > 0:
                validated[key] = weight
        return validated

    @staticmethod
    def _feature_label(feature_name: str) -> str:
        _, raw_name = feature_name.split(":", maxsplit=1)
        return raw_name.replace("_", " ").title()

    def _ensure_fitted(self) -> None:
        if (
            self.catalog is None
            or self.career_matrix is None
            or self._nn is None
            or self.model_version is None
        ):
            raise RuntimeError("The recommender must be fitted before use")
