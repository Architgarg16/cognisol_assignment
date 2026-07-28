from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from career_recommender import (
    CareerRecommender,
    InsufficientEvidenceError,
    ProfileValidationError,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CareerRecommenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = CareerRecommender.from_catalog_file(
            PROJECT_ROOT / "data" / "career_profiles.json"
        )
        cls.benchmarks = json.loads(
            (PROJECT_ROOT / "examples" / "benchmark_profiles.json").read_text(
                encoding="utf-8"
            )
        )["profiles"]

    def test_benchmark_profiles_rank_expected_career_first(self) -> None:
        for profile in self.benchmarks:
            with self.subTest(student_id=profile["student_id"]):
                result = self.model.recommend(profile, top_k=3)
                self.assertEqual(
                    profile["expected_top_1"],
                    result["recommendations"][0]["career_id"],
                )

    def test_result_has_explanations_and_bounded_scores(self) -> None:
        result = self.model.recommend(self.benchmarks[0], top_k=3)
        self.assertEqual(3, len(result["recommendations"]))
        for item in result["recommendations"]:
            self.assertTrue(item["reasons"])
            self.assertTrue(item["development_gaps"])
            self.assertGreaterEqual(item["match_score"], 0)
            self.assertLessEqual(item["match_score"], 100)
            self.assertGreaterEqual(item["confidence_score"], 0)
            self.assertLessEqual(item["confidence_score"], 100)

    def test_low_evidence_is_rejected(self) -> None:
        profile = {
            "student_id": "sparse",
            "skill_mastery": {
                "programming": 98,
                "problem_solving": 95,
            },
            "skill_evidence_counts": {
                "programming": 1,
                "problem_solving": 1,
            },
            "subject_scores": {"computer_science": 99},
            "subject_evidence_counts": {"computer_science": 1},
        }
        with self.assertRaises(InsufficientEvidenceError):
            self.model.recommend(profile)

    def test_sparse_valid_profile_has_lower_confidence(self) -> None:
        complete = self.benchmarks[1]
        sparse = {
            "student_id": "sparse-valid",
            "skill_mastery": {
                "problem_solving": 96,
                "programming": 98,
                "analytical_thinking": 82,
            },
            "skill_evidence_counts": {
                "problem_solving": 3,
                "programming": 3,
                "analytical_thinking": 3,
            },
            "subject_scores": {"computer_science": 98},
            "subject_evidence_counts": {"computer_science": 3},
        }
        complete_result = self.model.recommend(complete, top_k=1)["recommendations"][0]
        sparse_result = self.model.recommend(sparse, top_k=1)["recommendations"][0]
        self.assertEqual("software_engineer", sparse_result["career_id"])
        self.assertLess(
            sparse_result["confidence_score"],
            complete_result["confidence_score"],
        )

    def test_invalid_score_is_rejected(self) -> None:
        invalid = dict(self.benchmarks[0])
        invalid["skill_mastery"] = {"programming": 101}
        with self.assertRaises(ProfileValidationError):
            self.model.recommend(invalid)

    def test_unknown_features_are_reported_not_used(self) -> None:
        profile = json.loads(json.dumps(self.benchmarks[0]))
        profile["skill_mastery"]["astrology"] = 100
        profile["skill_evidence_counts"]["astrology"] = 20
        result = self.model.recommend(profile)
        self.assertIn(
            "skill:astrology (unknown feature)",
            result["ignored_or_low_evidence_features"],
        )

    def test_saved_model_reproduces_rankings(self) -> None:
        before = self.model.recommend(self.benchmarks[0], top_k=3)["recommendations"]
        before_ids = [item["career_id"] for item in before]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.joblib"
            self.model.save(path)
            loaded = CareerRecommender.load(path)
            after = loaded.recommend(self.benchmarks[0], top_k=3)["recommendations"]
        after_ids = [item["career_id"] for item in after]
        self.assertEqual(before_ids, after_ids)
        self.assertEqual(self.model.model_version, loaded.model_version)

    def test_model_version_is_deterministic(self) -> None:
        second = CareerRecommender.from_catalog_file(
            PROJECT_ROOT / "data" / "career_profiles.json"
        )
        self.assertEqual(self.model.model_version, second.model_version)


if __name__ == "__main__":
    unittest.main()
