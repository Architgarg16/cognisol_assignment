"""Evaluate ranking sanity and perturbation stability on benchmark profiles."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from career_recommender import CareerRecommender


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "career_recommender.joblib",
    )
    parser.add_argument(
        "--benchmarks",
        type=Path,
        default=PROJECT_ROOT / "examples" / "benchmark_profiles.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "evaluation.json",
    )
    parser.add_argument("--noise-runs", type=int, default=50)
    parser.add_argument("--noise-std", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def add_noise(
    profile: dict[str, Any],
    *,
    rng: np.random.Generator,
    standard_deviation: float,
) -> dict[str, Any]:
    perturbed = copy.deepcopy(profile)
    for field in ("skill_mastery", "subject_scores"):
        perturbed[field] = {
            key: round(
                float(
                    np.clip(
                        value + rng.normal(0, standard_deviation),
                        0,
                        100,
                    )
                ),
                4,
            )
            for key, value in perturbed.get(field, {}).items()
        }
    return perturbed


def main() -> None:
    args = parse_args()
    if args.noise_runs < 1:
        raise ValueError("--noise-runs must be positive")
    if args.noise_std < 0:
        raise ValueError("--noise-std cannot be negative")

    recommender = CareerRecommender.load(args.model)
    benchmarks = json.loads(args.benchmarks.read_text(encoding="utf-8"))["profiles"]
    rng = np.random.default_rng(args.seed)

    rows: list[dict[str, Any]] = []
    top_1_hits = 0
    top_3_hits = 0
    stable_runs = 0
    total_noise_runs = len(benchmarks) * args.noise_runs

    for profile in benchmarks:
        expected = profile["expected_top_1"]
        prediction = recommender.recommend(profile, top_k=3)
        ranked = [item["career_id"] for item in prediction["recommendations"]]
        top_1_hits += int(ranked[0] == expected)
        top_3_hits += int(expected in ranked)

        for _ in range(args.noise_runs):
            noisy = add_noise(
                profile,
                rng=rng,
                standard_deviation=args.noise_std,
            )
            noisy_top = recommender.recommend(noisy, top_k=1)["recommendations"][0][
                "career_id"
            ]
            stable_runs += int(noisy_top == ranked[0])

        rows.append(
            {
                "student_id": profile["student_id"],
                "expected_top_1": expected,
                "predicted_top_3": ranked,
                "top_1_match": ranked[0] == expected,
                "expected_in_top_3": expected in ranked,
            }
        )

    sample_count = len(benchmarks)
    report = {
        "model_version": recommender.model_version,
        "evaluation_type": "synthetic ranking sanity check",
        "benchmark_profile_count": sample_count,
        "top_1_accuracy": round(top_1_hits / sample_count, 4),
        "top_3_recall": round(top_3_hits / sample_count, 4),
        "noise_stability": round(stable_runs / total_noise_runs, 4),
        "noise_configuration": {
            "runs_per_profile": args.noise_runs,
            "score_standard_deviation": args.noise_std,
            "random_seed": args.seed,
        },
        "profiles": rows,
        "limitations": (
            "The benchmark is synthetic and catalog-aligned. These metrics "
            "verify implementation behavior, not real-world career outcomes."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
