"""Train and persist the content-based career recommender."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from career_recommender import CareerRecommender


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=PROJECT_ROOT / "data" / "career_profiles.json",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "career_recommender.joblib",
    )
    parser.add_argument(
        "--model-card",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "model_card.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recommender = CareerRecommender.from_catalog_file(args.catalog)
    recommender.save(args.model)

    args.model_card.parent.mkdir(parents=True, exist_ok=True)
    args.model_card.write_text(
        json.dumps(recommender.model_card(), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"model_version={recommender.model_version}")
    print(f"model={args.model}")
    print(f"model_card={args.model_card}")


if __name__ == "__main__":
    main()
