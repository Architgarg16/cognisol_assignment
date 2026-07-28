"""Generate career recommendations for one JSON student profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from career_recommender import CareerRecommender


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "career_recommender.joblib",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = CareerRecommender.load(args.model)
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    result = model.recommend(profile, top_k=args.top_k)
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
