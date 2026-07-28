"""PyCharm-ready entry point for the career recommender demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from career_recommender import CareerRecommender  # noqa: E402


DEFAULT_CATALOG = PROJECT_ROOT / "data" / "career_profiles.json"
DEFAULT_MODEL = PROJECT_ROOT / "artifacts" / "career_recommender.joblib"
DEFAULT_PROFILE = PROJECT_ROOT / "examples" / "student_profile.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "pycharm_demo_output.json"
DEFAULT_MODEL_CARD = PROJECT_ROOT / "artifacts" / "model_card.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train if necessary and run a career recommendation demo."
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE,
        help="Student profile JSON file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of recommendations to return.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain even when a model artifact already exists.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the complete JSON result.",
    )
    return parser.parse_args()


def load_or_train(*, retrain: bool) -> CareerRecommender:
    """Load the persisted model or create it from the career catalog."""

    if DEFAULT_MODEL.exists() and not retrain:
        print(f"Loading model: {DEFAULT_MODEL}")
        return CareerRecommender.load(DEFAULT_MODEL)

    print(f"Training from catalog: {DEFAULT_CATALOG}")
    model = CareerRecommender.from_catalog_file(DEFAULT_CATALOG)
    model.save(DEFAULT_MODEL)
    DEFAULT_MODEL_CARD.write_text(
        json.dumps(model.model_card(), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved model: {DEFAULT_MODEL}")
    return model


def print_summary(result: dict) -> None:
    """Print an interview-friendly summary to the Run console."""

    print()
    print("=" * 72)
    print("EXPLAINABLE ML CAREER RECOMMENDATIONS")
    print("=" * 72)
    print(f"Student: {result['student_id']}")
    print(f"Model:   {result['model_version']}")
    print(
        "Evidence: "
        f"{result['eligible_feature_count']}/"
        f"{result['total_feature_count']} eligible features"
    )
    print()

    for item in result["recommendations"]:
        print(
            f"{item['rank']}. {item['title']} | "
            f"match={item['match_score']:.2f} | "
            f"confidence={item['confidence_score']:.2f}"
        )
        strengths = ", ".join(reason["feature"] for reason in item["reasons"])
        gaps = ", ".join(gap["feature"] for gap in item["development_gaps"])
        print(f"   Evidence: {strengths}")
        print(f"   Develop:  {gaps}")
        print(f"   {item['explanation']}")
        print()

    print(result["disclaimer"])
    print("=" * 72)


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    if not args.profile.exists():
        raise FileNotFoundError(f"Profile not found: {args.profile}")

    model = load_or_train(retrain=args.retrain)
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    result = model.recommend(profile, top_k=args.top_k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print_summary(result)
    print(f"\nComplete JSON saved to: {args.output}")


if __name__ == "__main__":
    main()
