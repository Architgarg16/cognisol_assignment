# Explainable ML Career Recommender

This component implements the career recommendation requirement for the Student Learning Analytics Platform.

It is a content-based machine-learning recommender:

1. Student mastery and subject performance are converted into a normalized feature vector.
2. Career definitions are converted into vectors in the same feature space.
3. scikit-learn cosine-distance nearest neighbors generates candidate careers.
4. An evidence-aware reranker combines alignment, absolute proficiency, coverage, and evidence reliability.
5. Every recommendation includes supporting strengths, development gaps, missing evidence, and confidence.

The model does not use name, institute, age, gender, ethnicity, disability, economic status, or parent identity.

## PyCharm quick start

Open this folder as a PyCharm project, run `setup_pycharm.ps1` from PyCharm's Terminal, select `.venv\Scripts\python.exe` as the project interpreter, and run the shared **Web UI - Streamlit** configuration.

See [`PYCHARM_SETUP.md`](PYCHARM_SETUP.md) for exact instructions and the included Web UI, Demo, Evaluate, and Test configurations.

## Streamlit web UI

The interactive dashboard provides:

- preset student archetypes
- editable mastery and evidence inputs
- top one to five ranked career paths
- match, confidence, alignment, and coverage metrics
- strength explanations and development gaps
- missing-evidence warnings
- career comparison chart
- model diagnostics and JSON download

Run outside PyCharm with:

```bash
streamlit run streamlit_app.py
```

## Why this ML design

The assignment provides no historical student-career outcomes, interaction logs, or labels. Training a classifier or collaborative-filtering model on invented labels would give misleading accuracy.

Content-based nearest-neighbor ranking is the defensible ML choice because it:

- works during cold start
- needs no fabricated career outcome labels
- is deterministic and easy to explain
- can be evaluated with ranking and stability checks
- can later be replaced or blended with outcome-based learning when genuine data exists

This is not a probability-of-success model. It produces exploratory educational suggestions.

## Ranking contract

For each career, the model calculates:

```text
alignment = cosine_similarity(student_features, career_features)

proficiency =
    weighted_mean(student_scores, career_feature_importance)

coverage =
    observed_career_importance / total_career_importance

reliability =
    weighted_mean(min(evidence_count / 10, 1))

match_score =
    100
    * (0.65 * alignment + 0.35 * proficiency)
    * (0.70 + 0.30 * coverage)

confidence_score = 100 * coverage * reliability
```

Only observations supported by at least three evidence events are eligible. A career needs at least 35% feature coverage to be ranked.

`match_score` and `confidence_score` answer different questions:

- Match: how well the demonstrated strengths align with the career profile.
- Confidence: how much sufficiently repeated evidence supports that result.

## Repository layout

```text
.
|-- artifacts/
|   |-- career_recommender.joblib
|   |-- evaluation.json
|   `-- model_card.json
|-- data/career_profiles.json
|-- examples/
|   |-- benchmark_profiles.json
|   `-- student_profile.json
|-- scripts/
|   |-- evaluate_model.py
|   |-- recommend.py
|   `-- train_model.py
|-- src/career_recommender/
|   |-- __init__.py
|   `-- recommender.py
|-- tests/
|   |-- test_recommender.py
|   `-- test_streamlit_app.py
|-- main.py
|-- run_streamlit.py
|-- streamlit_app.py
|-- requirements.txt
`-- pyproject.toml
```

## Setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Train

```bash
python scripts/train_model.py
```

This validates the catalog and writes:

- `artifacts/career_recommender.joblib`
- `artifacts/model_card.json`

Joblib artifacts use Python pickle internally. Load only trusted artifacts created by this project.

## Evaluate

```bash
python scripts/evaluate_model.py --noise-runs 100 --noise-std 3
```

The checked-in evaluation currently reports:

- 8 benchmark profiles
- top-1 ranking sanity accuracy: 100%
- top-3 recall: 100%
- top-1 stability under 800 Gaussian score perturbations: 95%

These are synthetic, catalog-aligned sanity checks. They demonstrate implementation correctness and local stability, not real-world career validity.

## Recommend

```bash
python scripts/recommend.py examples/student_profile.json ^
  --output artifacts/sample_recommendation.json
```

On macOS/Linux, use `\` instead of `^` for line continuation.

Programmatic use:

```python
import json

from career_recommender import CareerRecommender

model = CareerRecommender.load("artifacts/career_recommender.joblib")
with open("examples/student_profile.json", encoding="utf-8") as handle:
    profile = json.load(handle)

result = model.recommend(profile, top_k=3)
```

## Input contract

```json
{
  "student_id": "student-123",
  "skill_mastery": {
    "programming": 90,
    "problem_solving": 86
  },
  "skill_evidence_counts": {
    "programming": 12,
    "problem_solving": 9
  },
  "subject_scores": {
    "computer_science": 92
  },
  "subject_evidence_counts": {
    "computer_science": 10
  }
}
```

Scores must be in `[0, 100]`; counts must be non-negative integers.

## FastAPI integration

Create one application-scoped model during startup, not once per request:

```python
from career_recommender import (
    CareerRecommender,
    InsufficientEvidenceError,
    ProfileValidationError,
)

recommender = CareerRecommender.load(
    "artifacts/career_recommender.joblib"
)


def recommend_for_student(analytics_profile: dict) -> dict:
    return recommender.recommend(analytics_profile, top_k=3)
```

The intended endpoint is:

```text
GET /api/v1/students/{student_id}/career-recommendations
```

The service layer should aggregate skill mastery, evidence counts, and subject scores from MongoDB, construct the input profile, and invoke the model. Map `InsufficientEvidenceError` to a normal `insufficient_data` response instead of an HTTP 500.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite verifies:

- all eight archetypes rank their expected career first
- match and confidence scores stay in range
- explanations and gaps are present
- low-evidence profiles are rejected
- sparse profiles receive lower confidence
- invalid scores are rejected
- unknown features are reported and ignored
- saved and loaded models reproduce rankings
- catalog-derived model versions are deterministic
- the default Streamlit dashboard renders without exceptions
- changing the selected student updates the leading recommendation

## Updating the catalog

Edit `data/career_profiles.json`, obtain domain-expert review, retrain, rerun evaluation, and commit the new model card and evaluation report together.

The model version contains the first 12 characters of the canonical catalog SHA-256 digest. A catalog change therefore creates a new model version automatically.

## Limitations and production roadmap

- Career profiles are curated and require career-guidance expert review.
- Synthetic benchmark accuracy is not evidence of real-world effectiveness.
- Student mastery scores must themselves be reliable and unbiased.
- Confidence measures evidence quantity and coverage, not certainty of success.
- The system should show several paths, never one deterministic verdict.
- Real deployment needs monitoring for selection rate, low-confidence rate, ranking drift, and subgroup fairness where lawful, consented demographic audit data exists.
- A future hybrid model can blend this content score with anonymized outcome or preference data after sufficient real observations are collected.
