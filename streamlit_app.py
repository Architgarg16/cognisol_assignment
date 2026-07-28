"""Streamlit UI for the explainable ML career recommender."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from career_recommender import (  # noqa: E402
    CareerRecommender,
    InsufficientEvidenceError,
    ProfileValidationError,
)


CATALOG_PATH = PROJECT_ROOT / "data" / "career_profiles.json"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "career_recommender.joblib"
MODEL_CARD_PATH = PROJECT_ROOT / "artifacts" / "model_card.json"
DEMO_PROFILE_PATH = PROJECT_ROOT / "examples" / "student_profile.json"
BENCHMARK_PATH = PROJECT_ROOT / "examples" / "benchmark_profiles.json"


st.set_page_config(
    page_title="Student Career Pathfinder",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_recommender() -> CareerRecommender:
    """Load the persisted model, training it when the artifact is absent."""

    if MODEL_PATH.exists():
        return CareerRecommender.load(MODEL_PATH)

    model = CareerRecommender.from_catalog_file(CATALOG_PATH)
    model.save(MODEL_PATH)
    MODEL_CARD_PATH.write_text(
        json.dumps(model.model_card(), indent=2) + "\n",
        encoding="utf-8",
    )
    return model


@st.cache_data(show_spinner=False)
def load_presets() -> dict[str, dict[str, Any]]:
    """Load the demo and synthetic benchmark profiles."""

    demo = json.loads(DEMO_PROFILE_PATH.read_text(encoding="utf-8"))
    benchmark_profiles = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))[
        "profiles"
    ]

    presets = {"Demo Student": demo}
    for profile in benchmark_profiles:
        expected = str(profile["expected_top_1"]).replace("_", " ").title()
        label = f"{expected} Archetype"
        clean_profile = {
            key: value for key, value in profile.items() if key != "expected_top_1"
        }
        presets[label] = clean_profile
    return presets


def feature_label(raw_name: str) -> str:
    return raw_name.replace("_", " ").title()


def profile_editor(
    *,
    model: CareerRecommender,
    selected_name: str,
    base_profile: dict[str, Any],
) -> dict[str, Any]:
    """Render score/evidence controls and return a validated input payload."""

    assert model.catalog is not None
    schema = model.catalog["feature_schema"]
    profile_key = selected_name.lower().replace(" ", "_").replace("/", "_")

    student_id = st.sidebar.text_input(
        "Student ID",
        value=str(base_profile.get("student_id", "student-demo")),
        key=f"{profile_key}_student_id",
    )
    top_k = st.sidebar.slider(
        "Number of career paths",
        min_value=1,
        max_value=min(5, len(model.career_ids)),
        value=3,
    )
    st.sidebar.caption(
        "Only features with at least three evidence observations are used."
    )

    skill_scores: dict[str, int] = {}
    skill_counts: dict[str, int] = {}
    subject_scores: dict[str, int] = {}
    subject_counts: dict[str, int] = {}

    with st.expander("Edit mastery and evidence", expanded=False):
        st.caption(
            "Scores describe current mastery. Evidence is the number of "
            "classes or assessments supporting each score."
        )
        skills_tab, subjects_tab = st.tabs(["Skills", "Subjects"])

        with skills_tab:
            st.markdown("#### Skill evidence")
            for skill in schema["skills"]:
                score_col, count_col = st.columns([3, 1])
                default_score = int(
                    base_profile.get("skill_mastery", {}).get(skill, 50)
                )
                default_count = int(
                    base_profile.get("skill_evidence_counts", {}).get(skill, 0)
                )
                with score_col:
                    skill_scores[skill] = st.slider(
                        feature_label(skill),
                        min_value=0,
                        max_value=100,
                        value=default_score,
                        key=f"{profile_key}_skill_score_{skill}",
                    )
                with count_col:
                    skill_counts[skill] = int(
                        st.number_input(
                            f"Evidence: {feature_label(skill)}",
                            min_value=0,
                            max_value=100,
                            value=default_count,
                            key=f"{profile_key}_skill_count_{skill}",
                            label_visibility="collapsed",
                        )
                    )

        with subjects_tab:
            st.markdown("#### Subject evidence")
            for subject in schema["subjects"]:
                score_col, count_col = st.columns([3, 1])
                default_score = int(
                    base_profile.get("subject_scores", {}).get(subject, 50)
                )
                default_count = int(
                    base_profile.get("subject_evidence_counts", {}).get(subject, 0)
                )
                with score_col:
                    subject_scores[subject] = st.slider(
                        feature_label(subject),
                        min_value=0,
                        max_value=100,
                        value=default_score,
                        key=f"{profile_key}_subject_score_{subject}",
                    )
                with count_col:
                    subject_counts[subject] = int(
                        st.number_input(
                            f"Evidence: {feature_label(subject)}",
                            min_value=0,
                            max_value=100,
                            value=default_count,
                            key=f"{profile_key}_subject_count_{subject}",
                            label_visibility="collapsed",
                        )
                    )

    return {
        "profile": {
            "student_id": student_id,
            "skill_mastery": skill_scores,
            "skill_evidence_counts": skill_counts,
            "subject_scores": subject_scores,
            "subject_evidence_counts": subject_counts,
        },
        "top_k": top_k,
    }


def render_recommendation_card(item: dict[str, Any]) -> None:
    """Render one explainable recommendation."""

    with st.container(border=True):
        heading_col, score_col = st.columns([4, 1])
        with heading_col:
            st.markdown(
                f"<span class='rank-chip'>#{item['rank']}</span>",
                unsafe_allow_html=True,
            )
            st.subheader(item["title"])
            st.caption(item["description"])
        with score_col:
            st.metric("Match", f"{item['match_score']:.1f}%")

        st.progress(
            min(max(float(item["match_score"]) / 100, 0.0), 1.0),
            text="Overall match",
        )

        confidence_col, alignment_col, coverage_col = st.columns(3)
        confidence_col.metric("Confidence", f"{item['confidence_score']:.1f}%")
        alignment_col.metric("Strength alignment", f"{item['alignment_score']:.1f}%")
        coverage_col.metric("Evidence coverage", f"{item['evidence_coverage']:.1f}%")

        strengths_col, gaps_col = st.columns(2)
        with strengths_col:
            st.markdown("**Why it fits**")
            for reason in item["reasons"]:
                st.markdown(
                    f"- **{reason['feature']}** — "
                    f"{reason['student_score']:.0f}/100 mastery"
                )
        with gaps_col:
            st.markdown("**Development opportunities**")
            for gap in item["development_gaps"]:
                st.markdown(
                    f"- **{gap['feature']}** — {gap['student_score']:.0f}/100 mastery"
                )

        if item["missing_critical_evidence"]:
            st.info(
                "Collect more evidence for: "
                + ", ".join(item["missing_critical_evidence"])
            )
        st.caption(item["explanation"])


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 85% 5%, rgba(26, 127, 100, 0.10), transparent 26rem),
                linear-gradient(180deg, #f8faf8 0%, #f4f2ed 100%);
        }
        [data-testid="stSidebar"] {
            background: #10231f;
        }
        [data-testid="stSidebar"] * {
            color: #f6f1e7;
        }
        [data-testid="stSidebar"] input {
            color: #10231f;
        }
        .eyebrow {
            color: #19795f;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }
        .hero-copy {
            color: #52605c;
            font-size: 1.05rem;
            max-width: 48rem;
            margin-top: -0.45rem;
        }
        .rank-chip {
            background: #dff4ea;
            border: 1px solid #a8d8c6;
            border-radius: 999px;
            color: #126149;
            display: inline-block;
            font-size: 0.75rem;
            font-weight: 800;
            padding: 0.22rem 0.6rem;
        }
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(35, 75, 63, 0.12);
            border-radius: 0.8rem;
            padding: 0.75rem;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.76);
            border-color: rgba(35, 75, 63, 0.14);
            box-shadow: 0 12px 32px rgba(31, 53, 45, 0.06);
        }
        .footer-note {
            border-top: 1px solid rgba(35, 75, 63, 0.14);
            color: #66736f;
            font-size: 0.82rem;
            margin-top: 2rem;
            padding-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_styles()
    model = load_recommender()
    presets = load_presets()

    with st.sidebar:
        st.markdown("## 🧭 Pathfinder")
        st.caption("Explainable ML career discovery")
        selected_name = st.selectbox(
            "Student profile",
            options=list(presets),
            index=0,
        )
        st.divider()
        st.markdown("**Privacy by design**")
        st.caption(
            "The model uses demonstrated skills and subjects only. Protected "
            "and demographic attributes are excluded."
        )

    editor = profile_editor(
        model=model,
        selected_name=selected_name,
        base_profile=presets[selected_name],
    )

    st.markdown(
        "<div class='eyebrow'>Student learning analytics</div>",
        unsafe_allow_html=True,
    )
    st.title("Student Career Pathfinder")
    st.markdown(
        "<p class='hero-copy'>Explore career paths using demonstrated "
        "mastery, subject performance, and the strength of the supporting "
        "evidence.</p>",
        unsafe_allow_html=True,
    )

    try:
        result = model.recommend(
            editor["profile"],
            top_k=int(editor["top_k"]),
        )
    except InsufficientEvidenceError as exc:
        st.warning(
            "There is not enough eligible evidence to produce a responsible "
            f"recommendation. {exc}"
        )
        st.info(
            "Open **Edit mastery and evidence** and provide at least three "
            "observations for several relevant skills or subjects."
        )
        return
    except ProfileValidationError as exc:
        st.error(f"Profile validation failed: {exc}")
        return

    top = result["recommendations"][0]
    eligible_col, top_col, confidence_col, version_col = st.columns(4)
    eligible_col.metric(
        "Eligible evidence",
        f"{result['eligible_feature_count']}/{result['total_feature_count']}",
    )
    top_col.metric("Leading path", top["title"])
    confidence_col.metric("Top confidence", f"{top['confidence_score']:.1f}%")
    version_col.metric(
        "Model",
        str(result["model_version"]).split("-")[-1],
    )

    st.markdown("### Ranked career paths")
    for recommendation in result["recommendations"]:
        render_recommendation_card(recommendation)

    st.markdown("### Compare paths")
    comparison = pd.DataFrame(
        [
            {
                "Career": item["title"],
                "Match": item["match_score"],
                "Confidence": item["confidence_score"],
            }
            for item in result["recommendations"]
        ]
    ).set_index("Career")
    st.bar_chart(
        comparison,
        color=["#19795f", "#d59644"],
        horizontal=True,
    )

    details_col, download_col = st.columns([3, 2])
    with details_col:
        with st.expander("How the model works"):
            st.markdown(
                """
                1. Mastery and subject scores form a student feature vector.
                2. Cosine-distance nearest neighbors finds aligned career profiles.
                3. The reranker combines strength alignment and absolute proficiency.
                4. Sparse evidence lowers confidence and can suppress a career.

                **Match is not probability.** It measures profile alignment.
                **Confidence measures evidence coverage and repetition.**
                """
            )
        with st.expander("Technical diagnostics"):
            if result["ignored_or_low_evidence_features"]:
                st.write(
                    "Ignored or low-evidence features:",
                    result["ignored_or_low_evidence_features"],
                )
            else:
                st.success("No supplied features were ignored.")
            st.json(model.model_card())
    with download_col:
        st.markdown("#### Export")
        st.caption(
            "Download the complete recommendation, evidence, model version, "
            "and disclaimer."
        )
        st.download_button(
            "Download recommendation JSON",
            data=json.dumps(result, indent=2),
            file_name=f"{result['student_id']}_career_recommendation.json",
            mime="application/json",
            use_container_width=True,
        )
        with st.expander("View raw result"):
            st.json(result)

    st.markdown(
        f"<div class='footer-note'>{result['disclaimer']}<br>"
        "Career profiles and synthetic evaluation require expert review "
        "before real educational use.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
