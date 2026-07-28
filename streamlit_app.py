"""Streamlit UI for the Student Learning Analytics Platform."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from datetime import date
from html import escape
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
PLATFORM_DATA_PATH = PROJECT_ROOT / "data" / "student_learning_analytics.json"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "career_recommender.joblib"
MODEL_CARD_PATH = PROJECT_ROOT / "artifacts" / "model_card.json"
DEMO_PROFILE_PATH = PROJECT_ROOT / "examples" / "student_profile.json"
BENCHMARK_PATH = PROJECT_ROOT / "examples" / "benchmark_profiles.json"


st.set_page_config(
    page_title="Student Learning Analytics",
    page_icon="\N{BAR CHART}",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_recommender() -> CareerRecommender:
    """Load the persisted recommender, training it when needed."""

    if MODEL_PATH.exists():
        return CareerRecommender.load(MODEL_PATH)

    model = CareerRecommender.from_catalog_file(CATALOG_PATH)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    MODEL_CARD_PATH.write_text(
        json.dumps(model.model_card(), indent=2) + "\n",
        encoding="utf-8",
    )
    return model


@st.cache_data(show_spinner=False)
def load_profiles() -> dict[str, dict[str, Any]]:
    """Load student evidence profiles used by the career model."""

    demo = json.loads(DEMO_PROFILE_PATH.read_text(encoding="utf-8"))
    benchmark_profiles = json.loads(
        BENCHMARK_PATH.read_text(encoding="utf-8")
    )["profiles"]

    profiles = {"Demo Student": demo}
    for profile in benchmark_profiles:
        expected = str(profile["expected_top_1"]).replace("_", " ").title()
        label = f"{expected} Archetype"
        profiles[label] = {
            key: value
            for key, value in profile.items()
            if key != "expected_top_1"
        }
    return profiles


@st.cache_data(show_spinner=False)
def load_platform_data() -> dict[str, Any]:
    """Load the assignment-scale deterministic dummy dataset."""

    if not PLATFORM_DATA_PATH.exists():
        raise FileNotFoundError(
            "Missing data/student_learning_analytics.json. "
            "Upload the supplied dataset file to the repository data folder."
        )
    return json.loads(PLATFORM_DATA_PATH.read_text(encoding="utf-8"))


def feature_label(raw_name: str) -> str:
    return raw_name.replace("_", " ").title()


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def streaks_from_daily_success(success_values: list[bool]) -> tuple[int, int]:
    """Return the current and best run of successful scheduled days."""

    best = 0
    running = 0
    for success in success_values:
        if success:
            running += 1
            best = max(best, running)
        else:
            running = 0

    current = 0
    for success in reversed(success_values):
        if not success:
            break
        current += 1
    return current, best


def prepare_student_analytics(
    *,
    dataset: dict[str, Any],
    student: dict[str, Any],
    profile: dict[str, Any],
    xp_rate: float,
) -> dict[str, Any]:
    """Join raw learning events and calculate explainable analytics."""

    session_by_id = {
        session["class_id"]: session
        for session in dataset["class_sessions"]
    }
    course_by_id = {
        course["course_id"]: course
        for course in dataset["courses"]
    }
    institute_by_id = {
        institute["institute_id"]: institute
        for institute in dataset["institutes"]
    }
    status_multiplier = {"Present": 1.0, "Late": 0.75, "Absent": 0.0}

    rows: list[dict[str, Any]] = []
    for record in dataset["attendance_records"]:
        if record["student_id"] != student["student_id"]:
            continue
        session = session_by_id[record["class_id"]]
        course = course_by_id[session["course_id"]]
        institute = institute_by_id[session["institute_id"]]
        xp = round(
            float(session["minutes"])
            * xp_rate
            * status_multiplier[record["status"]],
            2,
        )
        rows.append(
            {
                "Date": session["date"],
                "Class ID": session["class_id"],
                "Institute": institute["name"],
                "Course": course["title"],
                "Subject": course["subject"],
                "Minutes": session["minutes"],
                "Attendance": record["status"],
                "XP": xp,
                "Learning objective": session["learning_objective"],
                "Skill": session["primary_skill"],
                "Evidence score": record["evidence_score"],
            }
        )

    activity = pd.DataFrame(rows).sort_values(["Date", "Class ID"])
    status_counts = (
        activity["Attendance"]
        .value_counts()
        .reindex(["Present", "Late", "Absent"], fill_value=0)
    )
    scheduled = int(len(activity))
    present = int(status_counts["Present"])
    late = int(status_counts["Late"])
    absent = int(status_counts["Absent"])
    participation_rate = (
        100.0 * (present + late) / scheduled if scheduled else 0.0
    )
    attendance_score = (
        100.0 * (present + 0.75 * late) / scheduled if scheduled else 0.0
    )
    punctuality_rate = (
        100.0 * present / (present + late) if present + late else 0.0
    )

    daily_success = (
        activity.groupby("Date")["Attendance"]
        .apply(lambda statuses: bool((statuses != "Absent").any()))
        .tolist()
    )
    current_streak, best_streak = streaks_from_daily_success(daily_success)

    activity_with_week = activity.copy()
    activity_with_week["Week"] = (
        pd.to_datetime(activity_with_week["Date"])
        .dt.to_period("W")
        .apply(lambda period: period.start_time.date().isoformat())
    )
    activity_with_week["Attendance score"] = activity_with_week[
        "Attendance"
    ].map({"Present": 100.0, "Late": 75.0, "Absent": 0.0})
    weekly_attendance = (
        activity_with_week.groupby("Week", as_index=False)["Attendance score"]
        .mean()
        .sort_values("Week")
    )
    previous_window = weekly_attendance["Attendance score"].iloc[-8:-4]
    recent_window = weekly_attendance["Attendance score"].iloc[-4:]
    trend_delta = (
        float(recent_window.mean() - previous_window.mean())
        if len(previous_window) and len(recent_window)
        else 0.0
    )
    if trend_delta > 3:
        trend_label = "Improving"
    elif trend_delta < -3:
        trend_label = "Needs attention"
    else:
        trend_label = "Stable"

    subject_scores = {
        feature_label(name): float(score)
        for name, score in profile.get("subject_scores", {}).items()
    }
    skill_scores = {
        feature_label(name): float(score)
        for name, score in profile.get("skill_mastery", {}).items()
    }
    skill_evidence = {
        feature_label(name): int(count)
        for name, count in profile.get("skill_evidence_counts", {}).items()
    }
    eligible_skills = {
        name: score
        for name, score in skill_scores.items()
        if skill_evidence.get(name, 0) >= 3
    }
    ranked_subjects = sorted(
        subject_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )
    ranked_skills = sorted(
        eligible_skills.items(),
        key=lambda item: (-item[1], item[0]),
    )
    strongest_subject = ranked_subjects[0] if ranked_subjects else ("No data", 0)
    weakest_skill = ranked_skills[-1] if ranked_skills else ("No data", 0)
    strengths = ranked_skills[:3]

    mastery_score = safe_mean(list(eligible_skills.values()))
    consistency_score = min(current_streak / 5.0, 1.0) * 100.0
    overall_score = (
        0.55 * mastery_score
        + 0.25 * attendance_score
        + 0.20 * consistency_score
    )

    attended = activity[activity["Attendance"] != "Absent"]
    objective_progress = (
        attended.groupby(["Subject", "Learning objective"], as_index=False)
        .agg(
            Sessions=("Class ID", "count"),
            Progress=("Evidence score", "mean"),
        )
        .sort_values(["Subject", "Progress"], ascending=[True, False])
    )
    objective_progress["Progress"] = (
        objective_progress["Progress"].fillna(0).round(1)
    )

    enrollments = [
        enrollment
        for enrollment in dataset["enrollments"]
        if enrollment["student_id"] == student["student_id"]
    ]
    enrollment_details = []
    for enrollment in enrollments:
        course = course_by_id[enrollment["course_id"]]
        institute = institute_by_id[enrollment["institute_id"]]
        enrollment_details.append(
            {
                "Institute": institute["name"],
                "Course": course["title"],
                "Subject": course["subject"],
                "Status": enrollment["status"].title(),
            }
        )

    return {
        "activity": activity,
        "status_counts": status_counts,
        "scheduled": scheduled,
        "present": present,
        "late": late,
        "absent": absent,
        "participation_rate": participation_rate,
        "attendance_score": attendance_score,
        "punctuality_rate": punctuality_rate,
        "current_streak": current_streak,
        "best_streak": best_streak,
        "weekly_attendance": weekly_attendance,
        "trend_delta": trend_delta,
        "trend_label": trend_label,
        "subject_scores": subject_scores,
        "skill_scores": skill_scores,
        "skill_evidence": skill_evidence,
        "strongest_subject": strongest_subject,
        "weakest_skill": weakest_skill,
        "strengths": strengths,
        "mastery_score": mastery_score,
        "consistency_score": consistency_score,
        "overall_score": overall_score,
        "total_xp": float(activity["XP"].sum()),
        "objective_progress": objective_progress,
        "enrollments": enrollment_details,
    }


def render_page_header(
    student: dict[str, Any],
    page_name: str,
    dataset_version: str,
) -> None:
    with st.container(key="page_header"):
        st.markdown(
            "<div class='header-row'>"
            "<div>"
            "<div class='product-label'>Student Learning Analytics Platform</div>"
            f"<h1>{escape(page_name)}</h1>"
            f"<p>{escape(student['name'])} &middot; "
            f"{escape(student['grade'])} &middot; "
            f"{escape(student['student_id'])}</p>"
            "</div>"
            "<div class='header-badges'>"
            "<span>Live demo</span>"
            f"<span>{escape(dataset_version)}</span>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )


def render_section_intro(title: str, description: str) -> None:
    st.markdown(
        f"<div class='section-title'>{escape(title)}</div>"
        f"<div class='section-description'>{escape(description)}</div>",
        unsafe_allow_html=True,
    )


def render_overview(
    *,
    student: dict[str, Any],
    analytics: dict[str, Any],
) -> None:
    render_section_intro(
        "Student overview",
        "A single view of engagement, progress and the signals that need attention.",
    )

    xp_col, attendance_col, streak_col, score_col = st.columns(4)
    xp_col.metric("Total XP", f"{analytics['total_xp']:.1f}")
    attendance_col.metric(
        "Attendance score",
        f"{analytics['attendance_score']:.1f}%",
    )
    streak_col.metric(
        "Current streak",
        f"{analytics['current_streak']} days",
    )
    score_col.metric(
        "Overall score",
        f"{analytics['overall_score']:.1f}/100",
    )

    st.caption(
        "Overall score = 55% mastery + 25% attendance + 20% consistency."
    )

    enrollment_col, insight_col = st.columns([1, 1.35])
    with enrollment_col:
        st.subheader("Enrolled institutes")
        st.caption(
            "This demonstrates that one student can belong to multiple institutes."
        )
        for enrollment in analytics["enrollments"]:
            st.markdown(
                "<div class='simple-card'>"
                f"<strong>{escape(enrollment['Institute'])}</strong>"
                f"<span>{escape(enrollment['Course'])}</span>"
                f"<small>{escape(enrollment['Status'])}</small>"
                "</div>",
                unsafe_allow_html=True,
            )

    with insight_col:
        st.subheader("Generated insights")
        st.caption("Each insight is calculated from stored learning evidence.")
        insight_one, insight_two, insight_three = st.columns(3)
        with insight_one:
            st.markdown(
                "<div class='insight-card blue'>"
                "<span>Strongest subject</span>"
                f"<strong>{escape(analytics['strongest_subject'][0])}</strong>"
                f"<small>{analytics['strongest_subject'][1]:.0f}/100 mastery</small>"
                "</div>",
                unsafe_allow_html=True,
            )
        with insight_two:
            st.markdown(
                "<div class='insight-card amber'>"
                "<span>Skill to develop</span>"
                f"<strong>{escape(analytics['weakest_skill'][0])}</strong>"
                f"<small>{analytics['weakest_skill'][1]:.0f}/100 mastery</small>"
                "</div>",
                unsafe_allow_html=True,
            )
        with insight_three:
            delta = analytics["trend_delta"]
            st.markdown(
                "<div class='insight-card green'>"
                "<span>Attendance trend</span>"
                f"<strong>{escape(analytics['trend_label'])}</strong>"
                f"<small>{delta:+.1f} percentage points</small>"
                "</div>",
                unsafe_allow_html=True,
            )

    st.subheader("Recent learning activity")
    st.caption(
        "Every row connects a class to attendance, XP, an objective and a skill."
    )
    recent = analytics["activity"].tail(8).sort_values(
        ["Date", "Class ID"],
        ascending=False,
    )
    st.dataframe(
        recent[
            [
                "Date",
                "Course",
                "Attendance",
                "XP",
                "Learning objective",
                "Skill",
            ]
        ],
        hide_index=True,
        width="stretch",
    )


def render_attendance_and_xp(
    *,
    analytics: dict[str, Any],
    xp_rate: float,
) -> None:
    render_section_intro(
        "Attendance and XP",
        "Audit participation, punctuality and exactly how every XP value is calculated.",
    )

    st.markdown(
        "<div class='formula-card'>"
        "<strong>XP calculation</strong>"
        f"<code>awarded XP = class minutes × {xp_rate:.2f} × attendance multiplier</code>"
        "<span>Present = 1.00 &nbsp;·&nbsp; Late = 0.75 "
        "&nbsp;·&nbsp; Absent = 0.00</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    scheduled_col, present_col, late_col, absent_col = st.columns(4)
    scheduled_col.metric("Scheduled classes", analytics["scheduled"])
    present_col.metric("Present", analytics["present"])
    late_col.metric("Late", analytics["late"])
    absent_col.metric("Absent", analytics["absent"])

    attendance_chart_col, xp_chart_col = st.columns(2)
    with attendance_chart_col:
        st.subheader("Attendance distribution")
        attendance_chart = (
            analytics["status_counts"]
            .rename("Classes")
            .to_frame()
        )
        st.bar_chart(attendance_chart, color="#2878c7")
        st.caption(
            f"Participation: {analytics['participation_rate']:.1f}% · "
            f"Punctuality: {analytics['punctuality_rate']:.1f}%"
        )

    with xp_chart_col:
        st.subheader("XP by course")
        xp_by_course = (
            analytics["activity"]
            .groupby("Course")["XP"]
            .sum()
            .sort_values(ascending=False)
            .to_frame()
        )
        st.bar_chart(xp_by_course, color="#2b9d78")
        st.caption(
            "XP rewards attended learning time; it is not treated as mastery."
        )

    st.subheader("Weekly attendance trend")
    weekly_chart = analytics["weekly_attendance"].set_index("Week")
    st.line_chart(weekly_chart, color="#7d5fd0")

    with st.expander("How the learning streak works"):
        st.markdown(
            f"""
            A scheduled learning day is successful when at least one class is
            **Present** or **Late**. Days without a scheduled class do not break
            the streak. A day where every scheduled class is **Absent** does.

            - Current streak: **{analytics['current_streak']} days**
            - Best streak: **{analytics['best_streak']} days**
            """
        )

    with st.expander("View all attendance records"):
        st.dataframe(
            analytics["activity"][
                [
                    "Date",
                    "Class ID",
                    "Institute",
                    "Course",
                    "Minutes",
                    "Attendance",
                    "XP",
                ]
            ].sort_values(["Date", "Class ID"], ascending=False),
            hide_index=True,
            width="stretch",
        )


def render_learning_portfolio(
    *,
    student: dict[str, Any],
    analytics: dict[str, Any],
) -> None:
    render_section_intro(
        "Learning portfolio",
        "Turn class evidence into subject mastery, skill strengths and objective progress.",
    )

    subject_col, skills_col = st.columns(2)
    with subject_col:
        st.subheader("Subject performance")
        for subject, score in sorted(
            analytics["subject_scores"].items(),
            key=lambda item: (-item[1], item[0]),
        ):
            st.progress(
                min(max(score / 100.0, 0.0), 1.0),
                text=f"{subject}: {score:.0f}/100",
            )

    with skills_col:
        st.subheader("Demonstrated skills")
        eligible_skills = [
            (skill, score, analytics["skill_evidence"].get(skill, 0))
            for skill, score in analytics["skill_scores"].items()
            if analytics["skill_evidence"].get(skill, 0) >= 3
        ]
        for skill, score, evidence_count in sorted(
            eligible_skills,
            key=lambda item: (-item[1], item[0]),
        )[:8]:
            st.progress(
                min(max(score / 100.0, 0.0), 1.0),
                text=f"{skill}: {score:.0f}/100 · {evidence_count} observations",
            )

    st.subheader("Learning objectives")
    st.caption(
        "Progress is the average evidence score from attended classes mapped "
        "to each objective."
    )
    st.dataframe(
        analytics["objective_progress"],
        hide_index=True,
        width="stretch",
    )

    st.subheader("Generated student portfolio")
    strengths = [
        {"skill": skill, "mastery": round(score, 1)}
        for skill, score in analytics["strengths"]
    ]
    portfolio = {
        "student_id": student["student_id"],
        "student_name": student["name"],
        "grade": student["grade"],
        "institutes": analytics["enrollments"],
        "total_xp": round(analytics["total_xp"], 2),
        "attendance": {
            "participation_rate": round(
                analytics["participation_rate"],
                2,
            ),
            "attendance_score": round(
                analytics["attendance_score"],
                2,
            ),
            "punctuality_rate": round(
                analytics["punctuality_rate"],
                2,
            ),
        },
        "streaks": {
            "current": analytics["current_streak"],
            "best": analytics["best_streak"],
        },
        "strengths": strengths,
        "overall_learning_engagement_score": round(
            analytics["overall_score"],
            2,
        ),
    }
    summary_col, download_col = st.columns([2, 1])
    with summary_col:
        strength_text = ", ".join(item["skill"] for item in strengths)
        st.markdown(
            "<div class='portfolio-card'>"
            f"<strong>{escape(student['name'])}</strong>"
            f"<span>{analytics['overall_score']:.1f}/100 overall score</span>"
            f"<p>Top strengths: {escape(strength_text or 'Insufficient evidence')}</p>"
            f"<p>{analytics['total_xp']:.1f} XP · "
            f"{analytics['attendance_score']:.1f}% attendance score · "
            f"{analytics['current_streak']}-day current streak</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with download_col:
        st.download_button(
            "Download portfolio JSON",
            data=json.dumps(portfolio, indent=2),
            file_name=f"{student['student_id']}_portfolio.json",
            mime="application/json",
            width="stretch",
        )

    with st.expander("How the overall score is calculated"):
        st.markdown(
            f"""
            **Learning-engagement score**

            `0.55 × mastery + 0.25 × attendance + 0.20 × consistency`

            - Mastery component: **{analytics['mastery_score']:.1f}**
            - Attendance component: **{analytics['attendance_score']:.1f}**
            - Consistency component: **{analytics['consistency_score']:.1f}**

            Raw lifetime XP is not included because it would unfairly favour
            students with older enrollments.
            """
        )


def goal_current_value(metric: str, analytics: dict[str, Any]) -> float:
    values = {
        "cumulative_xp": analytics["total_xp"],
        "attendance_score": analytics["attendance_score"],
        "current_streak": float(analytics["current_streak"]),
        "overall_score": analytics["overall_score"],
    }
    return float(values[metric])


def goal_metric_label(metric: str) -> str:
    return {
        "cumulative_xp": "Cumulative XP",
        "attendance_score": "Attendance score",
        "current_streak": "Current streak",
        "overall_score": "Overall score",
    }[metric]


def render_parent_goals(
    *,
    student: dict[str, Any],
    dataset: dict[str, Any],
    analytics: dict[str, Any],
) -> None:
    render_section_intro(
        "Parent goals",
        "Create measurable goals and calculate progress from the same learning evidence.",
    )

    goal_key = f"goals_{student['student_id']}"
    if goal_key not in st.session_state:
        seeded_goals = [
            deepcopy(goal)
            for goal in dataset["parent_goals"]
            if goal["student_id"] == student["student_id"]
        ]
        st.session_state[goal_key] = seeded_goals

    create_col, progress_col = st.columns([1, 1.45])
    with create_col:
        st.subheader("Create a new goal")
        st.caption(
            f"Parent view for {student['parent_name']}. "
            "Goals are stored for this browser session."
        )
        with st.form(f"goal_form_{student['student_id']}"):
            title = st.text_input(
                "Goal title",
                value="Improve learning consistency",
            )
            metric = st.selectbox(
                "Metric",
                options=[
                    "cumulative_xp",
                    "attendance_score",
                    "current_streak",
                    "overall_score",
                ],
                format_func=goal_metric_label,
            )
            current = goal_current_value(metric, analytics)
            target = st.number_input(
                "Target value",
                min_value=1.0,
                value=float(max(round(current * 1.1, 1), 10.0)),
                step=1.0,
            )
            due_date = st.date_input(
                "Due date",
                value=date(2026, 8, 31),
            )
            submitted = st.form_submit_button(
                "Create goal",
                width="stretch",
            )
        if submitted:
            st.session_state[goal_key].append(
                {
                    "goal_id": f"SESSION-{len(st.session_state[goal_key]) + 1}",
                    "student_id": student["student_id"],
                    "created_by": student["parent_name"],
                    "title": title,
                    "metric": metric,
                    "baseline": current,
                    "target": float(target),
                    "due_date": due_date.isoformat(),
                }
            )
            st.success("Goal created. Progress will update from learning data.")

    with progress_col:
        st.subheader("Goal progress")
        if not st.session_state[goal_key]:
            st.info("No active goals yet.")
        for goal in st.session_state[goal_key]:
            current = goal_current_value(goal["metric"], analytics)
            baseline = float(goal["baseline"])
            target = float(goal["target"])
            denominator = target - baseline
            progress = (
                max(0.0, min((current - baseline) / denominator, 1.0))
                if denominator > 0
                else 1.0
            )
            status = "Achieved" if progress >= 1.0 else "Active"
            with st.container(border=True):
                title_col, status_col = st.columns([4, 1])
                title_col.markdown(f"**{goal['title']}**")
                status_col.markdown(f"`{status}`")
                st.progress(
                    progress,
                    text=(
                        f"{goal_metric_label(goal['metric'])}: "
                        f"{current:.1f} of {target:.1f}"
                    ),
                )
                st.caption(
                    f"Baseline {baseline:.1f} · Due {goal['due_date']} · "
                    f"Created by {goal['created_by']}"
                )


def recommendation_input_editor(
    *,
    model: CareerRecommender,
    base_profile: dict[str, Any],
    student_id: str,
) -> tuple[dict[str, Any], int]:
    """Allow evaluators to test the recommendation without cluttering the page."""

    profile = deepcopy(base_profile)
    profile["student_id"] = student_id
    top_k = st.slider(
        "Number of career paths",
        min_value=1,
        max_value=min(5, len(model.career_ids)),
        value=3,
    )

    assert model.catalog is not None
    schema = model.catalog["feature_schema"]
    signal_options = [
        *(f"Skill · {feature_label(name)}" for name in schema["skills"]),
        *(f"Subject · {feature_label(name)}" for name in schema["subjects"]),
    ]
    with st.expander("Test different learning evidence"):
        st.caption(
            "Optional evaluator control. Select only the signals you want to "
            "change; all other values continue to use the stored profile."
        )
        selected_signals = st.multiselect(
            "Signals to adjust",
            options=signal_options,
            default=[],
        )
        for signal in selected_signals:
            signal_type, label = signal.split(" · ", maxsplit=1)
            raw_name = label.lower().replace(" ", "_")
            score_col, evidence_col = st.columns([3, 1])
            if signal_type == "Skill":
                score_map = profile.setdefault("skill_mastery", {})
                evidence_map = profile.setdefault("skill_evidence_counts", {})
            else:
                score_map = profile.setdefault("subject_scores", {})
                evidence_map = profile.setdefault(
                    "subject_evidence_counts",
                    {},
                )
            with score_col:
                score_map[raw_name] = st.slider(
                    f"{label} mastery",
                    min_value=0,
                    max_value=100,
                    value=int(score_map.get(raw_name, 50)),
                    key=f"score_{student_id}_{signal_type}_{raw_name}",
                )
            with evidence_col:
                evidence_map[raw_name] = int(
                    st.number_input(
                        f"{label} observations",
                        min_value=0,
                        max_value=100,
                        value=int(evidence_map.get(raw_name, 0)),
                        key=f"evidence_{student_id}_{signal_type}_{raw_name}",
                        label_visibility="collapsed",
                    )
                )
    return profile, int(top_k)


def render_recommendation_card(item: dict[str, Any]) -> None:
    with st.container(
        border=True,
        key=f"recommendation_{item['rank']}",
    ):
        title_col, match_col = st.columns([4, 1])
        with title_col:
            st.markdown(
                f"<div class='rank-label'>#{item['rank']} recommended path</div>",
                unsafe_allow_html=True,
            )
            st.subheader(item["title"])
            st.caption(item["description"])
        with match_col:
            st.metric("Match", f"{item['match_score']:.1f}%")

        confidence_col, coverage_col, alignment_col = st.columns(3)
        confidence_col.metric(
            "Confidence",
            f"{item['confidence_score']:.1f}%",
        )
        coverage_col.metric(
            "Evidence coverage",
            f"{item['evidence_coverage']:.1f}%",
        )
        alignment_col.metric(
            "Strength alignment",
            f"{item['alignment_score']:.1f}%",
        )

        fit_col, development_col = st.columns(2)
        with fit_col:
            st.markdown("**Why this path is recommended**")
            for reason in item["reasons"]:
                st.markdown(
                    f"- {reason['feature']}: "
                    f"**{reason['student_score']:.0f}/100**"
                )
        with development_col:
            st.markdown("**What to develop next**")
            for gap in item["development_gaps"]:
                st.markdown(
                    f"- {gap['feature']}: "
                    f"**{gap['student_score']:.0f}/100**"
                )

        if item["missing_critical_evidence"]:
            st.info(
                "More evidence needed for: "
                + ", ".join(item["missing_critical_evidence"])
            )
        with st.expander("Why these scores were produced"):
            st.write(item["explanation"])
            st.markdown(
                "- **Match** combines profile alignment and proficiency.\n"
                "- **Confidence** reflects evidence coverage and repetition.\n"
                "- Match is not an employment probability."
            )


def render_career_recommendations(
    *,
    model: CareerRecommender,
    student: dict[str, Any],
    base_profile: dict[str, Any],
) -> None:
    render_section_intro(
        "Career recommendations",
        "Use demonstrated subjects and skills to rank explainable career paths.",
    )

    st.markdown(
        "<div class='process-row'>"
        "<div><b>1</b><span>Learning evidence</span></div>"
        "<i></i>"
        "<div><b>2</b><span>Cosine KNN matching</span></div>"
        "<i></i>"
        "<div><b>3</b><span>Evidence-aware ranking</span></div>"
        "<i></i>"
        "<div><b>4</b><span>Explainable paths</span></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    profile, top_k = recommendation_input_editor(
        model=model,
        base_profile=base_profile,
        student_id=student["student_id"],
    )
    try:
        result = model.recommend(profile, top_k=top_k)
    except InsufficientEvidenceError as exc:
        st.warning(
            "There is not enough eligible evidence for a responsible "
            f"recommendation. {exc}"
        )
        return
    except ProfileValidationError as exc:
        st.error(f"Profile validation failed: {exc}")
        return

    evidence_col, ignored_col, model_col = st.columns(3)
    evidence_col.metric(
        "Eligible signals",
        f"{result['eligible_feature_count']}/{result['total_feature_count']}",
    )
    ignored_col.metric(
        "Low-evidence signals",
        len(result["ignored_or_low_evidence_features"]),
    )
    model_col.metric(
        "Model version",
        str(result["model_version"]).split("-")[-1],
    )

    st.subheader("Ranked career paths")
    for item in result["recommendations"]:
        render_recommendation_card(item)

    st.caption(result["disclaimer"])
    st.download_button(
        "Download recommendation JSON",
        data=json.dumps(result, indent=2),
        file_name=f"{student['student_id']}_career_recommendation.json",
        mime="application/json",
    )


def render_system_and_data(
    *,
    dataset: dict[str, Any],
    model: CareerRecommender,
) -> None:
    render_section_intro(
        "System and data",
        "Evaluator-facing evidence showing how the UI maps to the assignment.",
    )

    counts = dataset["metadata"]["counts"]
    students_col, institutes_col, courses_col, classes_col = st.columns(4)
    students_col.metric("Students", counts["students"])
    institutes_col.metric("Institutes", counts["institutes"])
    courses_col.metric("Courses", counts["courses"])
    classes_col.metric("Class sessions", counts["class_sessions"])

    st.subheader("Functional requirement coverage")
    coverage = pd.DataFrame(
        [
            ["Multiple institutes", "Implemented", "Overview enrollment cards"],
            ["Present / Absent / Late", "Implemented", "Attendance records"],
            ["Configurable XP", "Implemented", "Sidebar XP rate and audit formula"],
            ["Objectives and skills", "Implemented", "Class-level evidence"],
            ["Student portfolio", "Implemented", "Downloadable portfolio JSON"],
            ["Learning streaks", "Implemented", "Scheduled-day streak logic"],
            ["Parent goals", "Implemented", "Create and monitor goals"],
            ["Generated insights", "Implemented", "Overview insight cards"],
            ["Career recommendations", "Implemented", "Explainable cosine KNN"],
        ],
        columns=["Requirement", "Status", "Where to test"],
    )
    st.dataframe(coverage, hide_index=True, width="stretch")

    st.subheader("Evidence-to-decision flow")
    st.markdown(
        "<div class='architecture-row'>"
        "<div><strong>1. Raw events</strong><span>Classes and attendance</span></div>"
        "<div><strong>2. Analytics</strong><span>XP, streaks and mastery</span></div>"
        "<div><strong>3. Portfolio</strong><span>Strengths and overall score</span></div>"
        "<div><strong>4. Decisions</strong><span>Goals, insights and careers</span></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Technical implementation status")
    st.warning(
        "This package contains the Streamlit UI, deterministic dummy dataset "
        "and ML recommender. A working database-backed REST API is still "
        "required to satisfy the complete assignment acceptance criteria."
    )
    technical_status = pd.DataFrame(
        [
            ["Python project structure", "Implemented"],
            ["ML recommendation algorithm", "Implemented"],
            ["10 students / 2 institutes / 2 courses / 100 classes", "Implemented"],
            ["Interactive Streamlit testing UI", "Implemented"],
            ["Database persistence", "Not included in this package"],
            ["Working REST API", "Not included in this package"],
        ],
        columns=["Technical area", "Status"],
    )
    st.dataframe(
        technical_status,
        hide_index=True,
        width="stretch",
    )

    download_col, model_col = st.columns(2)
    with download_col:
        st.download_button(
            "Download complete dummy dataset",
            data=json.dumps(dataset, indent=2),
            file_name="student_learning_analytics_dataset.json",
            mime="application/json",
            width="stretch",
        )
    with model_col:
        st.download_button(
            "Download ML model card",
            data=json.dumps(model.model_card(), indent=2),
            file_name="career_recommender_model_card.json",
            mime="application/json",
            width="stretch",
        )


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: light; }
        html, body, [data-testid="stAppViewContainer"], .stApp {
            background: #f4f7fb !important;
            color: #17263a !important;
        }
        [data-testid="stMainBlockContainer"] {
            max-width: 86rem;
            padding-top: 1.8rem;
            padding-bottom: 4rem;
        }
        [data-testid="stMain"] h1,
        [data-testid="stMain"] h2,
        [data-testid="stMain"] h3,
        [data-testid="stMain"] h4,
        [data-testid="stMain"] p,
        [data-testid="stMain"] li,
        [data-testid="stMain"] label,
        [data-testid="stMain"] [data-testid="stMarkdownContainer"],
        [data-testid="stMain"] [data-testid="stMetricLabel"],
        [data-testid="stMain"] [data-testid="stMetricValue"] {
            color: #17263a !important;
        }
        [data-testid="stMain"] [data-testid="stCaptionContainer"] {
            color: #627084 !important;
        }
        [data-testid="stSidebar"] {
            background: #102844 !important;
            border-right: 0;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            color: #f5f8fc !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="input"] > div {
            background: #ffffff !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] *,
        [data-testid="stSidebar"] [data-baseweb="input"] *,
        [data-testid="stSidebar"] input {
            color: #17263a !important;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label {
            background: transparent;
            border-radius: 0.55rem;
            padding: 0.35rem 0.45rem;
        }
        .brand {
            border-bottom: 1px solid rgba(255, 255, 255, 0.13);
            margin-bottom: 1rem;
            padding: 0.25rem 0 1rem;
        }
        .brand strong {
            color: #ffffff !important;
            display: block;
            font-size: 1.12rem;
        }
        .brand span {
            color: #b8c8da !important;
            display: block;
            font-size: 0.78rem;
            margin-top: 0.25rem;
        }
        .dataset-box {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 0.75rem;
            margin-top: 1rem;
            padding: 0.8rem;
        }
        .dataset-box strong,
        .dataset-box span {
            color: #ffffff !important;
            display: block;
        }
        .dataset-box strong { font-size: 0.78rem; }
        .dataset-box span {
            color: #c3d1df !important;
            font-size: 0.7rem;
            line-height: 1.6;
            margin-top: 0.3rem;
        }
        .st-key-page_header {
            background: #ffffff;
            border: 1px solid #dde5ef;
            border-radius: 1rem;
            box-shadow: 0 8px 24px rgba(26, 47, 72, 0.06);
            margin-bottom: 1.35rem;
            padding: 1.35rem 1.5rem;
        }
        .header-row {
            align-items: center;
            display: flex;
            justify-content: space-between;
        }
        .header-row h1 {
            color: #13263d !important;
            font-size: clamp(1.65rem, 2.6vw, 2.35rem);
            line-height: 1.15;
            margin: 0.2rem 0 0.35rem;
        }
        .header-row p {
            color: #627084 !important;
            margin: 0;
        }
        .product-label {
            color: #2878c7 !important;
            font-size: 0.73rem;
            font-weight: 800;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }
        .header-badges {
            display: flex;
            gap: 0.45rem;
        }
        .header-badges span {
            background: #edf5ff;
            border: 1px solid #d2e6fa;
            border-radius: 999px;
            color: #275f96 !important;
            font-size: 0.7rem;
            font-weight: 750;
            padding: 0.35rem 0.65rem;
        }
        .section-title {
            color: #13263d !important;
            font-size: 1.45rem;
            font-weight: 800;
            letter-spacing: -0.015em;
            margin-top: 0.35rem;
        }
        .section-description {
            color: #627084 !important;
            font-size: 0.92rem;
            margin: 0.25rem 0 1rem;
        }
        [data-testid="stMetric"] {
            background: #ffffff !important;
            border: 1px solid #dde5ef;
            border-radius: 0.85rem;
            min-height: 5.9rem;
            padding: 0.85rem 0.95rem;
        }
        [data-testid="stMetricLabel"] {
            color: #66768a !important;
            font-weight: 650;
        }
        [data-testid="stMetricValue"] {
            color: #163454 !important;
            font-size: clamp(1.35rem, 1.8vw, 1.9rem) !important;
            font-weight: 800;
        }
        .simple-card,
        .insight-card,
        .portfolio-card {
            background: #ffffff;
            border: 1px solid #dde5ef;
            border-radius: 0.8rem;
            margin-bottom: 0.7rem;
            padding: 0.9rem;
        }
        .simple-card strong,
        .simple-card span,
        .simple-card small {
            display: block;
        }
        .simple-card strong { color: #183653 !important; }
        .simple-card span {
            color: #536579 !important;
            font-size: 0.82rem;
            margin-top: 0.25rem;
        }
        .simple-card small {
            color: #218064 !important;
            font-weight: 700;
            margin-top: 0.35rem;
        }
        .insight-card {
            border-top: 0.28rem solid #2878c7;
            min-height: 7.7rem;
        }
        .insight-card.amber { border-top-color: #d89028; }
        .insight-card.green { border-top-color: #2b9d78; }
        .insight-card span,
        .insight-card strong,
        .insight-card small {
            display: block;
        }
        .insight-card span {
            color: #68778a !important;
            font-size: 0.68rem;
            font-weight: 750;
            text-transform: uppercase;
        }
        .insight-card strong {
            color: #173551 !important;
            font-size: 1rem;
            margin-top: 0.65rem;
        }
        .insight-card small {
            color: #68778a !important;
            margin-top: 0.25rem;
        }
        .formula-card {
            background: #eaf4ff;
            border: 1px solid #cfe4f8;
            border-left: 0.3rem solid #2878c7;
            border-radius: 0.75rem;
            margin-bottom: 1rem;
            padding: 1rem;
        }
        .formula-card strong,
        .formula-card code,
        .formula-card span {
            display: block;
        }
        .formula-card strong { color: #174d80 !important; }
        .formula-card code {
            background: #ffffff;
            border-radius: 0.35rem;
            color: #173551 !important;
            margin: 0.5rem 0;
            padding: 0.45rem;
        }
        .formula-card span {
            color: #526b83 !important;
            font-size: 0.78rem;
        }
        .portfolio-card {
            border-left: 0.3rem solid #2b9d78;
            padding: 1.1rem;
        }
        .portfolio-card strong,
        .portfolio-card span {
            display: block;
        }
        .portfolio-card strong {
            color: #173551 !important;
            font-size: 1.1rem;
        }
        .portfolio-card span {
            color: #21765c !important;
            font-weight: 750;
            margin-top: 0.25rem;
        }
        .portfolio-card p {
            color: #5a6c80 !important;
            font-size: 0.82rem;
            margin: 0.55rem 0 0;
        }
        .process-row,
        .architecture-row {
            align-items: stretch;
            display: flex;
            gap: 0.55rem;
            margin: 0.4rem 0 1.2rem;
        }
        .process-row > div,
        .architecture-row > div {
            background: #ffffff;
            border: 1px solid #dde5ef;
            border-radius: 0.75rem;
            flex: 1;
            padding: 0.8rem;
        }
        .process-row b {
            align-items: center;
            background: #2878c7;
            border-radius: 999px;
            color: #ffffff !important;
            display: inline-flex;
            height: 1.6rem;
            justify-content: center;
            margin-right: 0.35rem;
            width: 1.6rem;
        }
        .process-row span {
            color: #314a63 !important;
            font-size: 0.78rem;
            font-weight: 700;
        }
        .process-row i {
            align-self: center;
            background: #afbdca;
            height: 1px;
            width: 0.7rem;
        }
        .architecture-row strong,
        .architecture-row span {
            display: block;
        }
        .architecture-row strong {
            color: #173551 !important;
            font-size: 0.84rem;
        }
        .architecture-row span {
            color: #627084 !important;
            font-size: 0.73rem;
            margin-top: 0.35rem;
        }
        .rank-label {
            color: #2878c7 !important;
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        div[data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stExpander"] {
            background: #ffffff !important;
            border-color: #dde5ef !important;
            border-radius: 0.85rem !important;
        }
        [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] summary svg,
        [data-testid="stWidgetLabel"] p {
            color: #173551 !important;
            fill: #173551 !important;
        }
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            color: #f5f8fc !important;
        }
        [data-testid="stMain"] input {
            color: #17263a !important;
        }
        .stDownloadButton button,
        [data-testid="stFormSubmitButton"] button {
            background: #2878c7 !important;
            border: 1px solid #2878c7 !important;
            border-radius: 0.65rem !important;
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        .stDownloadButton button p,
        [data-testid="stFormSubmitButton"] button p {
            color: #ffffff !important;
        }
        @media (max-width: 760px) {
            [data-testid="stMainBlockContainer"] {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .header-row {
                align-items: flex-start;
                flex-direction: column;
                gap: 0.8rem;
            }
            .process-row,
            .architecture-row {
                flex-direction: column;
            }
            .process-row i { display: none; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_styles()
    try:
        dataset = load_platform_data()
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        st.error(str(exc))
        st.stop()

    model = load_recommender()
    profiles = load_profiles()
    students = dataset["students"]
    student_by_label = {
        f"{student['name']} · {student['grade']}": student
        for student in students
    }

    with st.sidebar:
        st.markdown(
            "<div class='brand'>"
            "<strong>Learning Analytics</strong>"
            "<span>Evidence-to-decision student platform</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        selected_label = st.selectbox(
            "Student",
            options=list(student_by_label),
            index=0,
        )
        navigation = st.radio(
            "Navigate",
            options=[
                "Overview",
                "Attendance & XP",
                "Learning Portfolio",
                "Parent Goals",
                "Career Recommendations",
                "System & Data",
            ],
        )
        st.divider()
        xp_rate = st.number_input(
            "XP per class minute",
            min_value=0.01,
            max_value=1.0,
            value=float(dataset["metadata"]["xp_policy"]["rate"]),
            step=0.01,
            help="Assignment default: class minutes × 0.1.",
        )
        counts = dataset["metadata"]["counts"]
        st.markdown(
            "<div class='dataset-box'>"
            "<strong>Dummy dataset loaded</strong>"
            f"<span>{counts['students']} students · "
            f"{counts['institutes']} institutes · "
            f"{counts['courses']} courses · "
            f"{counts['class_sessions']} classes</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    student = student_by_label[selected_label]
    profile_name = student["profile_preset"]
    base_profile = profiles.get(profile_name, profiles["Demo Student"])
    analytics = prepare_student_analytics(
        dataset=dataset,
        student=student,
        profile=base_profile,
        xp_rate=float(xp_rate),
    )

    render_page_header(
        student,
        navigation,
        dataset["metadata"]["dataset_version"],
    )
    if navigation == "Overview":
        render_overview(student=student, analytics=analytics)
    elif navigation == "Attendance & XP":
        render_attendance_and_xp(
            analytics=analytics,
            xp_rate=float(xp_rate),
        )
    elif navigation == "Learning Portfolio":
        render_learning_portfolio(
            student=student,
            analytics=analytics,
        )
    elif navigation == "Parent Goals":
        render_parent_goals(
            student=student,
            dataset=dataset,
            analytics=analytics,
        )
    elif navigation == "Career Recommendations":
        render_career_recommendations(
            model=model,
            student=student,
            base_profile=base_profile,
        )
    else:
        render_system_and_data(dataset=dataset, model=model)


if __name__ == "__main__":
    main()
