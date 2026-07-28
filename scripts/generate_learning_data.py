"""Generate the deterministic dummy dataset used by the Streamlit demo."""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any


STUDENTS = [
    {
        "student_id": "STU-001",
        "name": "Aarav Sharma",
        "grade": "Grade 10",
        "parent_name": "Neha Sharma",
        "profile_preset": "Demo Student",
    },
    {
        "student_id": "STU-002",
        "name": "Ishita Verma",
        "grade": "Grade 11",
        "parent_name": "Rohit Verma",
        "profile_preset": "Data Scientist Archetype",
    },
    {
        "student_id": "STU-003",
        "name": "Kabir Singh",
        "grade": "Grade 10",
        "parent_name": "Anita Singh",
        "profile_preset": "Software Engineer Archetype",
    },
    {
        "student_id": "STU-004",
        "name": "Meera Iyer",
        "grade": "Grade 11",
        "parent_name": "Lakshmi Iyer",
        "profile_preset": "Teacher Educator Archetype",
    },
    {
        "student_id": "STU-005",
        "name": "Riya Patel",
        "grade": "Grade 9",
        "parent_name": "Jay Patel",
        "profile_preset": "Data Analyst Archetype",
    },
    {
        "student_id": "STU-006",
        "name": "Arjun Rao",
        "grade": "Grade 12",
        "parent_name": "Kavita Rao",
        "profile_preset": "Financial Analyst Archetype",
    },
    {
        "student_id": "STU-007",
        "name": "Sana Khan",
        "grade": "Grade 10",
        "parent_name": "Farah Khan",
        "profile_preset": "Technical Writer Editor Archetype",
    },
    {
        "student_id": "STU-008",
        "name": "Dev Malhotra",
        "grade": "Grade 11",
        "parent_name": "Pooja Malhotra",
        "profile_preset": "Ux Researcher Archetype",
    },
    {
        "student_id": "STU-009",
        "name": "Anaya Bose",
        "grade": "Grade 9",
        "parent_name": "Sanjay Bose",
        "profile_preset": "Product Manager Archetype",
    },
    {
        "student_id": "STU-010",
        "name": "Vivaan Das",
        "grade": "Grade 12",
        "parent_name": "Maya Das",
        "profile_preset": "Demo Student",
    },
]

INSTITUTES = [
    {
        "institute_id": "INS-001",
        "name": "Northstar Mathematics Institute",
        "city": "Bengaluru",
    },
    {
        "institute_id": "INS-002",
        "name": "BrightPath English Academy",
        "city": "Bengaluru",
    },
]

COURSES = [
    {
        "course_id": "CRS-MATH",
        "institute_id": "INS-001",
        "title": "Applied Mathematics",
        "subject": "Mathematics",
        "objectives": [
            "Model a real-world problem using equations",
            "Interpret patterns in statistical data",
            "Explain a multi-step quantitative solution",
            "Validate a solution using an alternative method",
        ],
        "skills": [
            "Quantitative Reasoning",
            "Problem Solving",
            "Data Analysis",
            "Attention To Detail",
        ],
    },
    {
        "course_id": "CRS-ENG",
        "institute_id": "INS-002",
        "title": "Academic English",
        "subject": "English",
        "objectives": [
            "Build a clear evidence-based argument",
            "Summarize a complex text accurately",
            "Present an idea to a peer audience",
            "Revise writing using structured feedback",
        ],
        "skills": [
            "Communication",
            "Writing",
            "Research",
            "Creativity",
        ],
    },
]


def build_dataset() -> dict[str, Any]:
    """Return a deterministic assignment-scale dummy dataset."""

    xp_rate = 0.1
    status_multipliers = {"Present": 1.0, "Late": 0.75, "Absent": 0.0}
    start = date(2026, 4, 6)

    class_sessions: list[dict[str, Any]] = []
    attendance_records: list[dict[str, Any]] = []
    for class_index in range(100):
        course = COURSES[class_index % len(COURSES)]
        session_number = class_index + 1
        session_date = start + timedelta(days=class_index // 2)
        minutes = 60 if course["course_id"] == "CRS-MATH" else 75
        objective = course["objectives"][
            (class_index // len(COURSES)) % len(course["objectives"])
        ]
        skill = course["skills"][
            (class_index // len(COURSES)) % len(course["skills"])
        ]
        class_id = f"CLS-{session_number:03d}"
        class_sessions.append(
            {
                "class_id": class_id,
                "course_id": course["course_id"],
                "institute_id": course["institute_id"],
                "date": session_date.isoformat(),
                "minutes": minutes,
                "learning_objective": objective,
                "primary_skill": skill,
            }
        )

        for student_index, student in enumerate(STUDENTS):
            learning_day = class_index // len(COURSES)
            pattern = (learning_day * 7 + student_index * 3) % 20
            if pattern < 16:
                status = "Present"
            elif pattern < 18:
                status = "Late"
            else:
                status = "Absent"

            evidence_score = None
            if status != "Absent":
                evidence_score = min(
                    98,
                    64 + ((class_index * 3 + student_index * 5) % 33),
                )
            awarded_xp = round(
                minutes * xp_rate * status_multipliers[status],
                2,
            )
            attendance_records.append(
                {
                    "student_id": student["student_id"],
                    "class_id": class_id,
                    "status": status,
                    "evidence_score": evidence_score,
                    "awarded_xp": awarded_xp,
                }
            )

    enrollments = [
        {
            "student_id": student["student_id"],
            "institute_id": institute["institute_id"],
            "course_id": course["course_id"],
            "status": "active",
        }
        for student in STUDENTS
        for institute, course in zip(INSTITUTES, COURSES)
    ]

    parent_goals = [
        {
            "goal_id": f"GOAL-{index + 1:03d}",
            "student_id": student["student_id"],
            "created_by": student["parent_name"],
            "title": "Reach 550 learning XP",
            "metric": "cumulative_xp",
            "baseline": 0,
            "target": 550,
            "due_date": "2026-08-31",
        }
        for index, student in enumerate(STUDENTS)
    ]

    return {
        "metadata": {
            "dataset_version": "learning-demo-v1",
            "description": "Deterministic synthetic data for assignment testing.",
            "xp_policy": {
                "rate": xp_rate,
                "status_multipliers": status_multipliers,
            },
            "counts": {
                "students": len(STUDENTS),
                "institutes": len(INSTITUTES),
                "courses": len(COURSES),
                "class_sessions": len(class_sessions),
                "attendance_records": len(attendance_records),
            },
        },
        "students": STUDENTS,
        "institutes": INSTITUTES,
        "courses": COURSES,
        "enrollments": enrollments,
        "class_sessions": class_sessions,
        "attendance_records": attendance_records,
        "parent_goals": parent_goals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/student_learning_analytics.json"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(build_dataset(), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
