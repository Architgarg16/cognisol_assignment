from __future__ import annotations

import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "student_learning_analytics.json"


class LearningDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    def test_assignment_minimum_dataset_scale(self) -> None:
        counts = self.dataset["metadata"]["counts"]
        self.assertGreaterEqual(counts["students"], 10)
        self.assertGreaterEqual(counts["institutes"], 2)
        self.assertGreaterEqual(counts["courses"], 2)
        self.assertGreaterEqual(counts["class_sessions"], 100)

    def test_attendance_covers_every_status(self) -> None:
        statuses = {
            record["status"]
            for record in self.dataset["attendance_records"]
        }
        self.assertEqual({"Present", "Absent", "Late"}, statuses)

    def test_every_student_has_multiple_institutes(self) -> None:
        institute_ids_by_student: dict[str, set[str]] = {}
        for enrollment in self.dataset["enrollments"]:
            institute_ids_by_student.setdefault(
                enrollment["student_id"],
                set(),
            ).add(enrollment["institute_id"])

        self.assertTrue(institute_ids_by_student)
        self.assertTrue(
            all(
                len(institute_ids) >= 2
                for institute_ids in institute_ids_by_student.values()
            )
        )


if __name__ == "__main__":
    unittest.main()

