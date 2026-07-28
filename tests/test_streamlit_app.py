from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "streamlit_app.py"


class StreamlitAppTests(unittest.TestCase):
    def test_default_dashboard_renders_without_exceptions(self) -> None:
        app = AppTest.from_file(
            str(APP_PATH),
            default_timeout=30,
        ).run()

        self.assertEqual([], list(app.exception))
        self.assertEqual(
            "Aarav Sharma · Grade 10",
            app.selectbox[0].value,
        )
        self.assertIn(
            "Enrolled institutes",
            [item.value for item in app.subheader],
        )
        self.assertGreaterEqual(len(app.metric), 4)

    def test_teacher_student_updates_career_ranking(self) -> None:
        app = AppTest.from_file(
            str(APP_PATH),
            default_timeout=30,
        ).run()
        app.selectbox[0].select("Meera Iyer · Grade 11").run()
        app.radio[0].set_value("Career Recommendations").run()

        self.assertEqual([], list(app.exception))
        career_headings = [item.value for item in app.subheader]
        self.assertIn("Ranked career paths", career_headings)
        self.assertEqual("Teacher / Educator", career_headings[1])

    def test_every_assignment_view_renders_without_exceptions(self) -> None:
        app = AppTest.from_file(
            str(APP_PATH),
            default_timeout=30,
        ).run()

        for page in [
            "Attendance & XP",
            "Learning Portfolio",
            "Parent Goals",
            "Career Recommendations",
            "System & Data",
        ]:
            app.radio[0].set_value(page).run()
            self.assertEqual([], list(app.exception), page)


if __name__ == "__main__":
    unittest.main()
