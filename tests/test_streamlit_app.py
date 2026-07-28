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
            "Student Career Pathfinder",
            app.title[0].value,
        )
        self.assertEqual(
            ["Data Scientist", "Software Engineer", "Data Analyst"],
            [item.value for item in app.subheader[:3]],
        )
        self.assertGreaterEqual(len(app.metric), 4)

    def test_selecting_teacher_profile_updates_ranking(self) -> None:
        app = AppTest.from_file(
            str(APP_PATH),
            default_timeout=30,
        ).run()
        app.selectbox[0].select("Teacher Educator Archetype").run()

        self.assertEqual([], list(app.exception))
        self.assertEqual("Teacher / Educator", app.subheader[0].value)


if __name__ == "__main__":
    unittest.main()
