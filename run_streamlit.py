"""Portable PyCharm launcher for the Streamlit UI."""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.web import cli as streamlit_cli


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    sys.argv = [
        "streamlit",
        "run",
        str(PROJECT_ROOT / "streamlit_app.py"),
        "--server.headless=false",
        "--browser.gatherUsageStats=false",
    ]
    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    main()
