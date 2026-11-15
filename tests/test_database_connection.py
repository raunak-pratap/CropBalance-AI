"""Regression tests for lazy database connection initialization."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_agent_import_does_not_require_database_url():
    """Importing the agent remains safe without configured database access."""
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.pop("DATABASE_URL", None)
    environment.pop("TEST_DATABASE_URL", None)

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from agent.agent import CropBalanceAgent\n"
                "from database.connection import SessionLocal\n"
                "try:\n"
                "    SessionLocal()\n"
                "except RuntimeError as exc:\n"
                "    assert 'DATABASE_URL is not set' in str(exc)\n"
                "else:\n"
                "    raise AssertionError('A database session unexpectedly opened')\n"
            ),
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        "Agent/database import unexpectedly required a database URL.\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
