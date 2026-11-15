"""Shared safeguards for PostgreSQL integration tests.

These tests never fall back to DATABASE_URL. They require a separately
configured TEST_DATABASE_URL and additionally require the database name to
contain "test" as a guard against accidental use of an application database.
"""
from __future__ import annotations

import os

import pytest
from sqlalchemy.engine import make_url

from database.connection import dispose_engine


@pytest.fixture(autouse=True)
def isolated_postgres_test_database(monkeypatch: pytest.MonkeyPatch):
    test_database_url = os.getenv("TEST_DATABASE_URL")
    if not test_database_url:
        pytest.skip(
            "PostgreSQL integration test skipped: set TEST_DATABASE_URL "
            "to a dedicated test database."
        )

    try:
        parsed_test_url = make_url(test_database_url)
    except Exception:
        pytest.fail("TEST_DATABASE_URL is not a valid SQLAlchemy database URL.")

    if parsed_test_url.get_backend_name() != "postgresql":
        pytest.fail("TEST_DATABASE_URL must use PostgreSQL.")

    database_name = parsed_test_url.database or ""
    if "test" not in database_name.casefold():
        pytest.fail(
            "The database name in TEST_DATABASE_URL must contain 'test' "
            "to reduce the risk of using an application database."
        )

    application_database_url = os.getenv("DATABASE_URL")
    if application_database_url:
        try:
            parsed_application_url = make_url(application_database_url)
        except Exception:
            parsed_application_url = None

        if (
            parsed_application_url is not None
            and parsed_application_url.set(password=None)
            == parsed_test_url.set(password=None)
        ):
            pytest.fail(
                "TEST_DATABASE_URL must not point to the same database as DATABASE_URL."
            )

    # PostgreSQLContextStore continues to use the established DATABASE_URL
    # setting, but only after it has been redirected to the isolated test DB.
    monkeypatch.setenv("DATABASE_URL", test_database_url)
    dispose_engine()
    try:
        yield
    finally:
        dispose_engine()
