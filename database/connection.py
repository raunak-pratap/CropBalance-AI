"""Lazy SQLAlchemy connection setup for CropBalance AI.

Importing this module does not require DATABASE_URL or open a database
connection. Database configuration is validated when get_engine() or
SessionLocal() is first used.
"""
from __future__ import annotations

import os
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


_engine: Engine | None = None
_engine_url: str | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine(database_url: str | None = None) -> Engine:
    """Return the cached SQLAlchemy engine, creating it on first use.

    An explicitly supplied URL takes precedence over DATABASE_URL. If the
    configured URL changes (for example, in an isolated test fixture), the
    previous engine is disposed before a new one is created.
    """
    global _engine, _engine_url, _session_factory

    configured_url = (database_url or os.getenv("DATABASE_URL") or "").strip()
    if not configured_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Configure it before performing database operations."
        )

    if _engine is None or _engine_url != configured_url:
        dispose_engine()
        _engine = create_engine(
            configured_url,
            pool_pre_ping=True,
            future=True,
        )
        _engine_url = configured_url
        _session_factory = sessionmaker(
            bind=_engine,
            autoflush=False,
            autocommit=False,
        )

    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the lazily initialized SQLAlchemy session factory."""
    get_engine()
    assert _session_factory is not None
    return _session_factory


def SessionLocal(*args: Any, **kwargs: Any) -> Session:
    """Create a database session while preserving the existing call pattern."""
    return get_session_factory()(*args, **kwargs)


def dispose_engine() -> None:
    """Dispose and clear the cached engine and session factory.

    Primarily useful for controlled application shutdown and isolated tests.
    """
    global _engine, _engine_url, _session_factory

    if _engine is not None:
        _engine.dispose()

    _engine = None
    _engine_url = None
    _session_factory = None
