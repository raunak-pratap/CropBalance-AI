import sys
from pathlib import Path

import scripts.audit_national_market_prices as audit_cli


def test_load_project_environment_uses_repository_env_file(monkeypatch):
    loaded_paths = []

    monkeypatch.setattr(
        audit_cli,
        "load_dotenv",
        lambda path: loaded_paths.append(Path(path)),
    )

    audit_cli.load_project_environment()

    assert loaded_paths == [audit_cli.PROJECT_ROOT / ".env"]


def test_main_loads_environment_before_creating_database_engine(monkeypatch, capsys):
    events = []

    class FakeEngine:
        def dispose(self):
            events.append("dispose")

    class FakeAuditor:
        def __init__(self, engine):
            assert isinstance(engine, FakeEngine)
            events.append("auditor")

        def audit(self, **kwargs):
            events.append("audit")
            return {"status": "ok", "read_only": True}

    def fake_create_engine():
        assert events == ["dotenv"]
        events.append("engine")
        return FakeEngine()

    monkeypatch.setattr(sys, "argv", ["audit_national_market_prices.py"])
    monkeypatch.setattr(
        audit_cli,
        "load_project_environment",
        lambda: events.append("dotenv"),
    )
    monkeypatch.setattr(audit_cli, "create_database_engine", fake_create_engine)
    monkeypatch.setattr(audit_cli, "NationalMarketPriceQualityAuditor", FakeAuditor)

    assert audit_cli.main() == 0
    assert events == ["dotenv", "engine", "auditor", "audit", "dispose"]
    assert '"read_only": true' in capsys.readouterr().out
