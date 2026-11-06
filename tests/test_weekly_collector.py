import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from scripts import sync_dharashiv_mandi_archive as collector


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "success": True,
            "columns": [
                {"key": "2026-10-25"},
                {"key": "2026-10-26"},
            ],
            "data": [{
                "variety": "Red",
                "unitOfPrice": "Rs./Quintal",
                "2026-10-25": "3525",
                "2026-10-26": "NR",
            }],
        }


class FakeSession:
    def get(self, url, params, timeout):
        self.last_request = (url, params, timeout)
        return FakeResponse()


class TestWeeklyCollector(unittest.TestCase):
    def test_collector_keeps_nr_as_status_and_writes_raw_snapshot(self):
        with TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            with patch.object(collector, "RAW_DIR", raw_dir):
                rows, status = collector.collect_one(
                    FakeSession(),
                    "onion",
                    "APMC Dharashiv",
                    1655,
                    "2026-11-03T10:00:00+00:00",
                )

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["reported_price"], 3525.0)
            self.assertEqual(
                rows[0]["price_status"], "reported_unverified_metric"
            )
            self.assertIsNone(rows[1]["reported_price"])
            self.assertEqual(rows[1]["price_status"], "not_reported")
            self.assertEqual(status["valid_price_count"], 1)
            self.assertEqual(status["non_valid_price_count"], 1)
            self.assertTrue(Path(rows[0]["raw_response_file"]).exists())

    @staticmethod
    def _make_observation(
        *, date: str, price: str, fetched_at: str, snapshot_path: Path
    ) -> dict:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(
            json.dumps({"test_snapshot": snapshot_path.name}),
            encoding="utf-8",
        )
        return collector.make_observation(
            date_key=date,
            record={
                date: price,
                "unitOfPrice": "Rs./Quintal",
                "variety": "Red",
            },
            state="Maharashtra",
            state_id=20,
            district="Dharashiv",
            district_id=347,
            market="APMC Dharashiv",
            market_id=1655,
            crop="onion",
            commodity_name="Onion",
            commodity_id=23,
            fetched_at_utc=fetched_at,
            endpoint=collector.ENDPOINT_PATH,
            raw_response_file=str(snapshot_path),
        )

    def _run_main_with_scenarios(
        self,
        root: Path,
        scenarios: list[tuple[str, str]],
        *,
        existing_archive: pd.DataFrame | None = None,
        legacy_csv: str | None = None,
    ) -> tuple[list[int], Path, Path, Path]:
        """Run the real collector main() with a mocked one-market API call."""
        output_dir = root / "data" / "raw" / "agmarknet_weekly"
        raw_dir = output_dir / "responses"
        observations_file = output_dir / "dharashiv_prices_v3.csv"
        runs_file = output_dir / "collection_runs.csv"
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)

        if existing_archive is not None:
            existing_archive.to_csv(observations_file, index=False)
        if legacy_csv is not None:
            (output_dir / "dharashiv_prices.csv").write_text(
                legacy_csv, encoding="utf-8"
            )

        scenario_index = {"value": 0}

        def fake_collect_one(session, crop, market, market_id, fetched_at):
            index = scenario_index["value"]
            scenario_index["value"] += 1
            date, price = scenarios[index]
            snapshot_path = raw_dir / f"snapshot_{index + 1}.json"
            row = self._make_observation(
                date=date,
                price=price,
                fetched_at=fetched_at,
                snapshot_path=snapshot_path,
            )
            return [row], {
                "status": "success_with_observations",
                "records_seen": 1,
                "valid_price_count": 1,
                "non_valid_price_count": 0,
                "error": None,
                "raw_response_file": str(snapshot_path),
            }

        timestamps = [
            f"2026-11-0{3 + index}T10:00:00+00:00"
            for index in range(len(scenarios))
        ]
        results = []

        with (
            patch.object(collector, "OUTPUT_DIR", output_dir),
            patch.object(collector, "OBSERVATIONS_FILE", observations_file),
            patch.object(collector, "RUNS_FILE", runs_file),
            patch.object(collector, "RAW_DIR", raw_dir),
            patch.object(collector, "MARKETS", {"APMC Dharashiv": 1655}),
            patch.object(
                collector,
                "CROPS",
                {"onion": {"id": 23, "name": "Onion"}},
            ),
            patch.object(collector, "create_session", return_value=FakeSession()),
            patch.object(collector, "collect_one", side_effect=fake_collect_one),
            patch.object(collector, "utc_now_iso", side_effect=timestamps),
            patch.object(collector.time, "sleep"),
            patch.object(sys, "argv", ["sync_dharashiv_mandi_archive.py"]),
        ):
            for _ in scenarios:
                results.append(collector.main())

        return results, observations_file, runs_file, raw_dir

    def test_repeated_collection_is_idempotent_and_keeps_latest_snapshot(self):
        with TemporaryDirectory() as directory:
            results, observations_file, runs_file, raw_dir = (
                self._run_main_with_scenarios(
                    Path(directory),
                    [
                        ("2026-10-25", "3525"),
                        ("2026-10-25", "3525.0"),
                    ],
                )
            )

            archive = pd.read_csv(observations_file, dtype=str)
            run_log = pd.read_csv(runs_file, dtype=str)

            self.assertEqual(results, [0, 0])
            self.assertEqual(len(archive), 1)
            self.assertEqual(archive.iloc[0]["reported_price_raw"], "3525.0")
            self.assertEqual(
                archive.iloc[0]["fetched_at_utc"],
                "2026-11-04T10:00:00+00:00",
            )
            self.assertTrue(
                archive.iloc[0]["raw_response_file"].endswith("snapshot_2.json")
            )
            self.assertEqual(len(run_log), 2)
            self.assertEqual(len(list(raw_dir.glob("*.json"))), 2)

    def test_new_observation_date_is_added(self):
        with TemporaryDirectory() as directory:
            results, observations_file, runs_file, _ = (
                self._run_main_with_scenarios(
                    Path(directory),
                    [
                        ("2026-10-25", "3525"),
                        ("2026-10-26", "3550"),
                    ],
                )
            )

            archive = pd.read_csv(observations_file, dtype=str)

            self.assertEqual(results, [0, 0])
            self.assertEqual(len(archive), 2)
            self.assertEqual(
                set(archive["observation_date"]),
                {"2026-10-25", "2026-10-26"},
            )
            self.assertFalse(
                archive["conflict_flag"].astype(str).str.lower().eq("true").any()
            )

    def test_changed_price_is_preserved_and_flagged_as_conflict(self):
        with TemporaryDirectory() as directory:
            results, observations_file, runs_file, _ = (
                self._run_main_with_scenarios(
                    Path(directory),
                    [
                        ("2026-10-25", "3525"),
                        ("2026-10-25", "3600"),
                    ],
                )
            )

            archive = pd.read_csv(observations_file, dtype=str)

            self.assertEqual(results, [0, 0])
            self.assertEqual(len(archive), 2)
            self.assertEqual(
                set(archive["reported_price_raw"]), {"3525", "3600"}
            )
            self.assertTrue(
                archive["conflict_flag"].astype(str).str.lower().eq("true").all()
            )

    def test_existing_v3_archive_prevents_legacy_reimport_in_real_main(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "data" / "raw" / "agmarknet_weekly"
            raw_dir = output_dir / "responses"
            output_dir.mkdir(parents=True, exist_ok=True)
            raw_dir.mkdir(parents=True, exist_ok=True)

            existing = self._make_observation(
                date="2026-10-25",
                price="3525",
                fetched_at="2026-11-02T10:00:00+00:00",
                snapshot_path=raw_dir / "existing.json",
            )
            existing_archive = pd.DataFrame(
                [existing], columns=collector.OBSERVATION_COLUMNS
            )
            legacy_csv = (
                "date,market,market_id,crop,commodity_id,reported_price\n"
                "2026-10-25,APMC Dharashiv,1655,onion,23,3500\n"
            )

            results, observations_file, _, _ = self._run_main_with_scenarios(
                root,
                [("2026-10-25", "3525")],
                existing_archive=existing_archive,
                legacy_csv=legacy_csv,
            )

            archive = pd.read_csv(observations_file, dtype=str)

            self.assertEqual(results, [0])
            self.assertEqual(len(archive), 1)
            self.assertEqual(archive.iloc[0]["reported_price_raw"], "3525")
            self.assertFalse(
                archive["conflict_flag"].astype(str).str.lower().eq("true").any()
            )


if __name__ == "__main__":
    unittest.main()
