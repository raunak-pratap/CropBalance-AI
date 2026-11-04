import unittest
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

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
            with patch.object(collector, "RAW_DIR", Path(directory)):
                rows, status = collector.collect_one(
                    FakeSession(), "onion", "APMC Dharashiv", 1655,
                    "2026-11-03T10:00:00+00:00",
                )
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["reported_price"], 3525.0)
            self.assertEqual(rows[0]["price_status"], "reported_unverified_metric")
            self.assertIsNone(rows[1]["reported_price"])
            self.assertEqual(rows[1]["price_status"], "not_reported")
            self.assertEqual(status["valid_price_count"], 1)
            self.assertEqual(status["non_valid_price_count"], 1)
            self.assertTrue(Path(rows[0]["raw_response_file"]).exists())

    def test_existing_v3_archive_prevents_legacy_reimport(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)

            observations_file = root / "dharashiv_prices_v3.csv"
            legacy_file = root / "dharashiv_prices.csv"

            observations_file.write_text(
                "observation_date,market_id,commodity_id,variety,"
                "data_source,endpoint,reported_price_raw,price_status,"
                "raw_response_file,provenance_status\n"
                "2026-10-25,1655,23,Red,agmarknet,"
                "/prices-and-arrivals/commodity-price/lastweek,"
                "3525,reported_unverified_metric,,"
                "legacy_import_raw_unavailable\n",
                encoding="utf-8",
            )

            legacy_file.write_text(
                "date,market,market_id,crop,commodity_id,reported_price\n"
                "2026-10-25,APMC Dharashiv,1655,onion,23,3500\n",
                encoding="utf-8",
            )

            existing = pd.read_csv(observations_file, dtype=str)

            if observations_file.exists():
                existing_frames = [existing]
            else:
                legacy = pd.read_csv(legacy_file, dtype=str)
                existing_frames = [
                    collector.normalize_legacy_archive(legacy)
                ]

            self.assertEqual(len(existing_frames), 1)
            self.assertEqual(
                existing_frames[0].iloc[0]["reported_price_raw"],
                "3525",
            )
            self.assertEqual(len(existing_frames[0]), 1)


if __name__ == "__main__":
    unittest.main()
