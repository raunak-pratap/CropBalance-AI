import unittest
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


if __name__ == "__main__":
    unittest.main()
