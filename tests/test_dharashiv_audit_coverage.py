"""Regression tests for observation-date vs reported-price coverage metrics."""
import unittest

import pandas as pd

from scripts.audit_dharashiv_market_history import group_quality


class TestDharashivAuditCoverage(unittest.TestCase):
    def _row(self, date, price):
        return {
            "observation_date": pd.Timestamp(date),
            "reported_price_numeric": price,
            "crop": "onion",
            "market": "APMC Bhoom",
            "market_id": "1959",
            "variety": "Red",
            "price_unit": "Rs./Quintal" if price is not None else "NR",
            "data_source": "agmarknet",
            "price_metric": "unverified",
            "arrivals_numeric": float("nan"),
        }

    def test_observation_and_reported_price_gaps_are_separate(self):
        frame = pd.DataFrame(
            [
                self._row("2026-10-27", 1000.0),
                self._row("2026-10-29", None),
                self._row("2026-11-02", 1200.0),
            ]
        )

        result = group_quality(frame).iloc[0]

        self.assertEqual(result["unique_observation_dates"], 3)
        self.assertEqual(result["max_gap_between_observation_dates_days"], 4)
        self.assertEqual(result["missing_calendar_dates_count"], 4)
        self.assertEqual(result["unique_reported_price_dates"], 2)
        self.assertEqual(result["max_gap_between_reported_dates_days"], 6)
        self.assertEqual(
            result["max_missing_calendar_days_between_reported_prices"], 5
        )

    def test_no_reported_prices_has_no_reported_price_gap(self):
        frame = pd.DataFrame(
            [
                self._row("2026-10-27", None),
                self._row("2026-10-28", None),
            ]
        )

        result = group_quality(frame).iloc[0]

        self.assertEqual(result["unique_observation_dates"], 2)
        self.assertEqual(result["unique_reported_price_dates"], 0)
        self.assertIsNone(result["max_gap_between_reported_dates_days"])
        self.assertIsNone(
            result["max_missing_calendar_days_between_reported_prices"]
        )


if __name__ == "__main__":
    unittest.main()
