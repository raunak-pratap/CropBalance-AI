
import unittest
import pandas as pd

from scripts.audit_dharashiv_market_history import (
    metadata_inconsistency_report,
)


class TestMetadataInconsistencyReport(unittest.TestCase):

    def test_detects_umarga_wheat_metadata_difference(self):
        rows = [
            {
                "observation_date": pd.Timestamp("2026-10-27"),
                "market": "APMC Umarga",
                "market_id": "1962",
                "crop": "wheat",
                "commodity_name": "Wheat",
                "commodity_id": "1",
                "variety": "Wheat - Maharashtra 2189",
                "variety_id": None,
                "price_unit": "Rs./Quintal",
                "reported_price_raw": "NR",
                "reported_price_numeric": None,
                "price_status": "not_reported",
                "data_source": "agmarknet",
                "endpoint": "/prices-and-arrivals/commodity-price/lastweek",
                "raw_response_file": "snapshot_1.json",
            },
            {
                "observation_date": pd.Timestamp("2026-10-27"),
                "market": "APMC Umarga",
                "market_id": "1962",
                "crop": "wheat",
                "commodity_name": "Wheat",
                "commodity_id": "1",
                "variety": "Unknown",
                "variety_id": None,
                "price_unit": "NR",
                "reported_price_raw": "NR",
                "reported_price_numeric": None,
                "price_status": "not_reported",
                "data_source": "agmarknet",
                "endpoint": "/prices-and-arrivals/commodity-price/lastweek",
                "raw_response_file": "snapshot_2.json",
            },
        ]

        report = metadata_inconsistency_report(pd.DataFrame(rows))

        self.assertEqual(len(report), 1)

        result = report.iloc[0]

        self.assertEqual(result["row_count"], 2)
        self.assertIn(
            "Unknown",
            result["variety_labels"],
        )
        self.assertIn(
            "Wheat - Maharashtra 2189",
            result["variety_labels"],
        )
        self.assertIn(
            "multiple_variety_labels",
            result["metadata_flags"],
        )
        self.assertIn(
            "multiple_price_units",
            result["metadata_flags"],
        )
        self.assertEqual(
            result["distinct_numeric_price_values"],
            0,
        )
        self.assertEqual(
            result["review_status"],
            "review_required",
        )
        self.assertIn("snapshot_1.json", result["raw_response_files"])
        self.assertIn("snapshot_2.json", result["raw_response_files"])

    def test_does_not_flag_consistent_metadata(self):
        rows = [
            {
                "observation_date": pd.Timestamp("2026-10-27"),
                "market": "APMC Umarga",
                "market_id": "1962",
                "crop": "wheat",
                "commodity_id": "1",
                "variety": "Wheat - Maharashtra 2189",
                "variety_id": None,
                "price_unit": "Rs./Quintal",
                "reported_price_raw": "NR",
                "reported_price_numeric": None,
                "price_status": "not_reported",
                "data_source": "agmarknet",
                "endpoint": "/weekly",
            },
            {
                "observation_date": pd.Timestamp("2026-10-27"),
                "market": "APMC Umarga",
                "market_id": "1962",
                "crop": "wheat",
                "commodity_id": "1",
                "variety": "Wheat - Maharashtra 2189",
                "variety_id": None,
                "price_unit": "Rs./Quintal",
                "reported_price_raw": "NR",
                "reported_price_numeric": None,
                "price_status": "not_reported",
                "data_source": "agmarknet",
                "endpoint": "/weekly",
            },
        ]

        report = metadata_inconsistency_report(pd.DataFrame(rows))

        self.assertTrue(report.empty)


if __name__ == "__main__":
    unittest.main()
