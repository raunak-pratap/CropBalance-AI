
import tempfile
import unittest
from pathlib import Path
from datetime import date
import pandas as pd

from services.local_mandi_service import LocalMandiService


class TestLocalMandiService(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.archive_path = (
            Path(self.temp_dir.name) / "test_archive.csv"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_archive(self, records):
        pd.DataFrame(records).to_csv(
            self.archive_path,
            index=False,
        )

        return LocalMandiService(self.archive_path)

    @staticmethod
    def make_record(
        date,
        market,
        market_id,
        variety,
        price,
        unit="Rs./Quintal",
    ):
        return {
            "observation_date": date,
            "market": market,
            "market_id": str(market_id),
            "crop": "soybean",
            "reported_price": price,
            "reported_price_raw": str(price),
            "variety": variety,
            "price_unit": unit,
            "price_metric": "unverified",
            "price_status": (
                "not_reported"
                if str(price).upper() == "NR"
                else "reported_unverified_metric"
            ),
            "price_semantics_verified": "False",
            "data_source": "agmarknet",
            "endpoint": "/weekly",
            "conflict_flag": "False",
        }

    def test_latest_nr_is_not_replaced_by_old_price(self):
        service = self.write_archive([
            self.make_record(
                "2026-11-01", "APMC Kalamb", 1599,
                "Soyabean - Yellow", "7100"
            ),
            self.make_record(
                "2026-11-02", "APMC Kalamb", 1599,
                "Soyabean - Yellow", "NR"
            ),
            self.make_record(
                "2026-11-02", "APMC Murum", 1600,
                "Soyabean - Yellow", "6805"
            ),
        ])

        result = service.get_latest_prices("soybean")

        kalamb = next(
            r for r in result["records"]
            if r["market_id"] == "1599"
        )

        self.assertEqual(
            kalamb["observation_date"], "2026-11-02"
        )
        self.assertIsNone(kalamb["reported_price"])
        self.assertEqual(kalamb["price_status"], "not_reported")
        self.assertFalse(result["is_live"])

    def test_comparison_keeps_varieties_separate(self):
        service = self.write_archive([
            self.make_record(
                "2026-11-02", "APMC Kalamb", 1599,
                "Soyabean - Yellow", "7100"
            ),
            self.make_record(
                "2026-11-02", "APMC Murum", 1600,
                "Soyabean - Yellow", "6805"
            ),
            self.make_record(
                "2026-11-02", "APMC Tuljapur", 1961,
                "Soyabean - Other", "6900"
            ),
            self.make_record(
                "2026-11-02", "APMC Paranda", 1960,
                "Soyabean - Other", "5700"
            ),
        ])

        result = service.compare_markets("soybean")

        self.assertEqual(result["comparison_count"], 2)

        groups = {
            group["variety"]: group
            for group in result["comparisons"]
        }

        self.assertEqual(
            groups["Soyabean - Yellow"]["highest_reported_value"],
            7100,
        )
        self.assertEqual(
            groups["Soyabean - Other"]["highest_reported_value"],
            6900,
        )
        self.assertFalse(result["is_live"])

    def test_different_units_are_not_compared(self):
        service = self.write_archive([
            self.make_record(
                "2026-11-02", "APMC Kalamb", 1599,
                "Soyabean - Yellow", "7100", "Rs./Quintal"
            ),
            self.make_record(
                "2026-11-02", "APMC Murum", 1600,
                "Soyabean - Yellow", "70", "Rs./Kg"
            ),
        ])

        result = service.compare_markets("soybean")

        self.assertEqual(result["comparison_count"], 0)
        self.assertEqual(
            result["status"],
            "insufficient_comparable_markets",
        )

    def test_unknown_crop_returns_no_observations(self):
        service = self.write_archive([
            self.make_record(
                "2026-11-02", "APMC Kalamb", 1599,
                "Soyabean - Yellow", "7100"
            ),
        ])

        result = service.get_latest_prices("cotton")

        self.assertEqual(result["status"], "no_observations")
        self.assertEqual(result["records"], [])

    def test_ten_day_old_archive_is_marked_aging(self):
        service = self.write_archive([
            self.make_record(
                "2026-11-02",
                "APMC Kalamb",
                1599,
                "Soyabean - Yellow",
                "7100",
            ),
        ])

        frame = service._load_archive()

        freshness = service._freshness_metadata(
            frame,
            as_of=date(2026, 11, 12),
        )

        self.assertEqual(
            freshness["last_observation_date"],
            "2026-11-02",
        )
        self.assertEqual(freshness["data_age_days"], 10)
        self.assertEqual(freshness["freshness_status"], "aging")


if __name__ == "__main__":
    unittest.main()
