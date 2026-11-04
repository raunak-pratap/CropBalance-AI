import unittest

import pandas as pd

from services.agmarknet_local_archive import (
    classify_reported_price,
    make_observation,
    merge_observations,
    normalize_legacy_archive,
    parse_agmarknet_date,
)


class TestAgmarknetLocalArchive(unittest.TestCase):
    def test_nr_is_missing_not_zero(self):
        value, status = classify_reported_price("NR")
        self.assertIsNone(value)
        self.assertEqual(status, "not_reported")

    def test_blank_and_non_numeric_are_distinguished(self):
        self.assertEqual(classify_reported_price(" "), (None, "missing_value"))
        self.assertEqual(classify_reported_price("captcha"), (None, "non_numeric"))

    def test_nonpositive_price_is_not_valid(self):
        self.assertEqual(classify_reported_price("0"), (0.0, "non_positive"))

    def test_positive_value_remains_unverified(self):
        self.assertEqual(classify_reported_price("3525"), (3525.0, "reported_unverified_metric"))

    def test_date_parser_handles_iso_and_indian_day_first(self):
        self.assertEqual(parse_agmarknet_date("2026-10-25"), pd.Timestamp("2026-10-25"))
        self.assertEqual(parse_agmarknet_date("25/10/2026"), pd.Timestamp("2026-10-25"))

    def test_observation_preserves_unit_and_does_not_invent_arrivals(self):
        row = make_observation(
            date_key="25/10/2026",
            record={"25/10/2026": "3525", "unitOfPrice": "Rs./Quintal", "variety": "Red"},
            state="Maharashtra", state_id=20, district="Dharashiv", district_id=347,
            market="APMC Dharashiv", market_id=1655, crop="onion", commodity_name="Onion",
            commodity_id=23, fetched_at_utc="2026-11-03T10:00:00+00:00",
            endpoint="/prices-and-arrivals/commodity-price/lastweek", raw_response_file="raw.json",
        )
        self.assertEqual(row["reported_price"], 3525.0)
        self.assertEqual(row["price_unit"], "Rs./Quintal")
        self.assertEqual(row["price_metric"], "unverified")
        self.assertFalse(row["price_semantics_verified"])
        self.assertIsNone(row["arrivals_value"])
        self.assertIsNone(row["arrivals_unit"])

    def test_merge_is_idempotent_for_same_source_value(self):
        base = {
            "observation_date": "2026-10-25", "market_id": 1655,
            "commodity_id": 23, "variety": "Red", "price_unit": "Rs./Quintal",
            "data_source": "agmarknet", "endpoint": "/weekly",
            "reported_price_raw": "3525", "price_status": "reported_unverified_metric",
            "reported_price": 3525.0, "crop": "onion", "market": "APMC Dharashiv",
            "fetched_at_utc": "2026-11-03T10:00:00+00:00",
        }
        result = merge_observations(pd.DataFrame([base]), pd.DataFrame([base]))
        self.assertEqual(len(result), 1)
        self.assertFalse(bool(result.iloc[0]["conflict_flag"]))

    def test_merge_preserves_conflicting_report_values(self):
        base = {
            "observation_date": "2026-10-25", "market_id": 1655,
            "commodity_id": 23, "variety": "Red", "price_unit": "Rs./Quintal",
            "data_source": "agmarknet", "endpoint": "/weekly",
            "reported_price_raw": "3525", "price_status": "reported_unverified_metric",
            "reported_price": 3525.0, "crop": "onion", "market": "APMC Dharashiv",
            "fetched_at_utc": "2026-11-03T10:00:00+00:00",
        }
        changed = {**base, "reported_price_raw": "3600", "reported_price": 3600.0,
                   "fetched_at_utc": "2026-11-04T10:00:00+00:00"}
        result = merge_observations(pd.DataFrame([base]), pd.DataFrame([changed]))
        self.assertEqual(len(result), 2)
        self.assertTrue(result["conflict_flag"].astype(bool).all())

    def test_legacy_archive_migration_keeps_metric_unverified(self):
        legacy = pd.DataFrame([{
            "date": "2026-10-25", "state": "Maharashtra", "district": "Dharashiv",
            "market": "APMC Dharashiv", "market_id": 1655, "crop": "onion",
            "commodity_name": "Onion", "commodity_id": 23, "variety": "Red",
            "reported_price": 3525, "price_unit": "Rs./Quintal",
            "fetched_at": "2026-11-02T10:00:00+00:00",
        }])
        migrated = normalize_legacy_archive(legacy)
        self.assertEqual(len(migrated), 1)
        self.assertEqual(migrated.loc[0, "reported_price"], 3525.0)
        self.assertEqual(migrated.loc[0, "price_metric"], "unverified")
        self.assertFalse(migrated.loc[0, "price_semantics_verified"])
        self.assertIsNone(migrated.loc[0, "arrivals_value"])

    
    def test_merge_collapses_numeric_format_duplicates(self):
        base = {
            "observation_date": "2026-10-26",
            "market_id": 1599,
            "commodity_id": 1,
            "variety": "Wheat - Maharashtra",
            "price_unit": "Rs./Quintal",
            "data_source": "agmarknet",
            "endpoint": "/weekly",
            "reported_price_raw": "3000",
            "reported_price": 3000.0,
            "price_status": "reported_unverified_metric",
            "fetched_at_utc": "2026-11-03T10:00:00+00:00",
        }

        repeated = {
            **base,
            "reported_price_raw": "3000.0",
            "fetched_at_utc": "2026-11-04T10:00:00+00:00",
        }

        result = merge_observations(
            pd.DataFrame([base]),
            pd.DataFrame([repeated]),
        )

        self.assertEqual(len(result), 1)
        self.assertFalse(bool(result.iloc[0]["conflict_flag"]))
        self.assertEqual(
            result.iloc[0]["fetched_at_utc"],
            "2026-11-04T10:00:00+00:00",
        )

    def test_merge_detects_unit_change_as_conflict(self):
        base = {
            "observation_date": "2026-10-26",
            "market_id": 1599,
            "commodity_id": 1,
            "variety": "Wheat - Maharashtra",
            "price_unit": "Rs./Quintal",
            "data_source": "agmarknet",
            "endpoint": "/weekly",
            "reported_price_raw": "3000",
            "reported_price": 3000.0,
            "price_status": "reported_unverified_metric",
        }

        changed_unit = {
            **base,
            "price_unit": "Rs./Kg",
            "fetched_at_utc": "2026-11-04T10:00:00+00:00",
        }

        result = merge_observations(
            pd.DataFrame([base]),
            pd.DataFrame([changed_unit]),
        )

        self.assertEqual(len(result), 2)
        self.assertTrue(result["conflict_flag"].astype(bool).all())

    def test_merge_collapses_repeated_nr_records(self):
        base = {
            "observation_date": "2026-10-26",
            "market_id": 1599,
            "commodity_id": 23,
            "variety": "Unknown",
            "price_unit": "NR",
            "data_source": "agmarknet",
            "endpoint": "/weekly",
            "reported_price_raw": "NR",
            "reported_price": None,
            "price_status": "not_reported",
        }

        repeated = {
            **base,
            "fetched_at_utc": "2026-11-04T10:00:00+00:00",
        }

        result = merge_observations(
            pd.DataFrame([base]),
            pd.DataFrame([repeated]),
        )

        self.assertEqual(len(result), 1)
        self.assertFalse(bool(result.iloc[0]["conflict_flag"]))

def test_legacy_observation_has_legacy_provenance(self):
    legacy = pd.DataFrame([{
        "date": "2026-10-25",
        "market": "APMC Dharashiv",
        "market_id": 1655,
        "crop": "onion",
        "commodity_id": 23,
        "reported_price": "3500",
    }])

    result = normalize_legacy_archive(legacy)

    self.assertEqual(
        result.iloc[0]["provenance_status"],
        "legacy_import_raw_unavailable",
    )
    self.assertTrue(pd.isna(result.iloc[0]["raw_response_file"]))


def test_merge_preserves_provenance_status(self):
    legacy = pd.DataFrame([{
        "date": "2026-10-25",
        "market": "APMC Dharashiv",
        "market_id": 1655,
        "crop": "onion",
        "commodity_id": 23,
        "reported_price": "3500",
    }])

    result = merge_observations(
        normalize_legacy_archive(legacy),
        pd.DataFrame(),
    )

    self.assertEqual(
        result.iloc[0]["provenance_status"],
        "legacy_import_raw_unavailable",
    )


if __name__ == "__main__":
    unittest.main()
