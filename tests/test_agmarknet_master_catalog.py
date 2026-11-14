import json
import tempfile
import unittest
from pathlib import Path

from services.agmarknet_master_catalog import (
    AgmarknetMasterCatalog,
    build_agmarknet_master_snapshot,
    write_snapshot_files,
)


def sample_metadata():
    return {
        "state_data": [
            {"state_id": 100000, "state_name": "All States/UTs"},
            {"state_id": 20, "state_name": "Maharashtra"},
            {"state_id": 12, "state_name": "Haryana"},
        ],
        "district_data": [
            {"id": 100001, "state_id": None, "district_name": "All Districts"},
            {"id": 347, "state_id": 20, "district_name": "Dharashiv"},
            {"id": 200, "state_id": 12, "district_name": "Hisar"},
        ],
        "market_data": [
            {"id": 100002, "state_id": None, "district_id": None, "mkt_name": "All Markets"},
            {"id": 1959, "state_id": 20, "district_id": 347, "mkt_name": "APMC Bhoom"},
            {"id": 999, "state_id": 12, "district_id": 200, "mkt_name": "Hisar Mandi"},
        ],
        "cmdt_group_data": [
            {"id": 1, "cmdt_grp_name": "Cereals"},
            {"id": 2, "cmdt_grp_name": "Pulses"},
        ],
        "cmdt_data": [
            {"cmdt_id": 1, "cmdt_name": "Wheat", "cmdt_group_id": 1},
            {"cmdt_id": 13, "cmdt_name": "Soyabean", "cmdt_group_id": 2},
        ],
        "variety_data": [
            {"id": 100007, "variety_name": "All Varieties", "cmdt_id": None},
            {"id": 101, "variety_name": "Yellow", "cmdt_id": 13},
            {"id": 102, "variety_name": "Common", "cmdt_id": [1, 13]},
        ],
    }


class TestAgmarknetMasterSnapshot(unittest.TestCase):
    def test_selector_rows_are_excluded_and_counts_are_reported(self):
        snapshot = build_agmarknet_master_snapshot(
            sample_metadata(),
            metadata_sha256="abc123",
            synced_at_utc="2026-11-14T10:00:00+00:00",
        )

        self.assertEqual(
            snapshot["manifest"]["counts"],
            {
                "states": 2,
                "districts": 2,
                "markets": 2,
                "commodities": 2,
                "varieties": 2,
            },
        )
        self.assertEqual(
            snapshot["manifest"]["excluded_selector_rows"],
            {
                "states": 1,
                "districts": 1,
                "markets": 1,
                "commodities": 0,
                "varieties": 1,
            },
        )

    def test_parent_references_are_preserved_and_validated(self):
        snapshot = build_agmarknet_master_snapshot(sample_metadata())
        districts = snapshot["files"]["districts.json"]["records"]
        markets = snapshot["files"]["markets.json"]["records"]

        dharashiv = next(item for item in districts if item["source_id"] == "347")
        bhoom = next(item for item in markets if item["source_id"] == "1959")

        self.assertEqual(dharashiv["state_source_id"], "20")
        self.assertEqual(bhoom["state_source_id"], "20")
        self.assertEqual(bhoom["district_source_id"], "347")
        self.assertEqual(
            snapshot["manifest"]["relationship_validation"][
                "markets_with_consistent_state_and_district"
            ],
            2,
        )

    def test_commodity_group_names_and_variety_links_are_preserved(self):
        snapshot = build_agmarknet_master_snapshot(sample_metadata())
        commodities = snapshot["files"]["commodities.json"]["records"]
        varieties = snapshot["files"]["varieties.json"]["records"]

        soybean = next(item for item in commodities if item["source_id"] == "13")
        common = next(item for item in varieties if item["source_id"] == "102")

        self.assertEqual(soybean["group_name"], "Pulses")
        self.assertEqual(common["commodity_source_ids"], ["1", "13"])

    def test_duplicate_source_ids_are_rejected(self):
        metadata = sample_metadata()
        metadata["market_data"].append(dict(metadata["market_data"][1]))

        with self.assertRaisesRegex(ValueError, "Duplicate AGMARKNET markets source ID"):
            build_agmarknet_master_snapshot(metadata)

    def test_invalid_market_parent_is_rejected(self):
        metadata = sample_metadata()
        metadata["market_data"][1]["district_id"] = 200

        with self.assertRaisesRegex(ValueError, "inconsistent state/district"):
            build_agmarknet_master_snapshot(metadata)

    def test_unknown_variety_commodity_reference_is_rejected(self):
        metadata = sample_metadata()
        metadata["variety_data"][1]["cmdt_id"] = 9999

        with self.assertRaisesRegex(ValueError, "unknown commodity IDs"):
            build_agmarknet_master_snapshot(metadata)

    def test_missing_required_dimension_is_rejected(self):
        metadata = sample_metadata()
        del metadata["market_data"]

        with self.assertRaisesRegex(ValueError, "market_data"):
            build_agmarknet_master_snapshot(metadata)

    def test_snapshot_writes_atomically_and_catalog_paginates(self):
        snapshot = build_agmarknet_master_snapshot(sample_metadata())
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            written = write_snapshot_files(snapshot, output)

            self.assertEqual(len(written), 6)
            self.assertTrue((output / "manifest.json").is_file())

            catalog = AgmarknetMasterCatalog(output)
            coverage = catalog.coverage()
            self.assertEqual(coverage["counts"]["markets"], 2)

            markets = catalog.list_records(
                "markets",
                state_id="20",
                district_id="347",
                offset=0,
                limit=10,
            )
            self.assertEqual(markets["count"], 1)
            self.assertEqual(markets["items"][0]["name"], "APMC Bhoom")

            varieties = catalog.list_records(
                "varieties",
                commodity_id="13",
                offset=0,
                limit=10,
            )
            self.assertEqual(varieties["count"], 2)

            page = catalog.list_records(
                "markets",
                offset=0,
                limit=1,
            )
            self.assertEqual(len(page["items"]), 1)
            self.assertEqual(page["count"], 2)

    def test_invalid_page_size_is_rejected(self):
        snapshot = build_agmarknet_master_snapshot(sample_metadata())
        with tempfile.TemporaryDirectory() as temp_dir:
            write_snapshot_files(snapshot, temp_dir)
            catalog = AgmarknetMasterCatalog(temp_dir)

            with self.assertRaisesRegex(ValueError, "limit must be between"):
                catalog.list_records("markets", limit=501)


if __name__ == "__main__":
    unittest.main()
