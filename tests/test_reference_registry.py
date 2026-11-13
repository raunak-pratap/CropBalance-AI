import json
import shutil
import tempfile
import unittest
from pathlib import Path

from services.reference_registry import ReferenceRegistry


class TestReferenceRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = ReferenceRegistry()

    def test_state_seed_contains_36_states_and_union_territories(self):
        states = self.registry.list_states()
        self.assertEqual(len(states), 36)
        self.assertEqual(
            sum(item["administrative_type"] == "state" for item in states),
            28,
        )
        self.assertEqual(
            sum(
                item["administrative_type"] == "union_territory"
                for item in states
            ),
            8,
        )

    def test_crop_aliases_resolve_to_canonical_internal_codes(self):
        self.assertEqual(
            self.registry.resolve("crops", "soyabean")["code"],
            "soybean",
        )
        self.assertEqual(
            self.registry.resolve("crops", "chana")["code"],
            "chickpea",
        )
        self.assertEqual(
            self.registry.resolve("crops", "ज्वारी")["code"],
            "sorghum",
        )

    def test_crop_list_supports_category_and_language(self):
        oilseeds = self.registry.list_crops(
            category="oilseeds",
            language="hi",
        )
        self.assertTrue(oilseeds)
        self.assertTrue(
            all(item["category"] == "oilseeds" for item in oilseeds)
        )
        soybean = next(item for item in oilseeds if item["code"] == "soybean")
        self.assertEqual(soybean["display_name"], "सोयाबीन")
        self.assertTrue(soybean["market_data_configured"])

    def test_only_seeded_dharashiv_district_is_returned(self):
        districts = self.registry.list_districts(
            state_code="Maharashtra"
        )
        self.assertEqual([item["code"] for item in districts], ["dharashiv"])
        self.assertEqual(
            self.registry.resolve("districts", "Osmanabad")["code"],
            "dharashiv",
        )

    def test_dharashiv_market_catalog_has_eight_verified_markets(self):
        markets = self.registry.list_markets(
            state_code="maharashtra",
            district_code="dharashiv",
        )
        self.assertEqual(len(markets), 8)
        self.assertTrue(
            all(item["source_ids"].get("agmarknet") for item in markets)
        )

    def test_market_query_rejects_mismatched_state_and_district(self):
        with self.assertRaises(ValueError):
            self.registry.list_markets(
                state_code="karnataka",
                district_code="dharashiv",
            )

    def test_external_ids_are_source_namespaced_and_never_guessed(self):
        self.assertEqual(
            self.registry.external_id_for(
                "state", "maharashtra", "agmarknet"
            ),
            "20",
        )
        self.assertEqual(
            self.registry.external_id_for(
                "district", "dharashiv", "agmarknet"
            ),
            "347",
        )
        self.assertIsNone(
            self.registry.external_id_for(
                "state", "maharashtra", "lgd"
            )
        )
        self.assertIsNone(
            self.registry.external_id_for(
                "crop", "rice", "agmarknet"
            )
        )

    def test_coverage_summary_does_not_claim_national_market_coverage(self):
        summary = self.registry.coverage_summary()
        self.assertFalse(summary["is_national_market_coverage_complete"])
        self.assertFalse(summary["is_national_district_coverage_complete"])
        self.assertEqual(
            summary["counts"]["states_and_union_territories"], 36
        )
        self.assertEqual(summary["counts"]["districts_catalogued"], 1)
        self.assertEqual(summary["counts"]["markets_catalogued"], 8)

    def test_duplicate_codes_fail_registry_validation(self):
        source_dir = Path(__file__).resolve().parents[1] / "data" / "reference"
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir)
            for name in (
                "crops.json",
                "states.json",
                "districts.json",
                "markets.json",
                "source_mappings.json",
            ):
                shutil.copy(source_dir / name, target / name)

            states_path = target / "states.json"
            states = json.loads(states_path.read_text(encoding="utf-8"))
            states["items"].append(dict(states["items"][0]))
            states_path.write_text(
                json.dumps(states, ensure_ascii=False),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Duplicate states code"):
                ReferenceRegistry(target)


if __name__ == "__main__":
    unittest.main()
