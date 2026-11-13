
import unittest

from agent.response import build_response


class TestLocalMandiResponseFreshness(unittest.TestCase):

    def make_result(self, freshness_status, age_days=10):
        return {
            "get_local_mandi_prices": {
                "result": {
                    "crop": "soybean",
                    "district": "Dharashiv",
                    "query_mode": "comparison",
                    "comparisons": [
                        {
                            "observation_date": "2026-11-02",
                            "price_unit": "Rs./Quintal",
                            "variety": "Soyabean - Yellow",
                            "highest_reported_value": 7100,
                            "highest_reported_markets": [
                                "APMC Kalamb"
                            ],
                            "records_ranked_high_to_low": [],
                        }
                    ],
                    "data_quality": {
                        "numeric_price_count": 5,
                        "observation_count": 8,
                        "missing_or_invalid_price_count": 3,
                    },
                    "last_observation_date": "2026-11-02",
                    "data_age_days": age_days,
                    "freshness_status": freshness_status,
                    "is_live": False,
                    "caveat": "These are archived reported values, not live quotes.",
                }
            }
        }

    def test_aging_data_produces_warning(self):
        response = build_response(
            self.make_result("aging", 10)
        )

        message = response["message"]

        self.assertIn("Data freshness: Aging", message)
        self.assertIn("2026-11-02", message)
        self.assertIn("10 days old", message)
        self.assertIn("current market source", message)

    def test_stale_data_produces_stronger_warning(self):
        response = build_response(
            self.make_result("stale", 20)
        )

        message = response["message"]

        self.assertIn("Data freshness: Stale", message)
        self.assertIn("20 days old", message)
        self.assertIn("Verify directly with the market", message)

    def test_fresh_data_is_labelled_correctly(self):
        response = build_response(
            self.make_result("fresh", 3)
        )

        message = response["message"]

        self.assertIn("Data freshness: Fresh", message)
        self.assertIn("3 days old", message)


if __name__ == "__main__":
    unittest.main()
