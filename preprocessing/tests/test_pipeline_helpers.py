import unittest
import numpy as np
import pandas as pd

from preprocessing.clean_india_data import normalize_name, ALIAS_TO_CROP
from preprocessing.create_sequences import split_for_target, safe_std


class TestPipelineHelpers(unittest.TestCase):
    def test_soyabean_alias(self):
        self.assertEqual(ALIAS_TO_CROP[normalize_name("Soyabean")], "soybean")
        self.assertEqual(ALIAS_TO_CROP[normalize_name("Soybean")], "soybean")

    def test_normalizes_crop_punctuation(self):
        self.assertEqual(normalize_name("Tomato - Other"), "tomatoother")

    def test_chronological_split_boundaries(self):
        train_end = pd.Period("2023-12", freq="M").to_timestamp()
        val_end = pd.Period("2024-12", freq="M").to_timestamp()
        self.assertEqual(split_for_target(pd.Timestamp("2023-12-01"), train_end, val_end), "train")
        self.assertEqual(split_for_target(pd.Timestamp("2024-01-01"), train_end, val_end), "val")
        self.assertEqual(split_for_target(pd.Timestamp("2024-12-01"), train_end, val_end), "val")
        self.assertEqual(split_for_target(pd.Timestamp("2025-01-01"), train_end, val_end), "test")

    def test_safe_std_for_constant_data(self):
        self.assertEqual(safe_std(np.array([4.0, 4.0, 4.0])), 1.0)


if __name__ == "__main__":
    unittest.main()
