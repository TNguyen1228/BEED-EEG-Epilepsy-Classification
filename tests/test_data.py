import unittest
import pandas as pd
from src.data.data_loader import load_config, validate_data, load_and_split_data

class TestDataModule(unittest.TestCase):
    def test_load_config(self):
        config = load_config("config/config.yaml")
        self.assertIn("data", config)
        self.assertIn("model", config)
        self.assertEqual(len(config["data"]["feature_columns"]), 16)

    def test_validate_data(self):
        config = load_config("config/config.yaml")
        df = pd.read_csv(config["data"]["raw_path"])
        self.assertTrue(validate_data(df, config))

    def test_data_splits(self):
        X_train, X_test, y_train, y_test = load_and_split_data("config/config.yaml")
        self.assertEqual(len(X_train) + len(X_test), 8000)
        self.assertEqual(X_train.shape[1], 16)
        self.assertEqual(set(y_train.unique()), {0, 1, 2, 3})

if __name__ == "__main__":
    unittest.main()

