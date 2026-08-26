import unittest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.features.feature_engineering import build_preprocessing_pipeline
from src.models.evaluate import evaluate_model

class TestModelModule(unittest.TestCase):
    def test_preprocessing_pipeline(self):
        preprocessor = build_preprocessing_pipeline()
        dummy_X = np.random.randn(10, 16)
        transformed = preprocessor.fit_transform(dummy_X)
        self.assertEqual(transformed.shape[0], 10)
        self.assertGreater(transformed.shape[1], 16)

    def test_random_forest_training(self):
        preprocessor = build_preprocessing_pipeline()
        clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        
        # Train dummy batch
        X = pd.DataFrame(np.random.randn(40, 16), columns=[f"X{i}" for i in range(1, 17)])
        y = np.array([0, 1, 2, 3] * 10)
        
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        
        self.assertEqual(len(preds), 40)
        self.assertTrue(set(preds).issubset({0, 1, 2, 3}))
        
        metrics = evaluate_model(pipeline, X, y)
        self.assertIn("accuracy", metrics)
        self.assertIn("f1_macro", metrics)

if __name__ == "__main__":
    unittest.main()

