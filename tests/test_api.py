import unittest

try:
    from fastapi.testclient import TestClient
    from api.app import app
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

class TestAPIService(unittest.TestCase):
    def setUp(self):
        if not FASTAPI_AVAILABLE:
            self.skipTest("FastAPI / TestClient is not installed in the environment.")
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("model_loaded", data)

    def test_info_endpoint(self):
        response = self.client.get("/info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("labels", data)

    def test_predict_validation_error(self):
        # Sending 3 features instead of 16 should trigger validation error
        response = self.client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
        self.assertEqual(response.status_code, 422)

if __name__ == "__main__":
    unittest.main()

