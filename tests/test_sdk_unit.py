
import unittest
from unittest.mock import MagicMock, patch
from sdk.client import STANClusterClient

class TestSDK(unittest.TestCase):
    def setUp(self):
        self.client = STANClusterClient(odin_url="http://mock-odin", token="test-token")

    @patch("requests.Session.post")
    def test_submit_job(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"job_id": "123", "status": "QUEUED"}
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        def my_func(x): return x

        res = self.client.submit(my_func, 1, 2)
        self.assertEqual(res["job_id"], "123")
        
        # Verify payload
        args, kwargs = mock_post.call_args
        self.assertIn("code", kwargs["json"])
        self.assertEqual(kwargs["json"]["requires_gpu"], False)

    @patch("requests.Session.post")
    def test_submit_gpu_job(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"job_id": "gpu-1", "status": "QUEUED"}
        mock_post.return_value = mock_resp

        def my_func(): pass
        self.client.submit(my_func, gpu=True)
        
        args, kwargs = mock_post.call_args
        self.assertTrue(kwargs["json"]["requires_gpu"])

    @patch("requests.Session.get")
    def test_get_topology(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"url": "w1", "status": "ACTIVE", "active_jobs": 0, "metrics": {"gpus": []}},
            {"url": "w2", "status": "ACTIVE", "active_jobs": 0, "metrics": {"gpus": [{"temperature_c": 50}]}}
        ]
        mock_get.return_value = mock_resp

        topo = self.client.get_cluster_topology()
        self.assertEqual(len(topo), 2)

if __name__ == '__main__':
    unittest.main()
