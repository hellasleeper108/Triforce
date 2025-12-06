
import unittest
import importlib.util
import os

# Load 'main.py' from 'python-odin'
spec = importlib.util.spec_from_file_location("odin_main", os.path.join("python-odin", "main.py"))
odin_main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(odin_main)

ClusterManager = odin_main.ClusterManager
Worker = odin_main.Worker

class TestClusterLogic(unittest.TestCase):
    def test_calculate_score(self):
        cm = ClusterManager()
        w1 = Worker(url="http://w1", metrics={"cpu_percent": 10, "memory_percent": 10}, active_jobs=0)
        w2 = Worker(url="http://w2", metrics={"cpu_percent": 90, "memory_percent": 90}, active_jobs=4)
        
        s1 = cm._calculate_score(w1)
        s2 = cm._calculate_score(w2)
        
        # Lower score is better
        self.assertLess(s1, s2)
        
    def test_gpu_score(self):
        cm = ClusterManager()
        # GPU heat adds penalty
        w_cool = Worker(url="w_cool", metrics={"gpus": [{"usage_percent": 10}]})
        w_hot = Worker(url="w_hot", metrics={"gpus": [{"usage_percent": 100}]})
        
        self.assertLess(cm._calculate_score(w_cool), cm._calculate_score(w_hot))

if __name__ == '__main__':
    unittest.main()
