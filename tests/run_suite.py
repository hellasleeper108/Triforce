import unittest
import os
import sys

def run_tests():
    # Ensure project root is in path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    print(f"--- STAN Test Suite ---")
    print(f"Root: {project_root}\n")

    loader = unittest.TestLoader()
    start_dir = os.path.join(project_root, 'tests/stan')
    
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if not result.wasSuccessful():
        sys.exit(1)

if __name__ == '__main__':
    run_tests()
