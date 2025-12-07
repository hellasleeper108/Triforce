import unittest
import logging
from triforce.odin.stan.model_registry import ModelRegistry
from triforce.odin.stan.model_routing import ModelRouter, NodeCapability

class TestRegistryAndRouter(unittest.TestCase):

    def setUp(self):
        logging.disable(logging.CRITICAL) # Silence logs during test
        self.registry = ModelRegistry()
        self.router = ModelRouter(self.registry)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_registry_loading(self):
        """Verify registry loads default models."""
        self.assertGreater(len(self.registry.list_models()), 0)
        llama3 = self.registry.get_model("llama3:70b")
        self.assertIsNotNone(llama3)
        self.assertEqual(llama3.best_node, "thor")

    def test_get_model_for_role(self):
        """Verify role selection logic."""
        planner = self.registry.get_model_for_role("planning")
        self.assertEqual(planner.model_name, "llama3:8b")
        
        fast = self.registry.get_model_for_role("coding") # defaults to llama3:8b usually
        self.assertIn(fast.model_name, ["llama3:8b", "llama3:70b"])

    def test_routing_basic(self):
        """Verify basic routing to preferred node."""
        # 70B -> Thor
        res = self.router.route_generate("test", "llama3:70b")
        self.assertTrue(res.success)
        self.assertEqual(res.selected_node.hostname, "thor")
        
        # 8B -> Odin (preferred in config usually, or first capable)
        res = self.router.route_generate("test", "llama3:8b")
        self.assertTrue(res.success)
        # Odin is preferred for 8B in registry? Actually registry says 'any', router config has Odin first or load based.
        # Let's check what router returns. It checks capability.
        self.assertIn(res.selected_node.hostname, ["odin", "thor"])

    def test_routing_fallback_offline(self):
        """Verify routing handles offline nodes."""
        # Kill Thor
        self.router.update_node_status("thor", is_online=False)
        
        # Request 70B (Only on Thor) -> Should fail primary, trigger fallback to 8B on Odin
        res = self.router.route_generate("test", "llama3:70b")
        
        self.assertTrue(res.success)
        self.assertEqual(res.selected_model, "llama3:8b") # Downgraded
        self.assertNotEqual(res.selected_node.hostname, "thor")

    def test_routing_overload(self):
        """Verify routing skips overloaded nodes."""
        # Odin 99% load
        self.router.update_node_status("odin", is_online=True, load=0.99)
        
        # Request 8B -> Should go to Thor (if capable) or Loki
        # Thor supports 8B too.
        res = self.router.route_generate("test", "llama3:8b")
        self.assertTrue(res.success)
        self.assertNotEqual(res.selected_node.hostname, "odin")

if __name__ == '__main__':
    unittest.main()
