
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import asyncio

# Mock google.generativeai BEFORE import
mock_genai = MagicMock()
sys.modules["google.generativeai"] = mock_genai

from triforce.odin.stan.ai_provider import ProviderConfig, AIProviderFactory
from triforce.odin.stan.gemini_provider import GeminiProvider

class TestGeminiIntegration(unittest.TestCase):
    def setUp(self):
        self.config = ProviderConfig(
            type="gemini",
            api_key="fake-key",
            default_model="gemini-2.0-flash-exp"
        )
        
    def test_provider_initialization(self):
        provider = GeminiProvider(self.config)
        self.assertIsInstance(provider, GeminiProvider)
        mock_genai.configure.assert_called_with(api_key="fake-key")
        
    def test_factory_creation(self):
        provider = AIProviderFactory.create(self.config)
        self.assertIsInstance(provider, GeminiProvider)

    @patch("triforce.odin.stan.gemini_provider.genai.GenerativeModel")
    def test_generate_flow(self, mock_model_class):
        # Setup Mocks
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hello Gemini"
        
        # Async mock for generate_content_async
        async def async_gen(*args, **kwargs):
            return mock_response
            
        mock_instance.generate_content_async = async_gen
        mock_model_class.return_value = mock_instance
        
        # Run
        provider = GeminiProvider(self.config)
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(provider.generate("Hi", model="gemini-3"))
        
        self.assertEqual(result, "Hello Gemini")
        mock_model_class.assert_called_with(model_name="gemini-3", system_instruction=None)

    @patch("triforce.odin.stan.gemini_provider.genai.embed_content")
    def test_embed_flow(self, mock_embed):
        mock_embed.return_value = {"embedding": [0.1, 0.2, 0.3]}
        
        provider = GeminiProvider(self.config)
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(provider.embed("test"))
        
        self.assertEqual(result, [0.1, 0.2, 0.3])
        mock_embed.assert_called()

if __name__ == "__main__":
    unittest.main()
