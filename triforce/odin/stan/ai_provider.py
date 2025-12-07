import abc
import aiohttp
import json
import logging
import random
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field

# --- Type Definitions & Configuration ---

class ModelConfig(BaseModel):
    name: str
    size_params_b: float # Billions of params
    context_window: int = 4096
    capabilities: List[str] = ["generate"] # generate, embed, vision
    ideal_platform: str = "gpu" # gpu, cpu, metal
    fallback_for: Optional[str] = None

class ProviderConfig(BaseModel):
    type: str = "ollama" # ollama, openai, mock
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    timeout_sec: int = 30
    models: List[ModelConfig] = []
    default_model: str = "llama3"
    default_embed_model: str = "nomic-embed-text"

# --- Abstract Base Class ---

class AIProvider(abc.ABC):
    """
    Abstract interface for AI Providers (LLMs, Embeddings).
    Allows STAN to switch backends (Ollama, LM Studio, OpenAI) transparently.
    """
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.logger = logging.getLogger(f"stan.ai.{config.type}")
    
    @abc.abstractmethod
    async def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        """Generate text from a prompt."""
        pass

    @abc.abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate vector embeddings."""
        pass

    @abc.abstractmethod
    async def classify(self, text: str, labels: List[str]) -> str:
        """Classify text into one of the provided labels."""
        pass

    def get_best_model(self, task_type: str = "generate", min_size_b: float = 0) -> str:
        """Selects the best available model from config based on constraints."""
        candidates = [
            m for m in self.config.models 
            if task_type in m.capabilities and m.size_params_b >= min_size_b
        ]
        if not candidates:
            self.logger.warning(f"No exact match for {task_type} > {min_size_b}B. Using default.")
            return self.config.default_model
            
        # Sort by size (assume larger is better for complex tasks, smaller for speed?)
        # For now, pick smallest adequate model to save resources? 
        # Or largest? Let's say we pick the one closest to min_size_b without going under.
        candidates.sort(key=lambda x: x.size_params_b)
        return candidates[0].name

# --- Ollama Implementation ---

# --- Ollama Implementation ---
# Imported from external module
from triforce.odin.stan.ollama_provider import OllamaProvider

# --- Mock Implementation ---

class MockProvider(AIProvider):
    """
    Mock provider for testing without GPU/Ollama.
    """
    
    async def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        self.logger.info(f"Mock Generate: {prompt[:50]}...")
        if "json" in kwargs.get("format", ""):
            return '{"status": "mocked", "result": "success"}'
        return "This is a mocked response from the AI provider. Beep boop."

    async def embed(self, text: str) -> List[float]:
        # Return random vector of dim 768
        return [random.random() for _ in range(768)]

    async def classify(self, text: str, labels: List[str]) -> str:
        return random.choice(labels)

# --- Factory ---

class AIProviderFactory:
    @staticmethod
    def create(config: ProviderConfig) -> AIProvider:
        if config.type == "ollama":
            return OllamaProvider(config)
        elif config.type == "mock":
            return MockProvider(config)
        else:
            raise ValueError(f"Unknown provider type: {config.type}")

# --- Example Configs ---

DEFAULT_OLLAMA_CONFIG = ProviderConfig(
    type="ollama",
    base_url="http://localhost:11434",
    models=[
        ModelConfig(name="llama3", size_params_b=8.0, capabilities=["generate"], ideal_platform="gpu"),
        ModelConfig(name="mistral", size_params_b=7.0, capabilities=["generate"], ideal_platform="metal"),
        ModelConfig(name="nomic-embed-text", size_params_b=0.3, capabilities=["embed"], ideal_platform="cpu"),
    ]
)

DEFAULT_MOCK_CONFIG = ProviderConfig(type="mock")
