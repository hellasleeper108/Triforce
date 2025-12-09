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

    def set_active_model(self, model_name: str) -> bool:
        """Sets the default model for generation."""
        # Validate existence
        if any(m.name == model_name for m in self.config.models):
            self.logger.info(f"Switching active model to: {model_name}")
            self.config.default_model = model_name
            return True
        self.logger.warning(f"Model {model_name} not found in config.")
        return False
        
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Returns list of available models with metadata."""
        return [
            {
                "name": m.name, 
                "active": m.name == self.config.default_model,
                "platform": m.ideal_platform,
                "capabilities": m.capabilities
            }
            for m in self.config.models
        ]

    def get_best_model(self, task_type: str = "generate", min_size_b: float = 0) -> str:
        """Selects the best available model from config based on constraints."""
        candidates = [
            m for m in self.config.models 
            if task_type in m.capabilities and m.size_params_b >= min_size_b
        ]
        if not candidates:
            # self.logger.warning(f"No exact match for {task_type} > {min_size_b}B. Using default.")
            return self.config.default_model
            
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
        
        # Handle Metacognition Critique
        if system and "CRITIQUE" in system:
            return json.dumps({
                "score": 0.9, 
                "flaws": [], 
                "safety_violations": [],
                "suggestion": "Plan looks safe."
            })
            
        # Handle Metacognition Bias Check
        if system and "BIAS" in system:
            return json.dumps({
                "detected_bias": None,
                "severity": "low", 
                "mitigation": "None"
            })

        if "json" in kwargs.get("format", "") or kwargs.get("json_format"):
            # Generic JSON fallback if not matched above
            return '{"status": "mocked", "result": "success"}'
            
        return "This is a mocked response from the AI provider. Beep boop."

    async def embed(self, text: str) -> List[float]:
        # Return random vector of dim 768
        return [random.random() for _ in range(768)]

    async def classify(self, text: str, labels: List[str]) -> str:
        return random.choice(labels)

# --- Factory ---

from triforce.odin.stan.gemini_provider import GeminiProvider

# --- Factory ---

class AIProviderFactory:
    @staticmethod
    def create(config: ProviderConfig) -> AIProvider:
        if config.type == "ollama":
            return OllamaProvider(config)
        elif config.type == "gemini":
            return GeminiProvider(config)
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

DEFAULT_MOCK_CONFIG = ProviderConfig(
    type="mock",
    models=[
        ModelConfig(name="mock-gpt-4", size_params_b=100.0, capabilities=["generate"], ideal_platform="cloud"),
        ModelConfig(name="mock-local-7b", size_params_b=7.0, capabilities=["generate"], ideal_platform="gpu"),
    ],
    default_model="mock-gpt-4"
)

DEFAULT_GEMINI_CONFIG = ProviderConfig(
    type="gemini",
    models=[
        ModelConfig(name="gemini-2.0-flash-exp", size_params_b=100.0, capabilities=["generate", "vision"], ideal_platform="cloud"),
        ModelConfig(name="gemini-1.5-pro", size_params_b=100.0, capabilities=["generate", "vision"], ideal_platform="cloud"),
        ModelConfig(name="text-embedding-004", size_params_b=0.1, capabilities=["embed"], ideal_platform="cloud"),
    ],
    default_model="gemini-2.0-flash-exp", # User requested Gemini 3 (using 2.0 Flash Exp as proxy)
    default_embed_model="text-embedding-004"
)
