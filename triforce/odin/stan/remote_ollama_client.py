import logging
import time
import json
import asyncio
import aiohttp
from typing import List, Dict, Optional, Any, Union

# Imports
from triforce.odin.stan.ai_provider import AIProvider, ModelConfig

class RemoteOllamaClient(AIProvider):
    """
    Specialized client for communicating with Ollama instances on remote nodes (Thor, Loki).
    Includes aggressive retries, connection pooling, and latency benchmarking.
    """

    def __init__(self, base_url: str, node_id: str, timeout_sec: int = 30, retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.node_id = node_id
        self.timeout = aiohttp.ClientTimeout(total=timeout_sec)
        self.retries = retries
        self.logger = logging.getLogger(f"stan.remote.{node_id}")
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _post_with_retry(self, endpoint: str, payload: dict) -> Dict[str, Any]:
        """
        Executes HTTP POST with exponential backoff retry logic.
        """
        url = f"{self.base_url}{endpoint}"
        attempt = 0
        last_error = None
        
        while attempt < self.retries:
            session = await self._get_session()
            start_ts = time.time()
            try:
                async with session.post(url, json=payload) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    
                    latency = (time.time() - start_ts) * 1000
                    self.logger.debug(f"{endpoint} -> {self.node_id} took {latency:.2f}ms")
                    return data
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                attempt += 1
                last_error = e
                backoff = 2 ** attempt
                self.logger.warning(f"Connection to {self.node_id} failed ({attempt}/{self.retries}). Retrying in {backoff}s... Error: {e}")
                await asyncio.sleep(backoff)
            except Exception as e:
                # Non-recoverable
                self.logger.error(f"Critical error communicating with {self.node_id}: {e}")
                raise e
        
        raise ConnectionError(f"Failed to reach node {self.node_id} after {self.retries} attempts. Last error: {last_error}")

    async def generate(self, prompt: str, system: str = None, model: str = "llama3:8b", **kwargs) -> str:
        """
        Remote generation request.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_predict": kwargs.get("max_tokens", -1)
            }
        }
        if system:
            payload["system"] = system
        if kwargs.get("json_format"):
            payload["format"] = "json"

        try:
            self.logger.info(f"Sending GEN request to {self.node_id} ({model})...")
            resp = await self._post_with_retry("/api/generate", payload)
            return resp.get("response", "")
        except Exception as e:
            self.logger.error(f"Generate failed on {self.node_id}: {e}")
            raise e

    async def embed(self, text: str, model: str = None) -> List[float]:
        """
        Remote embedding request.
        """
        if not model: model = "nomic-embed-text"
        payload = {
            "model": model,
            "prompt": text
        }

        try:
            resp = await self._post_with_retry("/api/embeddings", payload)
            return resp.get("embedding", [])
        except Exception as e:
            self.logger.error(f"Embed failed on {self.node_id}: {e}")
            return []

    async def classify(self, text: str, labels: List[str], model: str = None) -> str:
        """
        Naive prompt-based classification via remote generation.
        """
        if not model: model = "llama3:8b"
        prompt = f"Classify the following text into exactly one of these labels: {labels}.\nText: {text}\nLabel:"
        
        try:
            resp_text = await self.generate(prompt, model=model)
            cleaned = resp_text.strip().lower()
            # Basic fuzzy match
            for label in labels:
                if label.lower() in cleaned:
                    return label
            return labels[0] # Default fallback
        except Exception as e:
            self.logger.error(f"Classify failed on {self.node_id}: {e}")
            return labels[0]

    # Interface compliance
    def get_best_model(self, **kwargs) -> str:
        # Remote node capability discovery is handled by the Router, 
        # so this method acts as a passthrough or queries the remote capability endpoint if implemented.
        # For now, we return a safe default or assume the caller knows the model.
        return "llama3:8b" 
