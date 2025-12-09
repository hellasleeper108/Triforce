import aiohttp
import json
import logging
import time
import asyncio
from typing import List, Optional, Dict, Any, Union, AsyncGenerator

from triforce.odin.stan.ai_provider import AIProvider, ProviderConfig

class OllamaProvider(AIProvider):
    """
    Robust backend for Ollama (http://localhost:11434).
    Supports streaming, advanced params, and detailed logging.
    """
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.logger = logging.getLogger("stan.ai.ollama")
        self.base_url = config.base_url.rstrip('/')
        
        # Auto-select valid model if default is missing
        available = self.get_available_models()
        valid_names = [m["name"] for m in available]
        
        if config.default_model not in valid_names and valid_names:
            self.logger.warning(f"Default model '{config.default_model}' not found. Switching to '{valid_names[0]}'.")
            self.config.default_model = valid_names[0]

    async def generate(self, prompt: str, system: Optional[str] = None, model: Optional[str] = None, **kwargs) -> str:
        """
        Standard non-streaming generation. Returns the full text response.
        """
        response_dict = await self._call_generate_api(prompt, system, model, stream=False, **kwargs)
        return response_dict["text"]

    async def generate_stream(self, prompt: str, system: Optional[str] = None, model: Optional[str] = None, **kwargs) -> AsyncGenerator[str, None]:
        """
        Streaming generation. Yields text chunks.
        """
        async for chunk in self._stream_generate_api(prompt, system, model, **kwargs):
            yield chunk

    async def embed(self, text: str) -> List[float]:
        """
        Generate vector embeddings.
        """
        start_ts = time.time()
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.config.default_embed_model,
            "prompt": text
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=self.config.timeout_sec) as resp:
                    if resp.status != 200:
                        err = await resp.text()
                        self.logger.error(f"Embed failed ({resp.status}): {err}")
                        raise Exception(f"Ollama Embed Error: {resp.status}")
                    
                    data = await resp.json()
                    latency = (time.time() - start_ts) * 1000
                    self.logger.debug(f"Embed success ({latency:.2f}ms)")
                    return data.get("embedding", [])
                    
        except asyncio.TimeoutError:
            self.logger.error("Embed timed out.")
            return []
        except Exception as e:
            self.logger.error(f"Embed connection failed: {e}")
            return []

    async def classify(self, text: str, labels: List[str]) -> str:
        """
        Constrained generation for classification.
        """
        labels_str = ", ".join([f'"{l}"' for l in labels])
        prompt = (
            f"Classify the following text into exactly one category.\n"
            f"Categories: [{labels_str}]\n"
            f"Text: \"{text}\"\n"
            f"Result (Output ONLY the category name):"
        )
        
        # Use low temperature for deterministic output
        result = await self.generate(
            prompt, 
            model=self.config.default_model, 
            options={"temperature": 0.1, "num_predict": 10}
        )
        
        clean_result = result.strip().strip('"').strip("'").lower()
        
        for l in labels:
            if l.lower() in clean_result:
                return l
        
        self.logger.warning(f"Classification ambiguous. Raw: '{clean_result}'. Defaulting to {labels[0]}.")
        return labels[0]

    def set_active_model(self, model_name: str) -> bool:
        """Sets the default model for generation."""
        # For Ollama, we trust the user if the model exists in the list we fetch
        self.logger.info(f"Switching active model to: {model_name}")
        self.config.default_model = model_name
        return True
        
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Returns list of available models from Ollama API."""
        import requests
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = []
                for m in data.get("models", []):
                    name = m.get("name")
                    models.append({
                        "name": name,
                        "active": name == self.config.default_model,
                        "platform": "gpu", # Assume local means GPU/CPU hybrid
                        "capabilities": ["generate"] # Basic assumption
                    })
                return models
        except Exception as e:
            self.logger.error(f"Failed to fetch models from Ollama: {e}")
        
        # Fallback to config if API fails
        return super().get_available_models()

    # --- Internal Helpers ---

    async def _call_generate_api(self, prompt: str, system: Optional[str], model: Optional[str], stream: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Internal method handling the HTTP request logic, error handling, and metric tracking.
        Returns unified dict: { "text": str, "raw": dict, "latency_ms": float }
        """
        target_model = model or self.config.default_model
        url = f"{self.base_url}/api/generate"
        
        # Extract known parameters
        options = kwargs.get("options", {})
        # Support direct passing of top-level params if not in options
        for key in ["temperature", "num_ctx", "seed", "top_k", "top_p", "repeat_penalty"]:
            if key in kwargs:
                options[key] = kwargs[key]
        
        payload = {
            "model": target_model,
            "prompt": prompt,
            "stream": stream,
            "options": options
        }
        
        if system:
            payload["system"] = system
        if kwargs.get("json_format"):
            # Ollama expects "format": "json"
            payload["format"] = "json"

        start_ts = time.time()
        retries = 2
        
        for attempt in range(retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, timeout=self.config.timeout_sec) as resp:
                        if resp.status != 200:
                            err_text = await resp.text()
                            raise Exception(f"HTTP {resp.status}: {err_text}")
                        
                        data = await resp.json()
                        latency = (time.time() - start_ts) * 1000
                        
                        self.logger.info(f"Generated {len(data.get('response', ''))} chars via {target_model} in {latency:.2f}ms")
                        
                        text_response = data.get("response", "")

                        # Fallback for models that fail with strict JSON format
                        if not text_response and payload.get("format") == "json":
                             self.logger.warning(f"Model {target_model} returned empty response with format='json'. Retrying without strict format...")
                             payload.pop("format")
                             # Retry immediately (nested to avoid loop complexity)
                             async with session.post(url, json=payload, timeout=self.config.timeout_sec) as retry_resp:
                                 if retry_resp.status == 200:
                                     retry_data = await retry_resp.json()
                                     text_response = retry_data.get("response", "")
                                     self.logger.info(f"Retry success: Generated {len(text_response)} chars.")
                        
                        return {
                            "text": text_response,
                            "raw": data,
                            "latency_ms": latency
                        }
                        
            except Exception as e:
                self.logger.warning(f"Attempt {attempt+1}/{retries+1} failed: {e}")
                if attempt == retries:
                    self.logger.error(f"All attempts failed for {target_model}.")
                    raise e
                await asyncio.sleep(0.5 * (attempt + 1)) # Backoff

        return {"text": "", "raw": {}, "latency_ms": 0.0}

    async def _stream_generate_api(self, prompt: str, system: Optional[str], model: Optional[str], **kwargs) -> AsyncGenerator[str, None]:
        target_model = model or self.config.default_model
        url = f"{self.base_url}/api/generate"
        
        options = kwargs.get("options", {})
        payload = {
            "model": target_model,
            "prompt": prompt,
            "stream": True,
            "options": options
        }
        if system: payload["system"] = system
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                         raise Exception(f"Stream HTTP {resp.status}")
                         
                    async for line in resp.content:
                        if line:
                            try:
                                chunk = json.loads(line)
                                token = chunk.get("response", "")
                                yield token
                                if chunk.get("done", False):
                                    break
                            except:
                                continue
            except Exception as e:
                self.logger.error(f"Stream failed: {e}")
                yield f"[System Error: {e}]"
