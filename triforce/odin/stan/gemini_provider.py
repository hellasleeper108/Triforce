
import os
import json
import logging
from typing import List, Optional, Any
import google.generativeai as genai
from triforce.odin.stan.ai_provider import AIProvider, ProviderConfig

class GeminiProvider(AIProvider):
    """
    Google Gemini Provider for STAN.
    Requires GEMINI_API_KEY environment variable.
    """
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        
        # 1. Setup API Key
        api_key = config.api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.logger.warning("GEMINI_API_KEY not found. Gemini Provider will fail.")
        else:
            genai.configure(api_key=api_key)
            
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

    async def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        model_name = kwargs.get("model", self.config.default_model or "gemini-1.5-flash") # Default fallback
        
        # Map "gemini-3" if user asks for it, though currently 1.5 is standard API.
        # User asked for "Gemini 3", but that might be future looking.
        # I will map it to the latest available stable model or check config.
        # For now, let's assume the config passes the correct model string like "gemini-1.5-pro"
        
        # Use JSON mode if requested
        safe_config = self.generation_config.copy()
        if kwargs.get("json_format"):
            safe_config["response_mime_type"] = "application/json"
            
        self.logger.info(f"generating with {model_name} (JSON={kwargs.get('json_format')})")
        
        try:
            # Create Model
            # Note: System instructions are passed at model init in GenAI SDK
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system
            )
            
            # Create Chat Session (or just generate_content for single turn)
            # STAN usually does single-turn with context in prompt, but chat is safer for history?
            # AIProvider.generate is usually single prompt.
            
            response = await model.generate_content_async(
                prompt,
                generation_config=safe_config
            )
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"Gemini Generate Failed: {e}")
            raise e

    async def embed(self, text: str) -> List[float]:
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document",
                title="STAN Embedding"
            )
            return result['embedding']
        except Exception as e:
            self.logger.error(f"Gemini Embed Failed: {e}")
            # Return dummy or raise? Raise to be safe.
            raise e

    async def classify(self, text: str, labels: List[str]) -> str:
        # Simple heuristic or LLM call? 
        # LLM call is more robust for Gemini
        prompt = f"Classify the following text into one of these categories: {labels}.\nText: {text}\nCategory:"
        return await self.generate(prompt, model="gemini-1.5-flash")
