import logging
import json
import time
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# Imports
from triforce.odin.stan.ai_provider import AIProvider, ProviderConfig, ModelConfig
from triforce.odin.stan.memory import MemoryEngine, MemoryType, RetrievalResult

class ImprovementInsight(BaseModel):
    category: str # "heuristics", "models", "resources"
    observation: str
    suggested_action: str
    confidence: float

class NightlyReport(BaseModel):
    timestamp: float
    tasks_analyzed: int
    success_rate: float
    insights: List[ImprovementInsight]
    system_summary: str

class EvolutionManager:
    """
    The 'Subconscious' of STAN.
    Periodically digests history to refine heuristics and prompts.
    """
    
    SYSTEM_PROMPT = """
    You are the EVOLUTION brain of STAN.
    Analyze the provided execution history and memory logs.
    Identify patterns of failure or inefficiency.
    Suggest concrete improvements to heuristics or model selection.
    
    Output JSON:
    {
        "insights": [
             {"category": "heuristics", "observation": "...", "suggested_action": "...", "confidence": 0.9}
        ],
        "system_summary": "One sentence summary."
    }
    """

    def __init__(self, ai: AIProvider, memory: MemoryEngine):
        self.ai = ai
        self.memory = memory
        self.logger = logging.getLogger("stan.evolution")

    async def run_nightly_cycle(self) -> NightlyReport:
        """
        The main loop. In prod, run this once every 24h.
        """
        self.logger.info("Starting Nightly Evolution Cycle...")
        
        # 1. Fetch History (Last 24h)
        # We simulate fetching by searching for "Reflection" logs
        recent_reflections = await self.memory.search_memory(
            query="Reflection", 
            type_filter=MemoryType.LONG_TERM, 
            limit=50
        )
        
        if not recent_reflections:
             self.logger.info("No history to analyze.")
             return NightlyReport(
                 timestamp=time.time(), tasks_analyzed=0, success_rate=0, insights=[], system_summary="No data."
             )

        # 2. Analyze
        # Aggregate stats
        failures = 0
        low_efficiency = 0
        texts = []
        for r in recent_reflections:
            # Parse text payload if possible (it's stored as plain text log usually)
            # RAG Memory stores: "Reflection on task-123: {'success': True...}"
            text = r.item.text
            texts.append(text)
            if "'success': False" in text or '"success": false' in text.lower():
                failures += 1
            if "efficiency_score" in text:
                 # Extraction - naive
                 pass 

        rate = 1.0 - (failures / len(recent_reflections))
        
        # 3. AI Insight Generation
        history_blob = "\n".join(texts[-10:]) # Summarize last 10 for prompt context
        prompt = (
            f"Analyze these recent execution reflections:\n{history_blob}\n\n"
            f"Total Analyzed: {len(texts)}. Failure Rate: {failures/len(texts):.2f}.\n"
            "Generate improvement insights."
        )
        
        try:
            resp = await self.ai.generate(
                 prompt=prompt,
                 system=self.SYSTEM_PROMPT,
                 model="llama3", # Use a decent model
                 json_format=True
            )
            data = json.loads(resp)
            
            insights = [ImprovementInsight(**i) for i in data.get("insights", [])]
            summary = data.get("system_summary", "Optimization complete.")
            
            # 4. Apply Improvements (Stub)
            # In a real system, this would write to a config file or update variables
            for i in insights:
                if i.confidence > 0.8:
                    self.logger.info(f"Applying Auto-Improvement: {i.suggested_action}")
                    # e.g. self.config.timeout += 10
            
            # 5. Store Report
            report = NightlyReport(
                timestamp=time.time(),
                tasks_analyzed=len(recent_reflections),
                success_rate=rate,
                insights=insights,
                system_summary=summary
            )
            
            # Memorize the Report itself
            await self.memory.add_memory(
                MemoryType.KNOWLEDGE, 
                f"Nightly Evolution Report: {summary}", 
                {"type": "report", "insights": len(insights)}
            )
            
            return report

        except Exception as e:
            self.logger.error(f"Evolution failed: {e}")
            raise e


# --- Example Driver ---

async def run_evolution_demo():
    from triforce.odin.stan.ai_provider import AIProviderFactory, ProviderConfig
    
    # Setup Mocks
    ai = AIProviderFactory.create(ProviderConfig(type="mock"))
    
    # Need a memory engine with some data
    from triforce.odin.stan.memory import MemoryEngine
    mem = MemoryEngine(ai, db_path=":memory:")
    
    # Inject fake history
    print("--- Seeding Memories ---")
    await mem.ingest_log({"level": "INFO", "message": "Reflection on t1: {'success': True, 'efficiency': 0.9}"})
    await mem.ingest_log({"level": "INFO", "message": "Reflection on t2: {'success': False, 'error': 'OOM on Loki'}"})
    await mem.ingest_log({"level": "INFO", "message": "Reflection on t3: {'success': True, 'efficiency': 0.5, 'note': 'Slow scheduling'}"})
    
    # Run Evolution
    evo = EvolutionManager(ai, mem)
    
    print("\n--- Running Nightly Cycle ---")
    # Mock AI response to ensure valid JSON for demo
    # We rely on the MockProvider returning "mocked response" usually, or we can patch it.
    # Let's subclass MockAI for this run to give good JSON
    class SmartMock(ai.__class__):
        async def generate(self, prompt, system, **kwargs):
            return json.dumps({
                "insights": [
                    {
                        "category": "resources",
                        "observation": "Loki consistently fails with OOM.",
                        "suggested_action": "Decrease max_batch_size for Loki to 4.",
                        "confidence": 0.95
                    },
                     {
                        "category": "heuristics",
                        "observation": "Scheduling latency is high.",
                        "suggested_action": "Switch planning model to 'fast-instruct'.",
                        "confidence": 0.8
                    }
                ],
                "system_summary": "Detected resource bottlenecks on edge nodes."
            })
    
    evo.ai = SmartMock(ProviderConfig(type="mock"))
    
    report = await evo.run_nightly_cycle()
    print(f"Report Summary: {report.system_summary}")
    print(f"Success Rate: {report.success_rate:.2f}")
    print("Insights:")
    for i in report.insights:
        print(f" - [{i.category.upper()}] {i.suggested_action} (Conf: {i.confidence})")

if __name__ == "__main__":
    asyncio.run(run_evolution_demo())
