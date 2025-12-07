import asyncio
import logging
import json
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel
from collections import Counter

from triforce.odin.stan.ai_provider import AIProvider

class PipelineStep(BaseModel):
    name: str
    model_selector: str # "tiny", "huge", or specific model name like "llama3:70b"
    system_prompt: str
    temperature: float = 0.7
    fallback_models: List[str] = []

class EnsembleConfig(BaseModel):
    models: List[str]
    strategy: str = "majority_vote" # "majority_vote", "arbitration"
    arbitrator_model: Optional[str] = None # For "arbitration" strategy

class AssemblyReport(BaseModel):
    final_output: str
    trace: List[Dict[str, Any]]
    ensemble_stats: Optional[Dict[str, Any]] = None

class AssemblyEngine:
    """
    Constructs complex reasoning architectures from atomic models.
    Supports:
    - Serial Pipelines (Chain of Thought across models)
    - Parallel Ensembles (Voting/Consistency)
    - Fallback Chains (Resilience)
    """

    def __init__(self, ai_provider: AIProvider):
        self.ai = ai_provider
        self.logger = logging.getLogger("stan.assembly")

    async def run_pipeline(self, steps: List[PipelineStep], initial_input: str) -> AssemblyReport:
        """
        Executes a sequence of steps where output of N becomes context for N+1.
        """
        current_context = initial_input
        trace = []
        
        for i, step in enumerate(steps):
            self.logger.info(f"Pipeline Step {i+1}/{len(steps)}: {step.name} ({step.model_selector})")
            
            # Resolve model (Primitive resolution for now, integration with Switcher in full system)
            # Here we assume model_selector is passed directly or mapped simply
            target_model = step.model_selector
            
            output = None
            error = None
            
            # Try Main + Fallbacks
            candidates = [target_model] + step.fallback_models
            
            for model_name in candidates:
                try:
                    self.logger.debug(f"Attempting execution with {model_name}...")
                    output = await self.ai.generate(
                        prompt=current_context,
                        system=step.system_prompt,
                        model=model_name,
                        temperature=step.temperature
                    )
                    break # Success
                except Exception as e:
                    self.logger.warning(f"Model {model_name} failed: {e}")
                    error = e
            
            if output is None:
                raise RuntimeError(f"Pipeline failed at step '{step.name}': All candidates failed. Last error: {error}")
            
            trace.append({
                "step": step.name,
                "model": model_name,
                "input_snippet": current_context[:50],
                "output": output
            })
            
            # Chain output to next input
            # For simple chains, output becomes input. 
            # Complex chains might append. We'll append for context retention.
            current_context = f"{current_context}\n\n[Previous Step Output]:\n{output}"

        return AssemblyReport(final_output=trace[-1]["output"], trace=trace)

    async def run_ensemble(self, config: EnsembleConfig, prompt: str, system: str) -> AssemblyReport:
        """
        Runs multiple models in parallel and synthesizes result.
        """
        self.logger.info(f"Running Ensemble ({config.strategy}) with {len(config.models)} models.")
        
        # 1. Parallel Execution
        tasks = []
        for m in config.models:
            tasks.append(self.ai.generate(prompt=prompt, system=system, model=m))
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter failures
        valid_outputs = []
        trace = []
        for i, res in enumerate(results):
            model_name = config.models[i]
            if isinstance(res, Exception):
                self.logger.error(f"Ensemble member {model_name} failed: {res}")
                trace.append({"model": model_name, "error": str(res)})
            else:
                valid_outputs.append(res)
                trace.append({"model": model_name, "output": res})
        
        if not valid_outputs:
            raise RuntimeError("Ensemble failed: No valid outputs.")

        # 2. Synthesis
        final_out = ""
        stats = {}
        
        if config.strategy == "majority_vote":
            # Naive exact match voting (unlikely for LLMs)
            # Better: Semantic clustering. For MVP: simple most common string.
            # OR simple selection of first valid if high variance.
            # Let's do a "Reviewer" pass if no consensus, or just return first.
            # Actually, let's implement the "Reflective Arbitration" requested.
            
            # For simple voting, we need exact matches. LLMs vary. 
            # So "majority_vote" on text is weak. 
            # We'll treat "majority_vote" as "concatenate and arbitrate" effectively 
            # unless outputs are tiny (like classification labels).
            # If short outputs, we count.
            if len(valid_outputs[0]) < 50:
                 counts = Counter(valid_outputs)
                 final_out, count = counts.most_common(1)[0]
                 stats["vote_confidence"] = count / len(valid_outputs)
            else:
                 # Fallback to arbitration for long text
                 config.strategy = "arbitration"
        
        if config.strategy == "arbitration":
            if not config.arbitrator_model:
                raise ValueError("Arbitration strategy requires 'arbitrator_model'.")
                
            judge_prompt = "You are a Judge. Synthesize the following model outputs into a single best answer.\n\n"
            for i, out in enumerate(valid_outputs):
                judge_prompt += f"--- Model {i+1} ---\n{out}\n\n"
                
            final_out = await self.ai.generate(
                prompt=judge_prompt,
                system="Synthesize the best answer. Correct errors.",
                model=config.arbitrator_model
            )
            stats["arbitrator"] = config.arbitrator_model

        return AssemblyReport(final_output=final_out, trace=trace, ensemble_stats=stats)

# --- Examples ---

async def run_assembly_demo():
    from triforce.odin.stan.ai_provider import AIProviderFactory, ProviderConfig
    
    # Mock Provider
    class AssemblyMock(AIProvider):
        async def generate(self, prompt, system, model, **kwargs):
            if "tiny" in model: return "Sort list."
            if "huge" in model: 
                if "Judge" in prompt: return "Final Answer: Use Timsort."
                return "Python's Timsort is efficient."
            return "Generic response."
            
        async def embed(self, t): return []
        async def classify(self, t, l): return l[0]

    engine = AssemblyEngine(AssemblyMock(ProviderConfig(type="mock")))
    
    # 1. Pipeline Example: Planner -> Coder
    print("--- Pipeline Demo ---")
    pipe = [
        PipelineStep(name="Plan", model_selector="tiny", system_prompt="Outline steps."),
        PipelineStep(name="Code", model_selector="huge", system_prompt="Write code for steps.")
    ]
    res = await engine.run_pipeline(pipe, "Write a sort function.")
    print(f"Result: {res.final_output}")
    print(f"Trace: {[step['model'] for step in res.trace]}")

    # 2. Ensemble Example: Voting
    print("\n--- Ensemble Demo (Arbitration) ---")
    ens = EnsembleConfig(
        models=["tiny", "small", "medium"], 
        strategy="arbitration", 
        arbitrator_model="huge"
    )
    res = await engine.run_ensemble(ens, "Best sort?", "Answer briefly.")
    print(f"Result: {res.final_output}")
    print(f"Stats: {res.ensemble_stats}")

if __name__ == "__main__":
    asyncio.run(run_assembly_demo())
