import asyncio
import logging
import json
import time
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel

from triforce.odin.stan.ai_provider import AIProvider
from triforce.odin.stan.memory import MemoryEngine, MemoryType, RetrievalResult

class SkillCategory(str, Enum):
    ARCHITECTURAL = "architectural"
    PROCEDURAL = "procedural"
    OPTIMIZATION = "optimization"
    COMMUNICATION = "communication"
    RECOVERY = "recovery"

class Skill(BaseModel):
    name: str
    category: SkillCategory
    trigger_condition: str
    action_logic: str
    confidence: float
    source_reflections: List[str] # IDs or summaries of reflections that led to this

class SkillReport(BaseModel):
    new_skills: List[Skill]
    improved_skills: List[Skill]
    analysis_summary: str

class SkillManager:
    """
    The 'Hippocampal Replay' mechanism for STAN.
    Consolidates daily experiences into abstract skills.
    """
    
    SYSTEM_PROMPT = """
    You are the LEARNING brain of STAN.
    Your goal is to extract reusable SKILLS from execution logs.
    
    Input: A list of "Reflection" logs describing failures or successes.
    Output: Abstracted rules or patterns (SKILLS) to apply in future.
    
    Skill Categories:
    - architectural (design patterns)
    - procedural (how to execute steps)
    - optimization (speed/cost)
    - recovery (how to fix errors)
    
    Format JSON:
    {
      "skills": [
         {
           "name": "Aggressive Sharding",
           "category": "optimization",
           "trigger_condition": "Input size > 10GB",
           "action_logic": "Split into 1GB chunks instead of 5GB",
           "confidence": 0.95
         }
      ]
    }
    """

    def __init__(self, ai: AIProvider, memory: MemoryEngine):
        self.ai = ai
        self.memory = memory
        self.logger = logging.getLogger("stan.skills")
        self.knowledge_base: Dict[str, Skill] = {} # In-memory cache of skills

    async def run_nightly_learning(self) -> SkillReport:
        """
        Analyzes recent history to discover new skills.
        """
        self.logger.info("Starting Nightly Skill Acquisition Cycle...")
        
        # 1. Fetch recent reflections
        # In a real system, we'd query by timestamp. 
        # Here we search for "Reflection" to simulate "all recent cognitive logs".
        reflections = await self.memory.search_memory(
            "Reflection", 
            type_filter=MemoryType.LONG_TERM, 
            limit=20
        )
        
        if not reflections:
            return SkillReport(new_skills=[], improved_skills=[], analysis_summary="No reflections found.")

        # 2. Synthesis
        reflection_text = "\n".join([f"- {r.item.text}" for r in reflections])
        prompt = (
            f"Analyze these execution logs:\n{reflection_text}\n\n"
            "Identify what worked, what failed, and synthesize reusable SKILLS."
        )
        
        try:
            resp = await self.ai.generate(prompt=prompt, system=self.SYSTEM_PROMPT, json_format=True)
            data = json.loads(resp)
            
            new_skills = []
            for s_data in data.get("skills", []):
                skill = Skill(**s_data, source_reflections=[r.item.text[:50] for r in reflections[:3]]) # loose linkage
                
                # Deduplication logic (naive)
                if skill.name not in self.knowledge_base:
                    self.knowledge_base[skill.name] = skill
                    new_skills.append(skill)
                    
                    # Persist as Knowledge
                    await self.memory.add_memory(
                        MemoryType.KNOWLEDGE,
                        f"LEARNED SKILL: {skill.name}. Trigger: {skill.trigger_condition}. Logic: {skill.action_logic}",
                        {"category": skill.category.value, "type": "skill"}
                    )
            
            return SkillReport(
                new_skills=new_skills, 
                improved_skills=[], 
                analysis_summary=f"Synthesized {len(new_skills)} new skills from {len(reflections)} logs."
            )

        except Exception as e:
            self.logger.error(f"Skill acquisition failed: {e}")
            raise e

    async def get_relevant_skills(self, context: str) -> List[Skill]:
        """
        Retrieves skills relevant to a current task.
        """
        # RAG Search for "Learning Skill"
        hits = await self.memory.search_memory(f"LEARNED SKILL {context}", limit=3)
        relevant = []
        for h in hits:
            if "LEARNED SKILL" in h.item.text:
                # Naive parsing or just assume specific format
                # In prod, we'd store structured ID and lookup in self.knowledge_base
                # Here we just construct a transient object for display
                relevant.append(Skill(
                    name="Retrieved Skill",
                    category=SkillCategory.PROCEDURAL,
                    trigger_condition="Context Match",
                    action_logic=h.item.text,
                    confidence=h.similarity,
                    source_reflections=[]
                ))
        return relevant


# --- Demo Driver ---

async def run_skill_demo():
    from triforce.odin.stan.ai_provider import AIProviderFactory, ProviderConfig
    
    # Mock Provider
    class SkillMock(AIProvider):
        sys_log = []
        async def generate(self, prompt, system, **kwargs):
            self.sys_log.append(system[:20])
            # Return a skill describing sharding strategy
            return json.dumps({
                "skills": [
                    {
                        "name": "Dynamic Sharding V2",
                        "category": "optimization",
                        "trigger_condition": "Task Memory > 32GB",
                        "action_logic": "Force partition into 4GB chunks to avoid OOM on Loki.",
                        "confidence": 0.98
                    }
                ]
            })
            
        async def embed(self, t): return [0.1] * 128
        async def classify(self, t, l): return l[0]

    ai = SkillMock(ProviderConfig(type="mock"))
    mem = MemoryEngine(ai, db_path=":memory:")
    learner = SkillManager(ai, mem)
    
    print("--- 1. Simulating Experience (Failures & Validations) ---")
    await mem.add_memory(MemoryType.LONG_TERM, "Reflection: Task A (50GB) failed on Loki due to OOM.", {})
    await mem.add_memory(MemoryType.LONG_TERM, "Reflection: Task B (50GB) succeeded when manually split into 10 chunks.", {})
    
    print("--- 2. Running Nightly Learning Cycle ---")
    report = await learner.run_nightly_learning()
    print(f"Summary: {report.analysis_summary}")
    for s in report.new_skills:
        print(f" [+] Learned: {s.name}")
        print(f"     Trigger: {s.trigger_condition}")
        print(f"     Action:  {s.action_logic}")
        
    print("\n--- 3. Verifying Knowledge Persistence ---")
    skills = await learner.get_relevant_skills("Large memory task")
    if skills:
        print(f" [OK] Retrieved relevant skill: {skills[0].action_logic}")
    else:
        print(" [FAIL] No skills found.")

if __name__ == "__main__":
    asyncio.run(run_skill_demo())
