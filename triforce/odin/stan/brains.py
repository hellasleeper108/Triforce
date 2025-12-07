import json
import logging
import time
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

# Imports
from triforce.odin.stan.ai_provider import AIProvider

# --- Data Models ---

class TaskNode(BaseModel):
    task_id: str = Field(default="uuid")
    high_level_action: str
    target_nodes: List[str] = []
    required_resources: Dict[str, Any] = {}
    execution_steps: List[Dict[str, Any]] = []
    dependencies: List[str] = []

class TaskGraph(BaseModel):
    request_id: str
    original_command: str
    tasks: List[TaskNode]

class ReflectionReport(BaseModel):
    task_id: str
    success: bool
    efficiency_score: float # 0.0 - 1.0
    anomalies: List[str]
    improvement_suggestions: List[str]

# --- A. Parsing Brain ---

class ParsingBrain:
    """
    Stage 1: Intent Extraction & Task Graph Generation.
    """
    
    SYSTEM_PROMPT = """
    You are the PARSING brain of STAN.
    Your goal: Convert Natural Language commands into a structured JSON Task Graph.
    
    Output Format (JSON):
    {
      "tasks": [
        {
          "task_id": "uuid",
          "high_level_action": "action_name",
          "target_nodes": ["node_name" or "role"],
          "required_resources": {"gpu": bool, "min_vram_gb": int, "platform": "linux/darwin/null"},
          "execution_steps": [{"step_id": "1", "action": "func", "args": {}}],
          "dependencies": []
        }
      ]
    }
    """

    def __init__(self, ai: AIProvider):
        self.ai = ai
        self.logger = logging.getLogger("stan.brain.parsing")

    async def think(self, command: str) -> TaskGraph:
        start_ts = time.time()
        
        # 1. Select Model
        model = self.ai.get_best_model(task_type="generate", min_size_b=7.0)
        self.logger.info(f"Parsing using {model}...")
        
        # 2. Generate
        try:
            response = await self.ai.generate(
                prompt=f"Command: {command}",
                system=self.SYSTEM_PROMPT,
                model=model,
                json_format=True
            )
            data = json.loads(response)
            
            # 3. Validate & Map
            tasks = []
            for t in data.get("tasks", []):
                tasks.append(TaskNode(**t))

            graph = TaskGraph(
                request_id=f"req-{int(start_ts)}",
                original_command=command,
                tasks=tasks
            )
            
            latency = (time.time() - start_ts) * 1000
            self.logger.info(f"Parsed {len(tasks)} tasks in {latency:.2f}ms")
            return graph
            
        except Exception as e:
            self.logger.error(f"Parsing failed: {e}")
            raise e

# --- B. Planning Brain ---

class PlanningBrain:
    """
    Stage 2: Graph refinement & Execution Planning.
    """

    SYSTEM_PROMPT = """
    You are the PLANNING brain of STAN.
    Refine the incoming Task Graph for execution.
    - Add timeouts.
    - Split complex steps if needed.
    - Assign priority.
    """
    
    def __init__(self, ai: AIProvider):
        self.ai = ai
        self.logger = logging.getLogger("stan.brain.planning")

    async def think(self, graph: TaskGraph) -> Dict[str, Any]:
        start_ts = time.time()
        model = self.ai.get_best_model(task_type="generate", min_size_b=13.0)
        self.logger.info(f"Planning using {model}...")
        
        try:
            response = await self.ai.generate(
                prompt=f"Optimize this Task Graph for a distributed cluster:\n{graph.model_dump_json()}",
                system=self.SYSTEM_PROMPT,
                model=model,
                json_format=True
            )
            
            plan = json.loads(response)
            latency = (time.time() - start_ts) * 1000
            self.logger.info(f"Plan generated in {latency:.2f}ms")
            return plan
            
        except Exception as e:
            self.logger.error(f"Planning failed: {e}")
            raise e

# --- C. Reflection Brain ---

class ReflectionBrain:
    """
    Stage 3: Post-Mortem Analysis.
    """

    SYSTEM_PROMPT = """
    You are the REFLECTION brain of STAN.
    Analyze the execution result.
    Output JSON: { "success": bool, "efficiency_score": 0.0-1.0, "anomalies": [], "improvement_suggestions": [] }
    """

    def __init__(self, ai: AIProvider):
        self.ai = ai
        self.logger = logging.getLogger("stan.brain.reflection")

    async def think(self, task_id: str, result: Dict[str, Any], metrics: Dict[str, Any]) -> ReflectionReport:
        start_ts = time.time()
        model = self.ai.get_best_model(task_type="generate", min_size_b=7.0)
        self.logger.info(f"Reflecting using {model}...")
        
        prompt = f"Task: {task_id}\nResult: {result}\nMetrics: {metrics}"
        
        try:
            response = await self.ai.generate(
                prompt=prompt,
                system=self.SYSTEM_PROMPT,
                model=model,
                json_format=True
            )
            data = json.loads(response)
            
            report = ReflectionReport(
                task_id=task_id,
                success=data.get("success", False),
                efficiency_score=data.get("efficiency_score", 0.0),
                anomalies=data.get("anomalies", []),
                improvement_suggestions=data.get("improvement_suggestions", [])
            )
            
            latency = (time.time() - start_ts) * 1000
            self.logger.info(f"Reflection complete in {latency:.2f}ms")
            return report
            
        except Exception as e:
            self.logger.error(f"Reflection failed: {e}")
            raise e
