import asyncio
import logging
import json
import time
from typing import Dict, Any

# --- Imports from all Subsystems ---
from triforce.odin.stan.ai_provider import ProviderConfig, AIProvider
from triforce.odin.stan.supernova import STANSupernova
from triforce.odin.stan.context import ContextManager, ContextSession
from triforce.odin.stan.bootstrapper import Bootstrapper, HardwareProfile
from triforce.odin.stan.ux_orchestrator import UXOrchestrator, UserCommand, CommandSource
from triforce.odin.stan.notifications import NotificationManager, ConsoleNotifier, WebNotifier, AlertEvent, AlertSeverity, AlertCategory
from triforce.odin.stan.empathy import UserEmotion
from triforce.odin.stan.memory import MemoryType

# --- Mock AI for the Narrative ---
class NarrativeMockAI(AIProvider):
    """
    A scripted AI that returns context-aware JSON responses to drive the demo.
    """
    async def generate(self, prompt, system, **kwargs):
        p_lower = prompt.lower()
        s_lower = system.lower()
        
        # Empathy Engine
        if "analyze the user's input" in s_lower:
            return json.dumps({
                "emotion": "focused",
                "confidence": 0.9,
                "intent_summary": "User wants a benchmark.",
                "evidence": ["benchmark", "log everything"]
            })
            
        # Metacognition
        if "critique" in s_lower:
            return json.dumps({
                "score": 0.85,
                "flaws": ["High resource usage potential"],
                "suggestion": "Proceed with monitoring."
            })
            
        # Skill Acquisition / Reflection
        if "summarize" in s_lower and "session" in p_lower:
            return "User onboarded a new A100 node and ran a distributed inference benchmark. Performance was optimal, though a thermal warning was noted."
            
        # Default / Fallback
        return "Operation Proceeding."

    async def embed(self, t): return [0.1] * 64
    async def classify(self, t, l): return l[0]

# --- Validating Log Handler ---
class ValidationHandler(logging.Handler):
    def emit(self, record):
        # We can capture logs here if needed for assertion
        pass

# --- The Simulation Script ---
async def run_day_in_the_life():
    print("\n================================================================")
    print("      STAN: SYSTEM ONLINE - DAY IN THE LIFE SIMULATION")
    print("================================================================\n")
    
    # 1. Initialization
    print(">>> [Phase 1] System Initialization")
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(ValidationHandler())
    
    # Wire Components
    mock_ai = NarrativeMockAI(ProviderConfig(type="mock"))
    
    # Supernova (The Core)
    stan = STANSupernova(use_mock=True)
    stan.ai = mock_ai
    # Inject mock into sub-components for consistency
    stan.metacognition.ai = mock_ai
    
    # Peripheral Systems
    bootstrapper = Bootstrapper()
    notifications = NotificationManager()
    notifications.register_notifier(ConsoleNotifier()) # Visual feedback
    
    # Orchestrator & Context
    # We need to manually wire the empathy engine in UX to use our mock
    ux = UXOrchestrator()
    ux.empathy.ai = mock_ai 
    ux.supernova = stan # Connect UX to the Supernova instance we created
    
    ctx_mgr = ContextManager(stan.memory, mock_ai)
    
    print("   [+] Supernova Core ... Online")
    print("   [+] Neural Interfaces ... Online")
    print("   [+] Cluster Awareness ... Active")
    
    # 2. Node Bootstrapping
    print("\n>>> [Phase 2] Hardware Event: New Node Detected")
    new_hardware = HardwareProfile(
        hostname="thor-gx-1",
        os_name="linux",
        cpu_cores=64,
        ram_gb=512,
        gpu_name="NVIDIA H100",
        vram_gb=80,
        disk_gb=2000
    )
    
    onboarding = bootstrapper.analyze_new_node(new_hardware)
    print(f"   [!] Detected: {new_hardware.hostname} ({new_hardware.gpu_name})")
    print(f"   [!] AI Analysis: Best Role -> {onboarding.target_role.role_type.upper()}")
    print(f"   [!] Generated Manifest: {json.loads(onboarding.manifest_json)['role']}")

    # Simulate saving manifest
    await stan.memory.add_memory(MemoryType.KNOWLEDGE, f"Added node {new_hardware.hostname}", {})

    # 3. User Session Start
    print("\n>>> [Phase 3] User Interaction")
    session = ctx_mgr.start_session(user_id="hella", project="DeepSeek-R1 Benchmarking")
    print(f"   [+] Session Started: {session.session_id} (Project: {session.active_project})")
    
    user_input = "Run a distributed inference benchmark across Odin and Thor, log everything."
    print(f"   [USER] (CLI): \"{user_input}\"")
    
    # 4. UX Orchestration & Intent
    print("\n>>> [Phase 4] STAN Reasoning Pipeline")
    
    # Create Command
    cmd = UserCommand(source=CommandSource.CLI, raw_text=user_input, cleaned_text=user_input)
    
    # UX Processing (Infers State -> Plans -> Executes)
    # We explicitly invoke parts here to show the flow, or let process_user_command do it.
    # Let's let the Orchestrator handle logic:
    response = await ux.process_user_command(cmd)
    
    print(f"   [AI] Inferred State: Focused (Confidence: 0.9)") # Matches Mock
    print(f"   [AI] Response Strategy: Direct, Technical")
    print("   [ Supernova Activity ]") 
    print("       -> Metacognition: Plan verified (Score 0.85).")
    print(f"       -> Agents: Spawning 'Dispatcher' to {new_hardware.hostname}.")
    print(f"       -> Simulation: Predicted 12% probability of thermal throttling.")
    
    ctx_mgr.update_context(session.session_id, user_input, response.message_text)
    
    # 5. Runtime Events (Alerts)
    print("\n>>> [Phase 5] Runtime Telemetry & Alerts")
    
    # Simulate a thermal spike on the new node
    thermal_event = AlertEvent(
        severity=AlertSeverity.WARNING,
        category=AlertCategory.PERFORMANCE,
        source="thor-gx-1",
        message="H100 GPU Junction Temp > 88C (High Load)"
    )
    await notifications.dispatch(thermal_event)
    
    # Simulate Recovery
    recovery_event = AlertEvent(
        severity=AlertSeverity.RECOVERY,
        category=AlertCategory.PERFORMANCE,
        source="thor-gx-1",
        message="Fan curves adjusted. Temp stabilized at 82C."
    )
    await notifications.dispatch(recovery_event)

    # 6. Reflection & Memory
    print("\n>>> [Phase 6] Session Cleanup & Reflection")
    await ctx_mgr.end_session(session.session_id)
    
    # Verify RAG Memory
    print("   [!] Searching Long-Term Memory for session artifact...")
    memories = await stan.memory.search_memory(f"Session Summary {session.session_id}", limit=1)
    if memories:
        print(f"   [RAG] Stored Reflection: \"{memories[0].item.text}\"")
    
    print("\n================================================================")
    print("                  SIMULATION COMPLETE")
    print("================================================================")

if __name__ == "__main__":
    asyncio.run(run_day_in_the_life())
