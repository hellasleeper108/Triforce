import asyncio
import logging
import json
from enum import IntEnum, Enum
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel

class AutonomyLevel(IntEnum):
    APPRENTICE = 0  # Read-only, no actions
    ADVISOR = 1     # Suggests actions, requires approval
    ENGINEER = 2    # Auto-executes LOW/MEDIUM risk, asks for HIGH
    GOVERNOR = 3    # Auto-executes HIGH risk with safety checks
    OVERMIND = 4    # Unrestricted (Use with caution)

class ActionRisk(str, Enum):
    LOW = "low"         # e.g., Cache clear, Log rotation
    MEDIUM = "medium"   # e.g., Restart service, Scale up
    HIGH = "high"       # e.g., Drain node, Drop database, Scale down (data risk)
    CRITICAL = "critical" # e.g., Cluster shutdown, Wipe storage

from enum import Enum

class PermissionResult(BaseModel):
    allowed: bool
    requires_approval: bool
    reason: str

class AutonomyManager:
    """
    The 'Super-Ego' of STAN.
    Enforces rules of engagement based on the current DEFCON/Autonomy level.
    """
    
    def __init__(self, initial_level: AutonomyLevel = AutonomyLevel.APPRENTICE):
        self.current_level = initial_level
        self.logger = logging.getLogger("stan.autonomy")
        self.audit_log: List[Dict[str, Any]] = []

    def set_mode(self, level: AutonomyLevel):
        self.logger.warning(f"Autonomy Mode changed from {self.current_level.name} to {level.name}")
        self.current_level = level

    def check_permission(self, action_name: str, risk: ActionRisk) -> PermissionResult:
        """
        Policy Matrix Logic
        """
        lvl = self.current_level
        
        # 1. Apprentice (Level 0)
        if lvl == AutonomyLevel.APPRENTICE:
            return PermissionResult(allowed=False, requires_approval=True, reason="Apprentice mode: Observe only.")

        # 2. Advisor (Level 1)
        if lvl == AutonomyLevel.ADVISOR:
            return PermissionResult(allowed=False, requires_approval=True, reason="Advisor mode: Recommendations only.")

        # 3. Engineer (Level 2)
        if lvl == AutonomyLevel.ENGINEER:
            if risk in [ActionRisk.LOW, ActionRisk.MEDIUM]:
                return PermissionResult(allowed=True, requires_approval=False, reason="Engineer mode permits Standard ops.")
            return PermissionResult(allowed=False, requires_approval=True, reason="Engineer mode requires approval for High/Critical risk.")

        # 4. Governor (Level 3)
        if lvl == AutonomyLevel.GOVERNOR:
            if risk == ActionRisk.CRITICAL:
                return PermissionResult(allowed=False, requires_approval=True, reason="Governor mode protects Critical assets.")
            return PermissionResult(allowed=True, requires_approval=False, reason="Governor mode permits High risk operations.")

        # 5. Overmind (Level 4)
        if lvl == AutonomyLevel.OVERMIND:
            return PermissionResult(allowed=True, requires_approval=False, reason="Overmind mode: UNRESTRICTED.")

        return PermissionResult(allowed=False, requires_approval=True, reason="Unknown state.")

    async def request_action(self, action: str, risk: ActionRisk, context: str) -> bool:
        """
        Main entry point for Brains to perform actions.
        """
        permit = self.check_permission(action, risk)
        
        entry = {
            "timestamp": time.time(),
            "level": self.current_level.name,
            "action": action,
            "risk": risk,
            "allowed": permit.allowed,
            "approval_req": permit.requires_approval
        }
        self.audit_log.append(entry)
        
        if permit.allowed:
            self.logger.info(f"ALLOWED: {action} ({risk}) [{permit.reason}]")
            return True
        elif permit.requires_approval:
            self.logger.info(f"BLOCKED (Approval Needed): {action} ({risk}) [{permit.reason}]")
            # In a real system, this would trigger a Notify User event
            return False
        else:
             self.logger.info(f"DENIED: {action} ({risk}) [{permit.reason}]")
             return False

import time

# --- Demo Driver ---

async def run_autonomy_demo():
    print("--- 1. Apprentice Mode (Safety First) ---")
    mgr = AutonomyManager(AutonomyLevel.APPRENTICE)
    
    # Try a low risk action
    await mgr.request_action("rotate_logs", ActionRisk.LOW, "Daily maintenance")
    
    print("\n--- 2. Engineer Mode (Standard Ops) ---")
    mgr.set_mode(AutonomyLevel.ENGINEER)
    
    # Try low risk (Should pass)
    await mgr.request_action("rotate_logs", ActionRisk.LOW, "Daily maintenance")
    # Try high risk (Should block)
    await mgr.request_action("drain_node", ActionRisk.HIGH, "Hardware failure suspected")
    
    print("\n--- 3. Overmind Mode (God Mode) ---")
    mgr.set_mode(AutonomyLevel.OVERMIND)
    
    # Try critical risk (Should pass)
    await mgr.request_action("cluster_shutdown", ActionRisk.CRITICAL, "Global reset")

if __name__ == "__main__":
    asyncio.run(run_autonomy_demo())
