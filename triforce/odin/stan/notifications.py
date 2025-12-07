import asyncio
import logging
import uuid
import time
import json
from enum import Enum
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

# --- Data Models ---

class AlertSeverity(str, Enum):
    INFO = "info"       # Routine (Scaling up)
    WARNING = "warning" # Threshold breach (High Temp)
    CRITICAL = "critical" # Service Failure (Node Down)
    RECOVERY = "recovery" # Resolved (Node Back Online)

class AlertCategory(str, Enum):
    SYSTEM = "system"
    SECURITY = "security"
    PERFORMANCE = "performance"
    TASK = "task"

class AlertEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    severity: AlertSeverity
    category: AlertCategory
    source: str # e.g. "thor", "autoscaler"
    message: str
    metadata: Dict[str, Any] = {}

# --- Abstract Notifier ---

class Notifier(ABC):
    @abstractmethod
    async def send(self, event: AlertEvent):
        pass

# --- Concrete Notifiers ---

class ConsoleNotifier(Notifier):
    """Prints alerts to the terminal."""
    async def send(self, event: AlertEvent):
        color = "\033[0m" # Reset
        if event.severity == AlertSeverity.CRITICAL: color = "\033[91m" # Red
        elif event.severity == AlertSeverity.WARNING: color = "\033[93m" # Yellow
        elif event.severity == AlertSeverity.RECOVERY: color = "\033[92m" # Green
        
        print(f"{color}[{event.severity.upper()}] {event.source}: {event.message}{chr(27)}[0m")

class WebNotifier(Notifier):
    """Pushes alerts to a dashboard queue (Mocked)."""
    def __init__(self):
        self.queue = []

    async def send(self, event: AlertEvent):
        # In a real app, this would use websockets or push to Redis
        payload = event.dict()
        self.queue.append(payload)
        # print(f"  -> Pushed to Web Dashboard: {event.message}")

class WebhookNotifier(Notifier):
    """Sends payloads to external APIs (discord/slack)."""
    def __init__(self, url: str = None):
        self.url = url

    async def send(self, event: AlertEvent):
        if not self.url:
            return
        # Mock HTTP request
        # print(f"  -> POST {self.url} payload={{...}}")
        pass

# --- Notification Manager ---

class NotificationManager:
    """
    Central hub for alerting.
    Routes events to appropriate channels based on severity.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("stan.notifications")
        self.notifiers: List[Notifier] = []

    def register_notifier(self, notifier: Notifier):
        self.notifiers.append(notifier)

    async def dispatch(self, event: AlertEvent):
        """
        Routes the event to notifiers based on rules.
        """
        tasks = []
        for n in self.notifiers:
            if self._should_route(n, event):
                tasks.append(n.send(event))
        
        if tasks:
            await asyncio.gather(*tasks)

    def _should_route(self, notifier: Notifier, event: AlertEvent) -> bool:
        # 1. Console gets everything
        if isinstance(notifier, ConsoleNotifier):
            return True
            
        # 2. Webhook only gets Critical/Recovery
        if isinstance(notifier, WebhookNotifier):
            return event.severity in [AlertSeverity.CRITICAL, AlertSeverity.RECOVERY]
            
        # 3. Web Dashboard gets Warning+ (skip pure Info spam)
        if isinstance(notifier, WebNotifier):
            return event.severity != AlertSeverity.INFO
            
        return True

# --- Demo Driver ---

async def run_notifications_demo():
    print("--- STAN Notification System ---")
    mgr = NotificationManager()
    
    # 1. Setup Channels
    console = ConsoleNotifier()
    web = WebNotifier()
    webhook = WebhookNotifier(url="https://discord.com/api/webhooks/...")
    
    mgr.register_notifier(console)
    mgr.register_notifier(web)
    mgr.register_notifier(webhook)
    
    # 2. Simulate Events
    events = [
        AlertEvent(
            severity=AlertSeverity.INFO, 
            category=AlertCategory.SYSTEM, 
            source="Autoscaler", 
            message="Scaling up to 4 nodes."
        ),
        AlertEvent(
            severity=AlertSeverity.WARNING, 
            category=AlertCategory.PERFORMANCE, 
            source="Thor", 
            message="GPU Temp at 88C (Threshold 85C)."
        ),
        AlertEvent(
            severity=AlertSeverity.CRITICAL, 
            category=AlertCategory.SYSTEM, 
            source="Loki", 
            message="Heartbeat lost! Node presumed dead."
        ),
        AlertEvent(
            severity=AlertSeverity.RECOVERY, 
            category=AlertCategory.SYSTEM, 
            source="Loki", 
            message="Node reconnection successful. Syncing state."
        ),
        AlertEvent(
            severity=AlertSeverity.INFO, 
            category=AlertCategory.TASK, 
            source="Optimizer", 
            message="Optimization suggestion: Move Task-12 to Node Odin."
        ),
    ]
    
    for e in events:
        await mgr.dispatch(e)
        # Check routing
        log_routing = []
        if mgr._should_route(console, e): log_routing.append("Console")
        if mgr._should_route(web, e): log_routing.append("Web")
        if mgr._should_route(webhook, e): log_routing.append("Webhook")
        # print(f"    (Routed to: {', '.join(log_routing)})")
        await asyncio.sleep(0.2)

if __name__ == "__main__":
    asyncio.run(run_notifications_demo())
