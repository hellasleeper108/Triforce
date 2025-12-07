import asyncio
import logging
import json
import time
import uuid
import heapq
from typing import List, Dict, Any, Optional, Set, Tuple
from pydantic import BaseModel, Field

# --- Data Models ---

class MeshObject(BaseModel):
    id: str
    type: str # "model", "embedding", "log"
    data: Any # The actual payload (or reference to it)
    version: int = 0
    updated_at: float = Field(default_factory=time.time)
    origin_node: str

class MeshNode(BaseModel):
    id: str
    address: str
    ram_available_gb: float
    disk_available_gb: float
    latency_ms: float = 10.0
    last_seen: float = Field(default_factory=time.time)

class SyncDelta(BaseModel):
    object_id: str
    version: int
    sender_node: str

# --- Knowledge Mesh ---

class KnowledgeMesh:
    """
    The distributed memory fabric of STAN.
    Synchronizes Knowledge Objects across the cluster.
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.logger = logging.getLogger(f"stan.mesh.{node_id}")
        
        # Local State
        self.objects: Dict[str, MeshObject] = {} # The Knowledge Store
        self.peers: Dict[str, MeshNode] = {}     # Known Peers
        self.sync_queue = asyncio.Queue()        # Incoming updates

    # --- 1. Peer Management ---
    
    def register_peer(self, node: MeshNode):
        self.peers[node.id] = node
        self.logger.info(f"Peer Registered: {node.id} ({node.address})")

    def update_peer_metrics(self, node_id: str, ram_gb: float, disk_gb: float):
        if node_id in self.peers:
            p = self.peers[node_id]
            p.ram_available_gb = ram_gb
            p.disk_available_gb = disk_gb
            p.last_seen = time.time()

    # --- 2. Data Operations ---

    async def put_object(self, obj_type: str, data: Any, obj_id: str = None) -> str:
        """
        Create or Update a local object.
        """
        if not obj_id:
            obj_id = f"{obj_type}-{uuid.uuid4().hex[:8]}"
            
        current = self.objects.get(obj_id)
        new_version = (current.version + 1) if current else 1
        
        obj = MeshObject(
            id=obj_id,
            type=obj_type,
            data=data,
            version=new_version,
            origin_node=self.node_id
        )
        
        self.objects[obj_id] = obj
        # In a real system, we'd broadcast the delta here
        # self.broadcast_update(obj)
        return obj_id

    def get_object(self, obj_id: str) -> Optional[MeshObject]:
        return self.objects.get(obj_id)

    # --- 3. Sync & Conflict Resolution ---

    async def receive_sync(self, obj: MeshObject):
        """
        Handle incoming update from a peer.
        Strategy: Last-Write-Wins (LWW) based on version > timestamp.
        """
        local_obj = self.objects.get(obj.id)
        
        if not local_obj:
            self.logger.info(f"Sync Accept [NEW]: {obj.id} v{obj.version} from {obj.origin_node}")
            self.objects[obj.id] = obj
            return

        # Conflict Resolution
        if obj.version > local_obj.version:
            self.logger.info(f"Sync Accept [UPDATE]: {obj.id} v{obj.version} > v{local_obj.version}")
            self.objects[obj.id] = obj
        elif obj.version == local_obj.version:
             # Tie-break with timestamp
             if obj.updated_at > local_obj.updated_at:
                 self.logger.info(f"Sync Accept [TIE-BREAK]: {obj.id}")
                 self.objects[obj.id] = obj
             else:
                 self.logger.debug(f"Sync Ignore [OLDER]: {obj.id}")
        else:
            self.logger.debug(f"Sync Ignore [STALE]: {obj.id} v{obj.version} < v{local_obj.version}")

    # --- 4. Resource-Aware Routing ---

    def find_best_peer_for_fetch(self, min_ram_gb: float = 0.5) -> Optional[str]:
        """
        Selects the best peer to fetch large data from.
        Criteria: Availability > RAM (Cache potential) > Latency
        """
        candidates = []
        for pid, peer in self.peers.items():
            if peer.ram_available_gb >= min_ram_gb:
                # Score = RAM / Latency (Simple Heuristic)
                score = peer.ram_available_gb / (peer.latency_ms + 1)
                candidates.append((score, pid))
        
        if not candidates:
            return None
            
        # Return peer with highest score
        best = max(candidates, key=lambda x: x[0])
        return best[1]

# --- Simulation Code ---

async def run_mesh_demo():
    print("--- 1. Initializing Mesh Nodes ---")
    odin = KnowledgeMesh("odin-1")
    thor = KnowledgeMesh("thor-1")
    loki = KnowledgeMesh("loki-1")
    
    # Connect them (Full Mesh)
    odin.register_peer(MeshNode(id="thor-1", address="10.0.0.2", ram_available_gb=64, disk_available_gb=500, latency_ms=5))
    odin.register_peer(MeshNode(id="loki-1", address="10.0.0.3", ram_available_gb=128, disk_available_gb=1000, latency_ms=2))
    
    thor.register_peer(MeshNode(id="odin-1", address="10.0.0.1", ram_available_gb=32, disk_available_gb=200, latency_ms=5))
    
    print("\n--- 2. Creating Object on Odin ---")
    model_id = await odin.put_object("model", "Llama-3-Weights-Blob", "model-llama-v1")
    print(f"Odin created {model_id} v1")
    
    print("\n--- 3. Syncing to Thor (Simulation) ---")
    # Simulate network packet
    obj = odin.get_object(model_id)
    await thor.receive_sync(obj)
    
    thor_copy = thor.get_object(model_id)
    print(f"Thor has {thor_copy.id} v{thor_copy.version}")

    print("\n--- 4. Conflict Resolution (Concurrent Updates) ---")
    # Thor updates v1 -> v2
    await thor.put_object("model", "Llama-3-Patch-A", model_id)
    thor_v2 = thor.get_object(model_id)
    
    # Loki (who only had v0/nothing) receives v2 from Thor
    await loki.receive_sync(thor_v2)
    print(f"Loki synced {loki.get_object(model_id).id} v{loki.get_object(model_id).version}")
    
    # Odin (who has v1) receives v2 from Thor
    await odin.receive_sync(thor_v2)
    print(f"Odin updated to v{odin.get_object(model_id).version}")
    
    print("\n--- 5. Smart Routing ---")
    # Odin wants to fetch a huge file. Who is best? Loki has 128GB RAM, Thor 64GB.
    best_peer = odin.find_best_peer_for_fetch(min_ram_gb=10)
    print(f"Best peer for Odin to fetch 10GB object: {best_peer} (Expected: loki-1 due to RAM/Latency)")

if __name__ == "__main__":
    asyncio.run(run_mesh_demo())
