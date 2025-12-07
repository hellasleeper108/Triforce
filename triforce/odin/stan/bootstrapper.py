import asyncio
import logging
import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# --- Data Models ---

class HardwareProfile(BaseModel):
    hostname: str
    os_name: str # "linux", "darwin"
    cpu_cores: int
    ram_gb: float
    gpu_name: Optional[str] = None
    vram_gb: float = 0.0
    disk_gb: float

class NodeRole(BaseModel):
    role_type: str # "master", "worker-gpu", "worker-cpu", "storage"
    services: List[str]
    description: str

class OnboardingPlan(BaseModel):
    target_role: NodeRole
    required_packages: List[str]
    env_vars: Dict[str, str]
    setup_script: str
    manifest_json: str

# --- Bootstrapping Assistant ---

class Bootstrapper:
    """
    Helps onboard new nodes into the Triforce Cluster.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("stan.bootstrapper")

    def analyze_new_node(self, profile: HardwareProfile) -> OnboardingPlan:
        """
        Determines role and requirements based on hardware.
        """
        role = self._determine_role(profile)
        
        # Base requirements for all
        packages = ["python3", "python3-pip", "git", "curl"]
        env = {
            "STAN_MASTER_HOST": "10.0.0.1", # Default placeholder
            "STAN_NODE_ID": profile.hostname
        }
        
        # Role specifics
        if role.role_type == "worker-gpu":
            packages.extend(["nvidia-cuda-toolkit", "nvidia-driver-535"])
            env["STAN_ACCELERATOR"] = "cuda"
        
        manifest = {
            "id": profile.hostname,
            "role": role.role_type,
            "hw": profile.dict()
        }
        
        script = self._render_script(profile, role, packages, env)
        
        return OnboardingPlan(
            target_role=role,
            required_packages=packages,
            env_vars=env,
            setup_script=script,
            manifest_json=json.dumps(manifest, indent=2)
        )

    def _determine_role(self, hw: HardwareProfile) -> NodeRole:
        # 1. GPU Worker?
        if hw.gpu_name and hw.vram_gb >= 8:
            return NodeRole(
                role_type="worker-gpu", 
                services=["thor", "ollama"], 
                description=f"AI Inference Node ({hw.gpu_name})"
            )
        
        # 2. Master Candidate? (High RAM/CPU, no GPU needed)
        if hw.ram_gb >= 32 and hw.cpu_cores >= 8 and not hw.gpu_name:
             return NodeRole(
                role_type="master", 
                services=["odin", "redis", "minio"], 
                description="Cluster Controller"
            )

        # 3. Storage? (High Disk)
        if hw.disk_gb >= 1000:
             return NodeRole(
                role_type="storage", 
                services=["minio", "loki"], 
                description="Data Persistence Node"
            )
            
        # Default: CPU Worker
        return NodeRole(
            role_type="worker-cpu", 
            services=["loki"], 
            description="General Purpose Worker"
        )

    def _render_script(self, hw: HardwareProfile, role: NodeRole, pkgs: List[str], env: Dict[str, str]) -> str:
        lines = ["#!/bin/bash", "# STAN Onboarding Script", "set -e", ""]
        
        lines.append(f"echo '>>> Onboarding {hw.hostname} as {role.role_type}...'")
        
        # Env Vars
        lines.append("\n# Environment")
        for k, v in env.items():
            lines.append(f"export {k}={v}")
            lines.append(f"echo 'export {k}={v}' >> ~/.bashrc")
            
        # Apt Install
        lines.append("\n# Dependencies")
        lines.append("sudo apt-get update")
        lines.append(f"sudo apt-get install -y {' '.join(pkgs)}")
        
        # Service Start
        lines.append("\n# Start Services")
        for svc in role.services:
            lines.append(f"echo 'Starting {svc} service...'")
            # Mock service start command
            lines.append(f"# ./bin/{svc} --start &")
            
        lines.append("\necho '>>> Node Ready. Broadcasting heartbeat...'")
        return "\n".join(lines)

# --- Demo Driver ---

async def run_bootstrap_demo():
    print("--- STAN Node Bootstrapper ---")
    bs = Bootstrapper()
    
    # Scene 1: New GPU Monster
    gpu_node = HardwareProfile(
        hostname="thor-2",
        os_name="linux",
        cpu_cores=32,
        ram_gb=128,
        gpu_name="NVIDIA A100",
        vram_gb=80,
        disk_gb=500
    )
    
    plan_gpu = bs.analyze_new_node(gpu_node)
    print(f"\n[Analysis] {gpu_node.hostname}: Identified as {plan_gpu.target_role.role_type.upper()}")
    print(f"           ({plan_gpu.target_role.description})")
    print(f"[Script Preview]:")
    print("\n".join(plan_gpu.setup_script.split("\n")[:10])) # Show head
    print("...")

    # Scene 2: Raspberry Pi Helper
    pi_node = HardwareProfile(
        hostname="rpi-worker-1",
        os_name="linux",
        cpu_cores=4,
        ram_gb=4,
        gpu_name=None,
        disk_gb=32
    )
    
    plan_pi = bs.analyze_new_node(pi_node)
    print(f"\n[Analysis] {pi_node.hostname}: Identified as {plan_pi.target_role.role_type.upper()}")
    print(f"           ({plan_pi.target_role.description})")

if __name__ == "__main__":
    asyncio.run(run_bootstrap_demo())
