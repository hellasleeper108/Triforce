#!/usr/bin/env python3
import argparse
import asyncio
import sys
import logging
from typing import List

# Import STAN Supernova
# Assuming PYTHONPATH includes project root
from triforce.odin.stan.supernova import STANSupernova

# Configure logging to be less verbose for CLI unless debug is on
logging.basicConfig(level=logging.ERROR)

async def handle_status(args):
    """Show high-level cluster status."""
    print("--- STAN CLUSTER STATUS ---")
    print("Health: [OK]")
    print("Mode:   ENGINEER (Autonomy Level 2)")
    print("Nodes:  3 Active (Odin, Thor, Loki)")
    print("Tasks:  12 Running, 0 Pending")
    print("Alerts: None")

async def handle_nodes(args):
    """List node details."""
    print(f"{'ID':<10} {'ROLE':<10} {'CPU':<8} {'RAM':<8} {'GPU':<8} {'STATUS':<10}")
    print("-" * 60)
    # Mock data relative to what Supernova uses
    nodes = [
        ("odin", "Master", "15%", "40%", "0%", "Healthy"),
        ("thor", "Worker", "80%", "92%", "95%", "High Load"),
        ("loki", "Worker", "20%", "30%", "10%", "Healthy"),
    ]
    for n in nodes:
        print(f"{n[0]:<10} {n[1]:<10} {n[2]:<8} {n[3]:<8} {n[4]:<8} {n[5]:<10}")

async def handle_tasks(args):
    """List active tasks."""
    print(f"{'TASK ID':<15} {'NODE':<10} {'PRIORITY':<10} {'STATE':<10}")
    print("-" * 50)
    print(f"{'task-a192':<15} {'thor':<10} {'HIGH':<10} {'RUNNING'}")
    print(f"{'task-b331':<15} {'odin':<10} {'NORMAL':<10} {'RUNNING'}")

async def handle_exec(args):
    """Execute a natural language command via STAN Supernova."""
    command = " ".join(args.command)
    print(f"> Sending command to STAN: '{command}'")
    
    # Instantiate Supernova (in a real app, this might connect to a daemon)
    stan = STANSupernova(use_mock=True)
    
    # We divert stdout/logging to capture output if needed, 
    # but Supernova logs to stdout/stderr by default.
    # We'll just let it run.
    print("--- STAN RESPONSE ---")
    await stan.act(command)
    print("--- END ---")

async def handle_plan(args):
    """Generate a plan without executing."""
    command = " ".join(args.command)
    print(f"> Planning for: '{command}'")
    
    stan = STANSupernova(use_mock=True)
    # Interact with Assembly engine directly for planning
    # This assumes we want to show the pipeline steps.
    # For now, we reuse act() but maybe we could add a flag to act() later.
    # We will simulate planning by calling metacognition.
    
    critique = await stan.metacognition.critique_plan(command)
    print(f"Metacognition Score: {critique.score}")
    print(f"Potential Flaws: {critique.flaws}")
    if critique.score > 0.5:
        print("Plan Status: VIABLE")
    else:
        print("Plan Status: REJECTED")

async def main():
    parser = argparse.ArgumentParser(description="STAN Control Interface (stanctl)")
    subparsers = parser.add_subparsers(dest="subcommand", help="Command to run")

    # Status
    subparsers.add_parser("status", help="Show cluster health")
    
    # Nodes
    subparsers.add_parser("nodes", help="List nodes and resources")
    
    # Tasks
    subparsers.add_parser("tasks", help="List running tasks")
    
    # Tail Logs (Mock)
    subparsers.add_parser("tail-logs", help="Follow cluster logs")

    # Exec (NL Command)
    exec_parser = subparsers.add_parser("exec", help="Execute natural language command")
    exec_parser.add_argument("command", nargs="+", help="The command string")

    # Plan (NL Command)
    plan_parser = subparsers.add_parser("plan", help="Dry-run a natural language command")
    plan_parser.add_argument("command", nargs="+", help="The command string")

    args = parser.parse_args()

    if args.subcommand == "status":
        await handle_status(args)
    elif args.subcommand == "nodes":
        await handle_nodes(args)
    elif args.subcommand == "tasks":
        await handle_tasks(args)
    elif args.subcommand == "tail-logs":
        print("Streaming logs... (Ctrl+C to stop)")
        try:
            while True:
                await asyncio.sleep(1)
                print("[INFO] Heartbeat from thor")
        except KeyboardInterrupt:
            pass
    elif args.subcommand == "exec":
        await handle_exec(args)
    elif args.subcommand == "plan":
        await handle_plan(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())
