import typer
import requests
import os
import sys
import json
import time # Added
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich import print as rprint
from typing import Optional

app = typer.Typer(help="Triforcectl: Command Line Interface for ODIN/THOR Cluster")
console = Console()

# Configuration
ODIN_URL = os.getenv("ODIN_URL", "http://localhost:8080")
API_TOKEN = os.getenv("CLIENT_TOKEN", os.getenv("API_TOKEN", "default-insecure-token"))

@app.callback()
def main(
    url: str = typer.Option(None, envvar="ODIN_URL", help="ODIN Master URL"),
    token: str = typer.Option(None, envvar=["CLIENT_TOKEN", "API_TOKEN"], help="Authentication Token")
):
    global ODIN_URL, API_TOKEN
    if url:
        ODIN_URL = url
    if token:
        API_TOKEN = token

def get_client():
    from sdk.client import STANCluster
    client = STANCluster(odin_url=ODIN_URL, token=API_TOKEN)
    client.session.headers.update({"X-Role": "client"})
    return client

def get_nodes_api():
    try:
        resp = requests.get(f"{ODIN_URL}/nodes", timeout=2)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error connecting to ODIN at {ODIN_URL}:[/bold red] {e}")
        sys.exit(1)

@app.command()
def nodes():
    """List all registered worker nodes"""
    client = get_client()
    try:
        # Client uses session with headers
        resp = client.session.get(f"{client.odin_url}/nodes", timeout=2)
        resp.raise_for_status()
        nodes = resp.json()
    except Exception as e:
        console.print(f"[bold red]Error checking nodes:[/bold red] {e}")
        return

    table = Table(title="Cluster Nodes")
    table.add_column("URL", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Last Seen", style="magenta")
    table.add_column("Jobs", justify="right")
    table.add_column("GPU", style="yellow")

    for n in nodes:
        gpu_info = "N/A"
        metrics = n.get("metrics", {})
        
        # New Format
        if "gpu" in metrics:
             usage = metrics.get('gpu', 0)
             temp = metrics.get('gpu_temp', 0)
             if usage > 0 or temp > 0:
                 gpu_info = f"{usage}% ({temp}°C)"
        # Fallback for old Format (if mixed cluster)
        elif "gpus" in metrics and metrics["gpus"]:
             gpu = metrics["gpus"][0]
             gpu_info = f"{gpu['usage_percent']}% ({gpu['temperature_c']}°C)"
             
        table.add_row(
            n["url"],
            n["status"],
            f"{time.time() - n['last_seen']:.1f}s ago",
            str(n["active_jobs"]),
            gpu_info
        )
    
    console.print(table)

@app.command()
def submit(file: str, args: str = ""):
    """Submit a python file as a job"""
    if not os.path.exists(file):
        console.print(f"[bold red]File {file} not found![/bold red]")
        return
        
    with open(file, "r") as f:
        code = f.read()
        
    parsed_args = []
    if args:
        for a in args.split(","):
            try:
                parsed_args.append(int(a))
            except:
                parsed_args.append(a)
    
    payload = {
        "code": code,
        "entrypoint": "main",
        "args": parsed_args
    }
    
    console.print(f"Submitting job from [bold]{file}[/bold]...")
    client = get_client()
    try:
        resp = client.session.post(f"{client.odin_url}/submit", json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        
        console.print_json(data=result)
        
        if result.get("status") == "COMPLETED":
             console.print(f"[bold green]Success! Result: {result['result']}[/bold green]")
        else:
             console.print(f"[bold red]Failed! Error: {result.get('error')}[/bold red]")
             
    except Exception as e:
        console.print(f"[bold red]Submission failed: {e}[/bold red]")

@app.command()
def topology():
    """
    Visualize cluster topology.
    """
    nodes_data = get_nodes_api()
    
    tree = Tree(f"[bold blue]ODIN Master[/bold blue] ({ODIN_URL})")
    
    workers_branch = tree.add("Workers")
    
    for node in nodes_data:
        url = node.get('url')
        status = node.get('status')
        icon = "🟢" if status == "ACTIVE" else "🔴"
        workers_branch.add(f"{icon} [bold]{url}[/bold] ({status})")

    console.print(tree)

@app.command()
def logs(job_id: str):
    """
    Get logs for a job (Mock).
    """
    console.print(f"[dim]Fetching logs for {job_id}...[/dim]")
    console.print("[yellow]Feature not implemented on server side yet.[/yellow]")

@app.command()
def restart(node_url: str):
    """
    Trigger a restart of a node (Mock).
    """
    console.print(f"[red]Restarting {node_url}... (Mock)[/red]")

if __name__ == "__main__":
    app()
