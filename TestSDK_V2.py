
from triforce.common.models.stan_client import STANClusterClient
import time

def demo_function(x):
    import math
    return math.sqrt(x)

def main():
    client = STANClusterClient(token="supersecret")
    
    print("--- Testing STANClusterClient ---")
    
    print("1. Submitting Job...")
    res = client.submit_job(demo_function, [16])
    print(f"   Result: {res}")
    
    print(f"2. Querying Status for {res.get('job_id')}...")
    status = client.get_job_status(res.get('job_id'))
    print(f"   Status: {status.get('status')}")
    
    print("3. Fetching Cluster State...")
    state = client.get_cluster_state()
    print(f"   Nodes: {len(state)}")
    
    print("4. Fetching Global Logs...")
    logs = client.fetch_logs()
    print(f"   Log Count: {len(logs)}")
    
    print("5. Recommending Worker...")
    best = client.recommend_best_worker_for("general")
    print(f"   Recommended: {best}")

if __name__ == "__main__":
    main()
