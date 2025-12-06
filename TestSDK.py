
import os
import time
from sdk.client import STANClusterClient

def gpu_task(n):
    return f"Processed {n} items on GPU"

def main():
    print("--- Testing STANClusterClient ---")
    client = STANClusterClient(token="supersecret")
    
    # 1. Submit Job
    print("1. Submitting Job...")
    result = client.submit(gpu_task, 100)
    print(f"   Result: {result}")
    job_id = result.get("job_id")
    
    if job_id:
        # 2. Query Status
        print(f"2. Querying Status for {job_id}...")
        status = client.get_job_status(job_id)
        print(f"   Status: {status.get('status')}")
    
    # 3. Topology
    print("3. Fetching Topology...")
    topo = client.get_cluster_topology()
    print(f"   Nodes: {len(topo)}")
    
    # 4. Optimal Worker
    print("4. Finding Optimal GPU Worker...")
    best = client.get_optimal_gpu_worker()
    if best:
        print(f"   Best Worker: {best['url']} (GPU: {best.get('metrics', {}).get('gpus', [{}])[0].get('name', 'N/A')})")
    else:
        print(f"   No GPU workers found (or they are offline).")

if __name__ == "__main__":
    main()
