
from triforce.common.models.stan_client import STANClusterClient
import concurrent.futures
import time

def slow_task(x):
    import time
    time.sleep(0.5)
    return x * x

def main():
    client = STANClusterClient(token="supersecret")
    print("--- Running Load Test to verify Worker Distribution ---")
    
    nodes = client.get_cluster_state()
    active_nodes = [n for n in nodes if n['status'] == 'ACTIVE']
    print(f"Active Workers: {len(active_nodes)}")
    for n in active_nodes:
        print(f" - {n.get('specs', {}).get('worker_name')} ({n['url']})")

    # Submit 10 jobs
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        for i in range(10):
            futures.append(executor.submit(client.submit_job, slow_task, [i]))
    
    results = [f.result() for f in futures]
    
    # Analyze distribution
    workers_hit = {}
    for r in results:
        w = r.get('worker', 'unknown')
        workers_hit[w] = workers_hit.get(w, 0) + 1
        
    print("\n--- Distribution Results ---")
    for w, count in workers_hit.items():
        print(f"{w}: {count} jobs")

if __name__ == "__main__":
    main()
