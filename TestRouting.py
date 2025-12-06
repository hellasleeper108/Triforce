
from triforce.common.models.stan_client import STANClusterClient
import time

def simple_task():
    return "done"

def main():
    client = STANClusterClient(token="supersecret")
    print("--- Testing Advanced Routing ---")

    # 1. Test Compute (Standard)
    print("\n[Submitting 'compute' job]")
    res = client.submit_job(simple_task, [], job_type="compute")
    print(f"Result: {res.get('status')} on {res.get('worker')}")

    # 2. Test GPU Train (Hard Constraint)
    print("\n[Submitting 'gpu_train' job]")
    res = client.submit_job(simple_task, [], job_type="gpu_train")
    if res.get('error'):
         print(f"Result: {res.get('error')}")
    else:
         print(f"Result: {res.get('status')} on {res.get('worker')}")

    # 3. Test GPU Infer (Soft Constraint)
    print("\n[Submitting 'gpu_infer' job]")
    res = client.submit_job(simple_task, [], job_type="gpu_infer")
    print(f"Result: {res.get('status')} on {res.get('worker')}")
    
    # 4. Test IO Heavy
    print("\n[Submitting 'io_heavy' job]")
    res = client.submit_job(simple_task, [], job_type="io_heavy")
    print(f"Result: {res.get('status')} on {res.get('worker')}")

if __name__ == "__main__":
    main()
