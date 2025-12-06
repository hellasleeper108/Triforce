# Triforce ⟁

**Triforce** is a distributed Python computing cluster designed for high-performance job scheduling, remote execution, and scaling. It features a master-worker architecture with `ODIN` (Controller) and `THOR` (Worker) nodes.

![Dashboard](https://via.placeholder.com/800x400?text=Triforce+Dashboard+Preview)

## 🚀 Key Features

*   **Distributed Architecture**: decoupled Master/Worker design.
*   **Smart Scheduling**: Weighted load balancing based on CPU, RAM, GPU, and job queue depth.
*   **Real-time Dashboard**: Web interface for monitoring node status, metrics, and logs.
*   **Strict Security**: Role-based token authentication (`controller`, `worker`, `client`) for all communication.
*   **Hardware Aware**: Automatic GPU detection (NVIDIA) and resource reporting.
*   **Python SDK**: Native client ensuring easy integration and function serialization.
*   **Self-Healing**: Automatic worker registration, heartbeats, and dead-node pruning.

## 📦 Components

### 1. ODIN (Controller)
The brain of the cluster. Manages the worker registry, schedules jobs, and serves the dashboard.
*   **Port**: 8080
*   **API**: RESTful (FastAPI)

### 2. THOR (Worker)
The muscle. Executes Python code in a sandboxed environment and reports telemetry.
*   **Port**: 8000 (auto-increments if busy)
*   **Capabilities**: CPU, GPU, RAM monitoring.

### 3. TriforceCTL (CLI)
Command-line tool for cluster management.

## 🛠️ Installation & Usage

### Prerequisites
*   Python 3.9+
*   NVIDIA Drivers (optional, for GPU support)

### Quick Start

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/hellasleeper108/Triforce.git
    cd Triforce
    ```

2.  **Install Dependencies:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
    *(Note: Create a requirements.txt if not present, main deps: fastapi, uvicorn, requests, httpx, psutil, jinja2, pynvml)*

3.  **Set Environment:**
    ```bash
    export API_TOKEN="your-super-secret-token"
    export PYTHONPATH=$PYTHONPATH:.
    ```

4.  **Start ODIN (Controller):**
    ```bash
    python triforce/odin/main.py
    ```

5.  **Start THOR (Worker):**
    ```bash
    # In a new terminal
    export ODIN_URL="http://localhost:8080"
    python triforce/thor/main.py
    ```

6.  **Access Dashboard:**
    Open [http://localhost:8080/dashboard](http://localhost:8080/dashboard)

### Using the SDK

```python
from triforce.common.models.stan_client import STANClusterClient

def my_task(x):
    return x * x

client = STANClusterClient(token="your-super-secret-token")

# Submit job
job = client.submit_job(my_task, [10])
print(f"Job ID: {job['job_id']}")

# Check status
status = client.get_job_status(job['job_id'])
print(f"Result: {status['result']}")
```

## 🔐 Security
All endpoints are protected by `API_TOKEN`. Ensure this environment variable is set on all nodes and clients.

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License
[MIT](https://choosealicense.com/licenses/mit/)
