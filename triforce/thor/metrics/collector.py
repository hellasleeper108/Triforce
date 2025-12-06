import psutil
import time
from triforce.thor.utils.logger import logger

try:
    import pynvml
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    logger.warning("pynvml not installed, GPU metrics disabled")

def init_gpu():
    if HAS_GPU:
        try:
            pynvml.nvmlInit()
            logger.info("NVIDIA Management Library initialized")
        except Exception as e:
            logger.warning(f"Failed to init NVML: {e}")

# State for IO rate calculation
_last_io = {
    "disk": None,
    "net": None,
    "time": 0
}

def get_system_metrics(start_time, node_id, active_jobs_count):
    global _last_io
    
    # CPU & RAM
    cpu_usage = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    
    # IO Counters (Cumulative)
    try:
        current_disk = psutil.disk_io_counters()
        current_net = psutil.net_io_counters()
        current_time = time.time()
        
        # Calculate Rates (MB/s)
        disk_io_rate = 0.0
        net_io_rate = 0.0
        
        if _last_io["time"] > 0 and current_time > _last_io["time"]:
            delta = current_time - _last_io["time"]
            
            if _last_io["disk"]:
                disk_bytes = (current_disk.read_bytes + current_disk.write_bytes) - \
                             (_last_io["disk"].read_bytes + _last_io["disk"].write_bytes)
                disk_io_rate = (disk_bytes / 1024 / 1024) / delta # MB/s
                
            if _last_io["net"]:
                net_bytes = (current_net.bytes_sent + current_net.bytes_recv) - \
                            (_last_io["net"].bytes_sent + _last_io["net"].bytes_recv)
                net_io_rate = (net_bytes / 1024 / 1024) / delta # MB/s
        
        # Update State
        _last_io["disk"] = current_disk
        _last_io["net"] = current_net
        _last_io["time"] = current_time
        
    except Exception as e:
        logger.error(f"IO Metrics Error: {e}")
        disk_io_rate = 0.0
        net_io_rate = 0.0

    # GPU (First GPU only for simplified metrics)
    gpu_usage = 0.0
    gpu_mem = {"used": 0, "total": 0}
    gpu_temp = 0
    
    if HAS_GPU:
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                gpu_usage = float(util.gpu)
                gpu_mem = {
                    "used": mem_info.used // 1024 // 1024, # MB
                    "total": mem_info.total // 1024 // 1024 # MB
                }
                gpu_temp = temp
        except pynvml.NVMLError as ne:
             logger.warning(f"NVML Error during metrics collection: {ne}")
        except Exception as e:
            logger.error(f"Failed to collect GPU metrics: {e}")
            
    return {
        "worker": "THOR",
        "node_id": node_id, # Keeping node_id for discovery/metadata
        "uptime": int(time.time() - start_time),
        "cpu": cpu_usage,
        "ram": mem.percent,
        "disk_io_mbps": round(disk_io_rate, 2),
        "net_io_mbps": round(net_io_rate, 2),
        "gpu": gpu_usage,
        "gpu_mem": gpu_mem,
        "gpu_temp": gpu_temp,
        "active_jobs": int(active_jobs_count)
    }

def get_gpu_specs():
    gpus = []
    total_mem = 0
    if HAS_GPU:
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                 handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                 name = pynvml.nvmlDeviceGetName(handle)
                 mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                 total_mem += mem.total
                 if isinstance(name, bytes):
                     name = name.decode("utf-8")
                 gpus.append({"index": i, "name": name})
        except Exception:
            pass
    return gpus, HAS_GPU, total_mem // 1024 // 1024 # MB

def get_hardware_specs():
    return {
        "cpu_cores": psutil.cpu_count(logical=True),
        "gpu_mem_total": 0 # Will be populated by get_gpu_specs call if needed, or we just call get_gpu_specs logic here.
    }
# Refactoring to just return specs tuple

