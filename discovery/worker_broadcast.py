import socket
import json
import time
import uuid
import sys

# Configuration
UDP_PORT = 9999
BROADCAST_INTERVAL = 5

class DiscoveryWorker:
    def __init__(self, service_name="worker"):
        self.worker_id = str(uuid.uuid4())[:8]
        self.service_name = service_name
        
        # Setup UDP Socket for Broadcast
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        print(f"[Worker] ID: {self.worker_id} initialized.")

    def run(self):
        print(f"[Worker] Starting broadcast loop (Interval: {BROADCAST_INTERVAL}s)")
        try:
            while True:
                payload = {
                    "id": self.worker_id,
                    "service": self.service_name,
                    "timestamp": time.time(),
                    "status": "READY"
                }
                msg = json.dumps(payload).encode('utf-8')
                
                # Send to Broadcast Address
                self.sock.sendto(msg, ('<broadcast>', UDP_PORT))
                print(f"[Worker] Sent broadcast packet")
                
                time.sleep(BROADCAST_INTERVAL)
        except KeyboardInterrupt:
            print("[Worker] Stopping...")

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "default-worker"
    worker = DiscoveryWorker(name)
    worker.run()
