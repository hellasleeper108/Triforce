import socket
import json
import time
import threading
import sys

# Configuration
UDP_PORT = 9999
TTL_SECONDS = 15

class DiscoveryMaster:
    def __init__(self):
        self.registry = {}  # {worker_id: {data: dict, last_seen: float}}
        self.lock = threading.Lock()
        self.running = True

        # Setup UDP Socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Bind to empty string to listen on all interfaces
        self.sock.bind(("", UDP_PORT))
        print(f"[Master] Listening for UDP broadcasts on port {UDP_PORT}")

    def prune_registry(self):
        """Background loop to remove expired workers."""
        while self.running:
            time.sleep(1)
            now = time.time()
            with self.lock:
                expired = []
                for wid, info in self.registry.items():
                    if now - info['last_seen'] > TTL_SECONDS:
                        expired.append(wid)
                
                for wid in expired:
                    print(f"[Master] Worker expired: {wid}")
                    del self.registry[wid]

    def listen(self):
        """Main loop to receive packets."""
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                self.handle_packet(data, addr)
            except Exception as e:
                print(f"[Master] Error receiving: {e}")

    def handle_packet(self, data, addr):
        try:
            msg = json.loads(data.decode('utf-8'))
            worker_id = msg.get('id')
            
            if not worker_id:
                return

            with self.lock:
                is_new = worker_id not in self.registry
                self.registry[worker_id] = {
                    'data': msg,
                    'last_seen': time.time(),
                    'ip': addr[0]
                }
                
            if is_new:
                print(f"[Master] New worker discovered: {worker_id} from {addr[0]}")
            else:
                # Debug log for heartbeat? Optional.
                # print(f"[Master] Heartbeat from {worker_id}")
                pass

        except json.JSONDecodeError:
            print(f"[Master] Received invalid JSON from {addr}")

    def start(self):
        # Start pruner
        t = threading.Thread(target=self.prune_registry, daemon=True)
        t.start()
        
        # Blocking listen
        try:
            self.listen()
        except KeyboardInterrupt:
            print("[Master] Shutting down...")
            self.running = False

if __name__ == "__main__":
    master = DiscoveryMaster()
    master.start()
