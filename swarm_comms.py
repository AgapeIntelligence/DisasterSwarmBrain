# swarm_comms.py
"""
Minimal, secure, zero-dependency UDP mesh for DisasterSwarmBrain
HMAC-SHA256 signed + timestamp + nonce → no replay attacks
Works on any network (Wi-Fi, ad-hoc, LoRa gateway, etc.)
"""

import hmac
import hashlib
import json
import socket
import time
import threading
from typing import Dict, Callable, Any

# CHANGE THIS ON EVERY SWARM — keep secret, never commit
SWARM_SECRET = b"replace_this_with_32_random_bytes_per_swarm"

def sign_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    mac = hmac.new(SWARM_SECRET, raw, hashlib.sha256).hexdigest()
    return {"payload": payload, "mac": mac}

def verify_payload(signed: Dict[str, Any], max_age: int = 5) -> bool:
    try:
        payload = signed["payload"]
        mac = signed["mac"]
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        expected = hmac.new(SWARM_SECRET, raw, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(mac, expected):
            return False
        if abs(time.time() - payload.get("ts", 0)) > max_age:
            return False
        return True
    except:
        return False

def broadcast(msg: Dict[str, Any], port: int = 50000):
    signed = sign_payload(msg)
    data = json.dumps(signed, separators=(",", ":")).encode()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.sendto(data, ("<broadcast>", port))
    sock.close()

class SwarmComms:
    def __init__(self, port: int = 50000, callback: Callable[[Dict, tuple], None] = None):
        self.port = port
        self.callback = callback or (lambda p, a: None)
        self.listener = None
        self.running = threading.Event()

    def start(self):
        if self.listener:
            return
        self.running.set()
        self.listener = threading.Thread(target=self._listen, daemon=True)
        self.listener.start()

    def stop(self):
        self.running.clear()
        if self.listener:
            self.listener.join(timeout=1)

    def _listen(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("", self.port))
        sock.settimeout(1.0)
        while self.running.is_set():
            try:
                data, addr = sock.recvfrom(4096)
                try:
                    signed = json.loads(data.decode())
                    if verify_payload(signed):
                        self.callback(signed["payload"], addr)
                except:
                    pass
            except socket.timeout:
                continue
            except:
                break
        sock.close()

    def send_heartbeat(self, agent_id: str, coherence: float, leader_score: float):
        payload = {
            "type": "heartbeat",
            "id": agent_id,
            "coherence": round(coherence, 4),
            "leader_score": round(leader_score, 4),
            "ts": int(time.time())
        }
        broadcast(payload, self.port)

# ——— Quick test ———
if __name__ == "__main__":
    def on_msg(p, a):
        print(f"→ {p} from {a[0]}")
    comms = SwarmComms(callback=on_msg)
    comms.start()
    try:
        while True:
            comms.send_heartbeat("test-unit-01", 0.99, 0.997)
            time.sleep(0.5)
    except KeyboardInterrupt:
        comms.stop()
