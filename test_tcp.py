import socket
import json
import struct
import random
import time

# ====== CONFIG ======
HOST = "10.119.13.38"
PORTS = [1002, 1003]
AGENT_IDS = ["A00", "B00"]

# 示例初始状态（可从 YAML 读取）
INITIAL_STATES = {
    "A00": {
    "camp_idx": 0,
    "camp_unit_idx": 0,
    "position": [0, 0, 100],
    "rotation": [0, 0, 0],
    "linear_v": [10, 0, 0],
    "angular_v": [0, 0, 0]
    },
    "B00": {
    "camp_idx": 1,
    "camp_unit_idx": 0,
    "position": [200, 0, 100],
    "rotation": [0, 0, 3.14],
    "linear_v": [-10, 0, 0],
    "angular_v": [0, 0, 0]
    }
}

# ====== TCP UTILS ======
def recv_all(sock, size):
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Socket closed.")
        data += chunk
    return data

def send_json(sock, tcp_code, obj):
    json_bytes = json.dumps(obj).encode()
    header = struct.pack("II", tcp_code, len(json_bytes)) # little-endian
    sock.sendall(header + json_bytes)

def recv_json(sock):
    header = recv_all(sock, 8)
    tcp_code, json_len = struct.unpack("II", header)
    json_bytes = recv_all(sock, json_len)
    json_data = json.loads(json_bytes.decode())
    return tcp_code, json_data
# ====== AGENT CLASS ======
class TestAgent:
    def __init__(self, agent_id, port):
        self.agent_id = agent_id
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((HOST, port))
        print(f"[{self.agent_id}] Connected to port {port}")

    def send_init_state(self, init_state):
        print(f"[{self.agent_id}] Sending init...")
        send_json(self.sock, 6, init_state)

    def observe(self):
        code, data = recv_json(self.sock)
        print(f"[{self.agent_id}] Observed:")
        print(json.dumps(data, indent=2))
        return data

    def send_action(self):
        action = {
        "throttle": random.uniform(0.3, 1.0),
        "rudder": random.uniform(-0.5, 0.5),
        "elevator": random.uniform(-0.5, 0.5),
        "aileron": random.uniform(-0.5, 0.5),
        "flap": 0.0,
        "is_done": False
        }
        send_json(self.sock, 5, action)
        print(f"[{self.agent_id}] Sent action: {action}")

    def close(self):
        self.sock.close()
        print(f"[{self.agent_id}] Socket closed.")
# ====== MAIN TEST LOOP ======
def test_loop():
    agents = []
    for i, agent_id in enumerate(AGENT_IDS):
        agent = TestAgent(agent_id, PORTS[i])
        agent.send_init_state(INITIAL_STATES[agent_id])
        agents.append(agent)

    for agent in agents:
        agent.observe()
    time.sleep(1.0)

    try:
        for step in range(1000):
            print(f"\n====== Step {step} ======")
            for agent in agents:
                agent.send_action()
            for agent in agents:
                agent.observe()

            time.sleep(0.2)
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        for agent in agents:
            agent.close()

if __name__ == "__main__":
    test_loop()