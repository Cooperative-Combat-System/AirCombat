import socket
import json
import struct
import numpy as np


class NetworkAdaptor:
    def __init__(self, config):
        self.host = getattr(config, "host", None)
        self.agent_ports = getattr(config, "agent_ports", None)
        self.agent_count = getattr(config, "agent_count", None)
        self.initial_states = getattr(config, "initial_states", None)

        assert self.host is not None, "Missing 'host' in config"
        assert self.agent_ports is not None, "Missing 'agent_ports' in config"
        assert self.agent_count is not None, "Missing 'agent_count' in config"
        assert len(self.agent_ports) == self.agent_count, "Agent count does not match the number of ports"

        self.sockets = {}
        for port in self.agent_ports:
            # print("connect port {}".format(port))
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.host, port))
            self.sockets[port] = sock

        if self.initial_states is None:
            raise ValueError("Initial states must be provided in the config.")
        self.camp_0_states = self.initial_states.get("camp_0", [])
        self.camp_1_states = self.initial_states.get("camp_1", [])



    def send_init_state(self, port, state_dict):
        sock = self.sockets[port]
        json_data = json.dumps(state_dict).encode('utf-8')
        header = struct.pack("II", 6, len(json_data))
        sock.sendall(header + json_data)

    def send_initial_states(self):
        idx_0, idx_1 = 0, 0
        for port, sock in self.sockets.items():
            if idx_0 < len(self.camp_0_states):
                init_state = self.camp_0_states[idx_0]
                idx_0 += 1
            else:
                init_state = self.camp_1_states[idx_1]
                idx_1 += 1
            json_data = json.dumps(init_state).encode('utf-8')
            header = struct.pack("II", 6, len(json_data))
            sock.sendall(header + json_data)

    def receive_all(self,sock, size):
        """持续接收直到拿满 size 字节"""
        data = b''
        while len(data) < size:
            packet = sock.recv(size - len(data))
            if not packet:
                raise RuntimeError("Socket connection lost during recv.")
            data += packet
        return data

    def receive_battlefield_data(self, port=None):
        sock = self.sockets[port]
        header = self.receive_all(sock,8)
        if len(header) < 8:
            raise RuntimeError("Header receive failed.")
        tcp_code, json_len = struct.unpack("II", header)
        if tcp_code == 1 and json_len > 0:
            json_data = self.receive_all(sock,json_len).decode()
            try:
                return json.loads(json_data)
            except json.JSONDecodeError as e:
                print("error")
                raise
        elif tcp_code == 2:
            return {"destroyed": True}
        elif tcp_code == 3:
            return {"round_done": True}
        elif tcp_code == 4:
            return {"training_done": True}
        return None


    def receive_all_battlefield_data(self):
        all_agent_obs = []
        special_flags = {"destroyed": [False] * self.agent_count, "round_done": False, "training_done": False}
        for idx, port in enumerate(self.agent_ports):
            obs = self.receive_battlefield_data(port)
            if obs is None:
                all_agent_obs.append(None)
            elif "destroyed" in obs:
                special_flags["destroyed"][idx] = True
                all_agent_obs.append(None)
            elif "round_done" in obs:
                special_flags["round_done"] = True
                all_agent_obs.append(None)
            elif "training_done" in obs:
                special_flags["training_done"] = True
                all_agent_obs.append(None)
            else:
                all_agent_obs.append(obs)
        return all_agent_obs, special_flags

    def send_action(self, action, port):
        sock = self.sockets[port]
        pay_load = self._to_jsonable(action)
        json_data = json.dumps(pay_load).encode('utf-8')
        header = struct.pack("II", 5, len(json_data))
        sock.sendall(header + json_data)

    def send_actions(self, actions):
        for port, action in zip(self.agent_ports, actions):
            self.send_action(action, port)

    def close(self):
        for sock in self.sockets.values():
            sock.close()

    def reconnect(self):
        for port in self.agent_ports:
            try:
                self.sockets[port].close()
            except Exception as e:
                print(f"Error closing socket {port}: {e}")

            new_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            new_sock.connect((self.host, port))
            self.sockets[port] = new_sock

    def _to_jsonable(self, obj):
        if isinstance(obj, (np.generic,)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.astype(float).tolist()
        if isinstance(obj, list):
            return [self._to_jsonable(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(self._to_jsonable(x) for x in obj)
        if isinstance(obj, dict):
            return {k: self._to_jsonable(v) for k, v in obj.items()}
        return obj