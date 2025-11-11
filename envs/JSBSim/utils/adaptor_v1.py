import socket
import json
import struct
import numpy as np


def get_packet(tcp_socket, packet_size):
    data = b''
    while len(data) < packet_size:
        packet = tcp_socket.recv(packet_size - len(data))
        data += packet
    return data


def send_packet(tcp_socket, packet_format, data):
    packed_data = struct.pack(packet_format, *data)
    tcp_socket.send(packed_data)

def pack_init_state(init_state):
    integer_observation = init_state.astype(np.int32)
    room = 1615727
    unit = 0
    initial_packet = np.array([room], dtype=np.int32)
    initial_packet = np.append(initial_packet, unit)
    initial_packet = np.append(initial_packet, integer_observation)
    return initial_packet

# 为每一个agent设置一个adaptor
class NetworkAdaptorV1:

    INITIAL_PACKET_FORMAT = "<26i296x"
    GETTING_PACKET_FORMAT = "=27d"
    SENDING_PACKET_FORMAT = "<5d"
    INITIAL_PACKET_SIZE = 400
    GETTING_PACKET_SIZE = 216
    SENDING_PACKET_SIZE = 40

    def __init__(self,config):
        self.host = getattr(config, "host", None)
        self.agent_ports = getattr(config, "agent_ports", None)
        self.agent_count = getattr(config, "agent_count", None)

        assert self.host is not None, "Missing 'host' in config"
        assert self.agent_ports is not None, "Missing 'agent_ports' in config"
        assert self.agent_count is not None, "Missing 'agent_count' in config"
        assert len(self.agent_ports) == self.agent_count, "Agent count does not match the number of ports"

        self.sockets = {}
        for port in self.agent_ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.host, port))
            self.sockets[port] = sock


    def send_init_state(self, port, init_state):
        sock = self.sockets[port]
        init_packet = pack_init_state(init_state)
        send_packet(sock, self.INITIAL_PACKET_FORMAT, init_packet)

    def receive_observations(self, port):
        sock = self.sockets[port]
        data = get_packet(sock, self.GETTING_PACKET_SIZE)
        unpacked_data = np.array(struct.unpack(self.GETTING_PACKET_FORMAT, data), dtype=np.float64)
        return unpacked_data

    def send_action(self, action, port):
        sock = self.sockets[port]
        send_packet(sock, self.SENDING_PACKET_FORMAT, action)

    def close(self):
        for sock in self.sockets.values():
            sock.close()

    def reconnect(self):
        for port in self.agent_ports:
            try:
                self.sockets[port].close()
            except Exception as e:
                print(f"Failed to close socket on port {port}: {e}")

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




