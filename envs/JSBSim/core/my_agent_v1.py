import numpy as np
def float_to_bool(f):
    """
    将发送来的is_done信号转换为bool类型
    :param f: is_done信号即terminated信号
    :return:bool类型
    """
    if f == 0.0:
        # print("False!")
        return False
    elif f == 1.0:
        # print("True!")
        return True
    else:
        # Should not be here!
        print(f)
        print("Float comparison error!")
        return None

def split_observation(observation):
    """
    拆分观测值，包括己方状态，敌方状态，终止信号
    :param observation: xyz + uvw + v_xyz + v_uvw + hp + enemey_state + is_done
    :return:
    """
    my_state = observation[0:13].astype(np.float64).copy()
    enemy_state = observation[13:26].astype(np.float64).copy()
    terminated = float_to_bool(observation[26])
    return my_state, enemy_state, terminated

class MyAirAgentSimV1:
    def __init__(self, agent_id, port, network_adaptor, init_state):
        """
        初始化智能体。
        :param agent_id: 智能体编号（如 0, 1,...）
        :param port: 与该智能体通信的 TCP 端口。
        :param network_adaptor: NetworkAdaptor 实例。
        """
        self.agent_id = agent_id
        self.port = port
        self.adaptor = network_adaptor
        self.my_state = None
        self.enemy_state = None
        self.done = False  # 是否摧毁/结束
        self.partners = []
        self.enemies = []
        self.init_state = init_state

    def initialize(self):
        """
        用于首次加载（或外部设定 init_state）。
        """
        print("Initializing agent {} with init state {}".format(self.agent_id, self.init_state))
        return self.reload(self.init_state)

    def reload(self, new_state=None):
        """
        重新加载智能体状态，用于 reset/restart。
        """
        if new_state is not None:
            self.init_state = new_state
        self.done = False
        self.my_state = self.init_state[0:12].astype(np.float32).copy()
        self.enemy_state = self.init_state[12:24].astype(np.float32).copy()
        # 发送初始状态并接收初始观测
        self.adaptor.send_init_state(self.port, self.init_state)

    def observe(self):
        """
        接收新观测数据并更新状态。
        """
        result = self.adaptor.receive_observations(self.port)
        my_state, enemy_state, terminated = split_observation(result)
        self.my_state = my_state
        self.enemy_state = enemy_state
        self.done = terminated
