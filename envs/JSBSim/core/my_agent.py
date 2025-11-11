class MyAirAgentSim:
    def __init__(self, agent_id, port, network_adaptor):
        """
        初始化智能体。
        :param agent_id: 智能体编号（如 0, 1,...）
        :param port: 与该智能体通信的 TCP 端口。
        :param network_adaptor: NetworkAdaptor 实例。
        """
        self.agent_id = agent_id
        self.port = port
        self.adaptor = network_adaptor
        self.last_raw_obs = None
        self.done = False  # 是否摧毁/结束
        self.partners = []
        self.enemies = []
        self.init_state = None

    def initialize(self, init_state):
        """
        用于首次加载（或外部设定 init_state）。
        """
        self.init_state = init_state
        print("Initializing agent {} with init state {}".format(self.agent_id, init_state))
        return self.reload(init_state)

    def reload(self, new_state=None):
        """
        重新加载智能体状态，用于 reset/restart。
        """
        if new_state is not None:
            self.init_state = new_state
        self.done = False
        self.last_raw_obs = None
        # 发送初始状态并接收初始观测
        self.adaptor.send_init_state(self.port, self.init_state)



    def observe(self):
        """
        接收新观测数据并更新状态。
        """
        result = self.adaptor.receive_battlefield_data(self.port)
        if result is None:
            self.last_raw_obs = None
            return None

        # 判断是否特殊标志包
        if isinstance(result, dict) and ("destroyed" in result or "round_done" in result or "training_done" in result):
            self.done = True
            self.last_raw_obs = [result]  # 关键：即使是 dict，也转成 list
        elif isinstance(result, list):
            self.last_raw_obs = result
        else:
            self.last_raw_obs = [result]  # 正常单架单位也转 list，保证结构统一

        return self.last_raw_obs