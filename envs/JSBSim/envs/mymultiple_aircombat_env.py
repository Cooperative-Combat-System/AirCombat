import numpy as np
from typing import Tuple, Dict, Any
from .env_base import BaseEnv
from ..tasks.mymultiple_aircombat_task import MyMultipleAirCombatTask
from ..core.my_agent import MyAirAgentSim

class MyMultipleAirCombatEnv(BaseEnv):

    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        self._create_records = False
        self.init_states = None
    @property
    def share_observation_space(self):
        return self.task.share_observation_space

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'mymultiple_aircombat':
            self.task = MyMultipleAirCombatTask(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")

    def load_agent(self):
        self._air_agent = {}
        port_list = self.config.agent_ports
        initial_states = self.config.initial_states

        # 生成 ego_ids 和 enm_ids 模拟 uid 命名规则，如 A00, B00
        camp0_size = len(initial_states["camp_0"])
        camp1_size = len(initial_states["camp_1"])
        self.ego_ids = [f"A{i:02d}" for i in range(camp0_size)]
        self.enm_ids = [f"B{i:02d}" for i in range(camp1_size)]
        all_agent_ids = self.ego_ids + self.enm_ids

        # 预创建 agent 并挂载初始状态
        for idx, (agent_id, port) in enumerate(zip(all_agent_ids, port_list)):
            camp = 0 if agent_id in self.ego_ids else 1
            unit_idx = self.ego_ids.index(agent_id) if camp == 0 else self.enm_ids.index(agent_id)
            init_state = initial_states[f"camp_{camp}"][unit_idx]
            agent = MyAirAgentSim(agent_id, port, self.task.adaptor)
            agent.init_state = init_state
            self._air_agent[agent_id] = agent
            agent.initialize(init_state)

        # 设置伙伴和敌人
        for agent_id, agent in self._air_agent.items():
            agent.partners = [self._air_agent[pid] for pid in
                              (self.ego_ids if agent_id in self.ego_ids else self.enm_ids) if pid != agent_id]
            agent.enemies = [self._air_agent[eid] for eid in
                             (self.enm_ids if agent_id in self.ego_ids else self.ego_ids)]

    def load(self):
        self.load_task()
        self.load_agent()
        self.seed()

    def reload_agent(self):
        if self.init_states is None:
            self.init_states = {agent_id: agent.init_state.copy() for agent_id, agent in self.agents.items()}
        for agent_id, agent in self.agents.items():
            agent.reload(self.init_states[agent_id])

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        self.current_step = 0
        self.reload_agent()
        self.task.reset(self)
        obs = self.get_obs()
        share_obs = self.get_state()
        return self._pack(obs), self._pack(share_obs)

    @property
    def agents(self) -> Dict[str, MyAirAgentSim]:
        return self._air_agent

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        self.current_step += 1
        info = {"current_step": self.current_step}

        unpacked_action = self._unpack(action)
        for agent_id, agent in self._air_agent.items():
            norm_action = self.task.normalize_action(self, agent_id, unpacked_action[agent_id])
            self.task.adaptor.send_action(norm_action, agent.port)

        self.task.step(self)
        for agent in self.agents.values():
            agent.observe()

        obs = self.get_obs()
        share_obs = self.get_state()

        rewards = {}
        for agent_id in self.agents.keys():
            reward, info = self.task.get_reward(self, agent_id, info)
            rewards[agent_id] = [reward]
        ego_reward = np.mean([rewards[ego_id] for ego_id in self.ego_ids])
        enm_reward = np.mean([rewards[enm_id] for enm_id in self.enm_ids])
        for ego_id in self.ego_ids:
            rewards[ego_id] = [ego_reward]
        for enm_id in self.enm_ids:
            rewards[enm_id] = [enm_reward]

        dones = {}
        for agent_id in self.agents.keys():
            done, info = self.task.get_termination(self, agent_id, info)
            dones[agent_id] = [done]

        return self._pack(obs), self._pack(share_obs), self._pack(rewards), self._pack(dones), info