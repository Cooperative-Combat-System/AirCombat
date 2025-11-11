from typing import Dict, Any, Tuple
import numpy as np
from .env_base import BaseEnv
from ..core.my_agent import MyAirAgentSim
from ..utils.adaptor import NetworkAdaptor
from ..tasks import MyCombatTask


class AirCombatEnv(BaseEnv):
    def __init__(self, config_name: str):
        super().__init__(config_name)
        self.current_step = 0
        self.init_states = None

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

    @property
    def agents(self) -> Dict[str, MyAirAgentSim]:
        return self._air_agent

    def reset(self) -> np.ndarray:
        print('Resetting the environment.')
        self.current_step = 0
        self.task.adaptor.reconnect()
        self.reload_agent()
        self.task.reset(self)
        for agent in self.agents.values():
            obs = agent.observe()
            if isinstance(obs,dict) and obs.get("round_done"):
                print(f"Round done detected in agent {agent.agent_id}. Resetting...")
            if isinstance(obs,dict) and obs.get("training_done"):
                print(f"Training done detected in agent {agent.agent_id}. Resetting...")
                agent.done = True
        obs = self.get_obs()
        return self._pack(obs)

    def reload_agent(self):
        if self.init_states is None:
            self.init_states = {agent_id: agent.init_state.copy() for agent_id, agent in self.agents.items()}
        for agent_id, agent in self.agents.items():
            agent.reload(self.init_states[agent_id])

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        self.current_step += 1
        info = {"current_step": self.current_step}

        # 解包动作
        unpacked_action = self._unpack(action)
        for agent_id, agent in self._air_agent.items():
            norm_action = self.task.normalize_action(self, agent_id, unpacked_action[agent_id])
            self.task.adaptor.send_action(norm_action, agent.port)

        self.task.step(self)
        round_done_flags = []
        training_done_flags = []
        for agent in self.agents.values():
            obs = agent.observe()
            if isinstance(obs,dict) and obs.get("round_done"):
                round_done_flags.append(True)
            else:
                round_done_flags.append(False)

            if isinstance(obs,dict) and obs.get("training_done"):
                training_done_flags.append(True)
            else:
                training_done_flags.append(False)

        obs = self.get_obs()
        global_done = all(round_done_flags) or any(training_done_flags)

        dones = {}
        for agent_id, agent in self.agents.items():
            done, info = self.task.get_termination(self, agent_id, info)
            dones[agent_id] = [global_done or done]

        rewards = {}
        for agent_id, agent in self.agents.items():
            reward, info = self.task.get_reward(self, agent_id, info)
            rewards[agent_id] = [reward]

        return self._pack(obs), self._pack(rewards), self._pack(dones), info

    def close(self):
        self.task.adaptor.close()

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'mycombat':
            self.task = MyCombatTask(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")
