import numpy as np
from gym import spaces
from typing import Tuple
import torch
from sympy.physics.units.systems.si import all_units

from ..tasks import MyCombatTask
from ..core.my_agent import MyAirAgentSim
from ..utils.adaptor import NetworkAdaptor


class MyMultipleAirCombatTask(MyCombatTask):
    def __init__(self, config):
        super().__init__(config)
        self.reward_functions = [

        ]
        self.termination_conditions = [

        ]

    @property
    def num_agents(self) -> int:
        return 4

    def load_variables(self):
        self.state_var = [

        ]
        self.action_var = [

        ]

    def load_observation_space(self):
        self.obs_length = 9+7*(self.num_agents-1)
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=((self.num_agents + 2) * self.obs_length,))

    def load_action_space(self):
        self.action_space = spaces.MultiDiscrete([30, 41, 41, 41])

    def get_obs(self, env, agent_id):
        norm_obs = np.zeros(self.obs_length)
        # 自己的状态
        all_units = env.agents[agent_id].last_raw_obs
        if agent_id in env.ego_ids:
            ego_camp = 0
        else:
            ego_camp = 1

        ego_unit = next(u for u in all_units if u["camp_idx"] == ego_camp and u["own"])

        ego_pos = np.array(ego_unit["position"])
        ego_rot = np.array(ego_unit["rotation"])
        ego_vel = np.array(ego_unit["line_v"])
        ego_forward = np.array(ego_unit["forward"])
        ego_hp = ego_unit["hp"]

        norm_obs[0] = ego_pos[2] / 100  # 高度 单位：1km
        norm_obs[1] = np.sin(ego_rot[0])  # roll_sin
        norm_obs[2] = np.cos(ego_rot[0])  # roll_cos
        norm_obs[3] = np.sin(ego_rot[1])  # pitch_sin
        norm_obs[4] = np.cos(ego_rot[1])  # pitch_cos
        norm_obs[5:8] = ego_vel  # vx, vy, vz（单位化）
        norm_obs[8] = ego_hp  # ego_hp
        ego_feature = np.concatenate([ego_pos, ego_vel])
        # 敌人和队友的状态
        offset = 8
        for agent in env.agents[agent_id].partners + env.agents[agent_id].enemies:
            raw_obs = agent.last_raw_obs
            pos = np.array(raw_obs["position"])
            vel = np.array(raw_obs["line_v"])
            hp = raw_obs["hp"]
            feature = np.concatenate([pos, vel])
            AO, TA, R, side_flag = self.get2d_AO_TA_R_ue(ego_feature, feature, return_side=True)
            norm_obs[offset + 1] = hp
            norm_obs[offset + 2] = (vel[0] - ego_vel[0])  # delta_vx
            norm_obs[offset + 3] = (pos[2] - ego_pos[2]) / 100  # delta_height
            norm_obs[offset + 4] = AO
            norm_obs[offset + 5] = TA
            norm_obs[offset + 6] = R / 1000  # 距离单位化
            norm_obs[offset + 7] = side_flag
            offset += 7
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        norm_act = {
            "throttle": action[0] * 0.5 / (self.action_space.nvec[0] - 1.) + 0.4,
            "rudder": action[1] * 2. / (self.action_space.nvec[1] - 1.) - 1.,
            "elevator": action[2] * 2. / (self.action_space.nvec[2] - 1.) - 1.,
            "aileron": action[3] * 2. / (self.action_space.nvec[3] - 1.) - 1.,
            "flap": 0.0,
            "is_done": False
        }
        return norm_act

    def get_reward(self, env, agent_id, info: dict = ...) -> Tuple[float, dict]:
        return super().get_reward(env, agent_id, info=info)
