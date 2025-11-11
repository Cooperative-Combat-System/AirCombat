import torch
import numpy as np
from gym import spaces
from .task_base import BaseTask
from ..termination_conditions import MyTermination
from ..reward_functions import MyRewardFunction
from ..utils.adaptor import NetworkAdaptor

class MyCombatTask(BaseTask):
    def __init__(self, config):
        super().__init__(config)
        self.reward_functions = [MyRewardFunction(self.config)]
        self.termination_conditions = [MyTermination(self.config)]
        self.adaptor = NetworkAdaptor(self.config)

    @property
    def num_agents(self) -> int:
        return 2

    def load_variables(self):
        self.state_var = [

        ]
        self.action_var = [

        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(16,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.Box(
            low=np.array([0.4, -1, -1, -1], dtype=np.float32),
            high=np.array([1.0, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

    def get_obs(self, env, agent_id):
        all_units = env.agents[agent_id].last_raw_obs
        if not all_units or not isinstance(all_units, list):
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        if agent_id in env.ego_ids:
            ego_camp = 0
        else:
            ego_camp = 1

        try:
            ego_unit = next(u for u in all_units if u.get("camp_idx") == ego_camp and u.get("own"))
            enm_unit = next(u for u in all_units if u.get("camp_idx") != ego_camp)
        except StopIteration:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        ego_pos = np.array(ego_unit["position"])
        ego_rot = np.array(ego_unit["rotation"])
        ego_vel = np.array(ego_unit["line_v"])
        ego_forward = np.array(ego_unit["forward"])
        ego_hp = ego_unit["hp"]

        enm_pos = np.array(enm_unit["position"])
        enm_rot = np.array(enm_unit["rotation"])
        enm_vel = np.array(enm_unit["line_v"])
        enm_forward = np.array(enm_unit["forward"])
        enm_hp = enm_unit["hp"]

        norm_obs = np.zeros(16)
        norm_obs[0] = enm_pos[2] / 100 #高度 单位：1km
        norm_obs[1] = np.sin(ego_rot[0])  # roll_sin
        norm_obs[2] = np.cos(ego_rot[0])  # roll_cos
        norm_obs[3] = np.sin(ego_rot[1])  # pitch_sin
        norm_obs[4] = np.cos(ego_rot[1])  # pitch_cos
        norm_obs[5:8] = ego_vel  # vx, vy, vz（单位化）
        norm_obs[8] = ego_hp  # ego_hp
        norm_obs[9] = enm_hp  # enm_hp

        ego_feature = np.concatenate([ego_pos, ego_vel])
        enm_feature = np.concatenate([enm_pos, enm_vel])
        AO, TA, R, side_flag = self.get2d_AO_TA_R_ue(ego_feature, enm_feature, return_side=True)
        norm_obs[10] = (enm_vel[0] - ego_vel[0])  # delta_vx
        norm_obs[11] = (enm_pos[2] - ego_pos[2]) / 100  # delta_height
        norm_obs[12] = AO
        norm_obs[13] = TA
        norm_obs[14] = R / 1000  # 距离单位化
        norm_obs[15] = side_flag

        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        norm_act = {
            "throttle": action[0],
            "rudder": action[1],
            "elevator": action[2],
            "aileron": action[3],
            "flap": 0.0,
            "is_done": False
        }
        return norm_act

    def reset(self, env):
        # self.adaptor.reconnect()
        # for agent in env.agents():
        #     agent.initialize(env.config.initial_states[f"camp_{agent.camp}"][0])
        return super().reset(env)


    def step(self, env):
        """
        这是task独有的step函数，这里应该不需要多做处理
        """
        pass

    def get_reward(self, env, agent_id, info=...):
        return super().get_reward(env, agent_id, info=info)

    def get2d_AO_TA_R_ue(self, ego_feature, enm_feature, return_side=False):
        ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
        ego_v = np.linalg.norm([ego_vx, ego_vy])
        enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = enm_feature
        enm_v = np.linalg.norm([enm_vx, enm_vy])
        delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
        R = np.linalg.norm([delta_x, delta_y]) # 避免除0

        proj_dist = delta_x * ego_vx + delta_y * ego_vy
        ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))

        proj_dist = delta_x * enm_vx + delta_y * enm_vy
        ego_TA = np.arccos(np.clip(proj_dist / (R * enm_v + 1e-8), -1, 1))

        if not return_side:
            return ego_AO, ego_TA, R
        else:
            side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
            return ego_AO, ego_TA, R, side_flag

