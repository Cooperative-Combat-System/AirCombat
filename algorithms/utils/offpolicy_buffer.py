import numpy as np
import torch
from typing import Dict, Any

class OffPolicyReplayBuffer:
    """
    Simple transition replay buffer for TD3/SAC.
    Stores (obs, act, rew, next_obs, done).
    """
    def __init__(self, obs_shape, act_shape, capacity: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device

        self.obs = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.acts = np.zeros((self.capacity, *act_shape), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.rewards[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        assert self.size > 0
        idx = np.random.randint(0, self.size, size=batch_size)

        batch = dict(
            obs=torch.as_tensor(self.obs[idx], device=self.device),
            acts=torch.as_tensor(self.acts[idx], device=self.device),
            rewards=torch.as_tensor(self.rewards[idx], device=self.device),
            next_obs=torch.as_tensor(self.next_obs[idx], device=self.device),
            dones=torch.as_tensor(self.dones[idx], device=self.device),
        )
        return batch

    def __len__(self):
        return self.size
