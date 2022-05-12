# Born time: 2022-5-12
# Latest update: 2022-5-12
# RL Training Phase
# Hengxi
import numpy as np


class Env_SAGI:
    def __init__(self, n_GU, n_UAV, a_k1, a_k2):
        self.n_GU = n_GU
        self.n_UAV = n_UAV
        self.a_k1 = a_k1
        self.a_k2 = a_k2

    def get_state(self):
        self.current_obs_GUs = np.random.random(size=(self.n_GU, 4))
        self.current_obs_UAVs = np.random.random(size=(self.n_UAV, 4))

        return self.current_obs_GUs, self.current_obs_UAVs

    def step(self, action_GUs, action_UAVs):
        self.action_GUs = action_GUs
        self.action_UAVs = action_UAVs
        next_obs_k1 = np.random.random(size=(self.n_GU, 4))
        next_obs_k2 = np.random.random(size=(self.n_UAV, 4))
        r = 3

        return next_obs_k1, next_obs_k2, r

    def update(self):
        pass
