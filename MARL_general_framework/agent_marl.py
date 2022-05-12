# Born time: 2022-5-10
# Latest update: 2022-5-11
# RL agent
# Hengxi

import numpy as np
import torch
import torch.nn as nn
import random

random.seed(2022)

class ReplayMemory_GU:
    def __init__(self, dim_s, dim_a):
        self.dim_s = dim_s
        self.dim_a = dim_a

        self.MEMORY_SIZE = 1000
        self.BATCH_SIZE = 64
        self.all_s = np.empty(shape=(self.MEMORY_SIZE, self.dim_s), dtype=np.float64)
        self.all_a = np.empty(shape=(self.MEMORY_SIZE, self.dim_a), dtype=np.uint8)
        self.all_r = np.empty(self.MEMORY_SIZE, dtype=np.float64)
        self.all_s_ = np.empty(shape=(self.MEMORY_SIZE, self.dim_s), dtype=np.float64)
        self.count = 0
        self.t = 0

    def add_memo(self, s, a, r, s_):
        self.all_s[self.t] = s
        self.all_a[self.t] = a
        self.all_r[self.t] = r
        self.all_s_[self.t] = s_
        self.count = max(self.count, self.t + 1)
        self.t = (self.t + 1) % self.MEMORY_SIZE

    def sample(self):
        if self.count < self.BATCH_SIZE:
            indexes = range(0, self.count)
        else:
            indexes = random.sample(range(0, self.count), self.BATCH_SIZE)

        batch_s = []
        batch_a = []
        batch_r = []
        batch_s_ = []
        for idx in indexes:
            batch_s.append(self.all_s[idx])
            batch_a.append(self.all_a[idx])
            batch_r.append(self.all_r[idx])
            batch_s_.append(self.all_s_[idx])

        batch_s_tensor = torch.as_tensor(np.asarray(batch_s), dtype=torch.float32)
        # batch_a_tensor = torch.as_tensor(np.asarray(batch_a), dtype=torch.int64).unsqueeze(-1)
        batch_a_tensor = torch.as_tensor(np.asarray(batch_a), dtype=torch.int64)
        batch_r_tensor = torch.as_tensor(np.asarray(batch_r), dtype=torch.float32).unsqueeze(-1)
        batch_s__tensor = torch.as_tensor(np.asarray(batch_s_), dtype=torch.float32)

        return batch_s_tensor, batch_a_tensor, batch_r_tensor, batch_s__tensor


class ReplayMemory_UAV:
    def __init__(self, dim_s, dim_a):
        self.dim_s = dim_s
        self.dim_a = dim_a

        self.MEMORY_SIZE = 1000
        self.BATCH_SIZE = 64
        self.all_s = np.empty(shape=(self.MEMORY_SIZE, self.dim_s), dtype=np.float64)
        self.all_a = np.empty(shape=(self.MEMORY_SIZE, self.dim_a), dtype=np.uint8)
        self.all_r = np.empty(self.MEMORY_SIZE, dtype=np.float64)
        self.all_s_ = np.empty(shape=(self.MEMORY_SIZE, self.dim_s), dtype=np.float64)
        self.count = 0
        self.t = 0

    def add_memo(self, s, a, r, s_):
        self.all_s[self.t] = s
        self.all_a[self.t] = a
        self.all_r[self.t] = r
        self.all_s_[self.t] = s_
        self.count = max(self.count, self.t + 1)
        self.t = (self.t + 1) % self.MEMORY_SIZE

    def sample(self):
        if self.count < self.BATCH_SIZE:
            indexes = range(0, self.count)
        else:
            indexes = random.sample(range(0, self.count), self.BATCH_SIZE)

        batch_s = []
        batch_a = []
        batch_r = []
        batch_s_ = []
        for idx in indexes:
            batch_s.append(self.all_s[idx])
            batch_a.append(self.all_a[idx])
            batch_r.append(self.all_r[idx])
            batch_s_.append(self.all_s_[idx])

        batch_s_tensor = torch.as_tensor(np.asarray(batch_s), dtype=torch.float32)
        batch_a_tensor = torch.as_tensor(np.asarray(batch_a), dtype=torch.int64)
        batch_r_tensor = torch.as_tensor(np.asarray(batch_r), dtype=torch.float32).unsqueeze(-1)
        batch_s__tensor = torch.as_tensor(np.asarray(batch_s_), dtype=torch.float32)

        return batch_s_tensor, batch_a_tensor, batch_r_tensor, batch_s__tensor


class DQN_GU(nn.Module):
    def __init__(self, dim_input, dim_output):
        super().__init__()
        in_features = dim_input  # ?

        # nn.Sequential() ?
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, dim_output))

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)  # ?
        q_values = self(obs_tensor.unsqueeze(0))  # ?
        max_q_index = torch.argmax(q_values, dim=1)[0]  # ?
        action = max_q_index.detach().item()  # ?
        return action


class DQN_UAV(nn.Module):
    def __init__(self, dim_input, dim_output):
        super().__init__()
        in_features = dim_input  # ?

        # nn.Sequential() ?
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, dim_output))

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)  # ?
        q_values = self(obs_tensor.unsqueeze(0))  # ?
        max_q_index = torch.argmax(q_values, dim=1)[0]  # ?
        action = max_q_index.detach().item()  # ?
        return action


class Agent_GU:
    def __init__(self, idx, n_input, n_output, n_UAV, mode="train"):
        self.idx = idx
        self.mode = mode
        self.n_input = n_input
        self.n_output = n_output

        self.GAMMA = 0.99
        self.learning_rate = 1e-3
        # self.BUFFER_SIZE = 50000
        self.MIN_REPLAY_SIZE = 1000
        self.TARGET_UPDATE_FREQUENCY = 500
        self.BATCH_SIZE = 64

        self.memo = ReplayMemory_GU(dim_s=self.n_input, dim_a=self.n_output)

        self.n_UAV = n_UAV
        self.n_BS = 1
        self.n_satellite = 1
        self.n_power = 4
        self.n_channel = 10

        # action space of GU
        self.a1_space = np.arange(self.n_UAV + self.n_BS + self.n_satellite)
        self.a2_space = np.arange(self.n_power)  # 4 powers
        self.a3_space = np.arange(self.n_channel)  # 10 channels
        self.action_space = [self.a1_space, self.a2_space, self.a3_space]

        # Initialize the replay buffer of agent i
        if self.mode == "train":
            self.online_net = DQN_GU(self.n_input, self.n_output)
            self.target_net = DQN_GU(self.n_input, self.n_output)

            self.target_net.load_state_dict(self.online_net.state_dict())  # copy the current state of online_net

            self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate)


class Agent_UAV:
    def __init__(self, idx, n_input, n_output, mode="train"):
        self.idx = idx
        self.mode = mode
        self.n_input = n_input
        self.n_output = n_output

        self.GAMMA = 0.99
        self.learning_rate = 1e-3
        # self.BUFFER_SIZE = 50000
        self.MIN_REPLAY_SIZE = 1000
        self.TARGET_UPDATE_FREQUENCY = 1000
        self.BATCH_SIZE = 64

        self.memo = ReplayMemory_UAV(dim_s=self.n_input, dim_a=self.n_output)

        self.n_BS = 1
        self.n_satellite = 1
        self.n_power = 4
        self.n_channel = 10

        # action space of UAV
        self.a1_space = np.arange(self.n_BS + self.n_satellite)
        self.a2_space = np.arange(self.n_power)  # 4 powers
        self.a3_space = np.arange(self.n_channel)  # 10 channels
        self.action_space = [self.a1_space, self.a2_space, self.a3_space]

        # Initialize the replay buffer of agent i
        if self.mode == "train":
            self.online_net = DQN_UAV(self.n_input, self.n_output)
            self.target_net = DQN_UAV(self.n_input, self.n_output)

            self.target_net.load_state_dict(self.online_net.state_dict())  # copy the current state of online_net

            self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate)


if __name__ == "__main__":
    a = Agent_UAV
    print(len(a.action_space))
