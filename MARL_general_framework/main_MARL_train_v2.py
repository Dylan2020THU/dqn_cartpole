# Born time: 2022-5-10
# Latest update: 2022-5-12
# RL Training Phase
# Hengxi
import itertools

import gym
from agent_marl import Agent_GU, Agent_UAV, ReplayMemory_UAV, ReplayMemory_GU
import numpy as np
import random
import torch
import torch.nn as nn

from env_SAGI import Env_SAGI

MIN_REPLAY_SIZE = 1000
BUFFER_SIZE = 500000

# env = gym.make("CartPole-v1")

n_agent_GU = 80
n_agent_UAV = 60
n_BS = 1
n_satellite = 1
n_power = 4
n_channel = 10

n_action_GU = (n_agent_UAV + n_BS + n_satellite) * n_power * n_channel  # one-hot coding
n_action_UAV = (n_agent_GU + n_BS + n_satellite) * n_power * n_channel

a_GUs = np.random.randint(low=0, high=4, size=(n_agent_GU, n_action_GU))
a_UAVs = np.random.randint(low=0, high=4, size=(n_agent_UAV, n_action_UAV))

env = Env_SAGI(n_agent_GU, n_agent_UAV, a_GUs, a_UAVs)

current_obs_GUs, current_obs_UAVs = env.get_state()
dim_obs_GUs = len(current_obs_GUs.T)
dim_obs_UAVs = len(current_obs_UAVs.T)

"""Generate GU and UAV agents"""
agent_GUs = []
for i in range(n_agent_GU):
    GU_i = Agent_GU(idx=i,
                    n_input=dim_obs_GUs,
                    n_output=n_action_GU,  # 3 types of outputs
                    n_UAV=n_agent_UAV,
                    mode='train')
    agent_GUs.append(GU_i)

agent_UAVs = []
for j in range(n_agent_UAV):
    UAV_j = Agent_UAV(idx=j,
                      n_input=dim_obs_UAVs,
                      n_output=n_action_UAV,
                      mode='train')
    agent_UAVs.append(UAV_j)

IS_TRAIN = 1
IS_TEST = 1 - IS_TRAIN

"""Main training loop"""
if IS_TRAIN:

    n_episode = 3000
    n_time_step = 1000

    EPSILON_START = 1.0
    EPSILON_END = 0.02
    EPSILON_DECAY = 10000

    # REWARD_BUFFER = np.empty(shape=n_episode)
    REWARD_BUFFER = []

    a_GUs = np.empty(shape=n_agent_GU)  # to initialize the action space of all
    a_UAVs = np.empty(shape=n_agent_UAV)
    # for episode_i in range(n_episode):
    for episode_i in itertools.count():
        """Action selection"""
        epsilon = np.interp(episode_i, [0, EPSILON_DECAY],
                            [EPSILON_START, EPSILON_END])  # epsilon annealing interpolation
        reward_per_episode = 0
        for time_step_i in range(n_time_step):
            random_sample = random.random()
            if random_sample <= epsilon:  # randomly choose actions
                # GUs
                for i in range(n_agent_GU):
                    a_GUs[i] = np.random.randint(low=0, high=n_action_GU)
                # UAVs
                for j in range(n_agent_UAV):
                    # a_q_UAV = np.random.random(size=n_action_UAV)
                    # a_softmax_UAV = nn.funcional.softmax(a_q_UAV, dim=0)  # obtain the maximum using softmax
                    # a_UAVs[i] = np.zeros(shape=n_action_UAV)  # make it 1
                    # a_UAVs[i][a_softmax_UAV.argmax().detach().item()] = 1   # one-hot action
                    a_UAVs[j] = np.random.randint(low=0, high=n_action_UAV)
            else:
                # GUs
                for i in range(n_agent_GU):  # choose actions using nn
                    a_GUs[i] = agent_GUs[i].online_net.act(current_obs_GUs[i])
                # UAVs
                for j in range(n_agent_UAV):
                    a_UAVs[j] = agent_UAVs[j].online_net.act(current_obs_UAVs[i])

            """Env transition"""
            next_obs_GUs, next_obs_UAVs, r = env.step(a_GUs, a_UAVs)  # same reward: cooperative game
            env.update()

            """Agents updating memories"""
            for i in range(n_agent_GU):
                agent_GUs[i].memo.add_memo(current_obs_GUs[i], a_GUs[i], r, next_obs_GUs[i])

            for j in range(n_agent_UAV):
                agent_UAVs[j].memo.add_memo(current_obs_UAVs[j], a_UAVs[j], r, next_obs_UAVs[j])

            current_obs_GUs = next_obs_GUs
            current_obs_UAVs = next_obs_UAVs
            reward_per_episode += r

        # REWARD_BUFFER[episode_i] = reward_per_episode
        REWARD_BUFFER.append(reward_per_episode)

        """Start Gradient Step"""
        """GU"""
        # GUs batch storage space
        batch_current_obs_GUs = torch.empty(size=(n_agent_GU, agent_GUs[0].BATCH_SIZE, dim_obs_GUs),
                                            dtype=torch.float32)
        batch_a_GUs = torch.randint(0, n_action_GU, size=(n_agent_GU, agent_GUs[0].BATCH_SIZE, 1), dtype=torch.int64)
        batch_r = torch.empty(size=(n_agent_GU, agent_GUs[0].BATCH_SIZE, 1), dtype=torch.float32)
        batch_next_obs_GUs = torch.empty(size=(n_agent_GU, agent_GUs[0].BATCH_SIZE, dim_obs_GUs), dtype=torch.float32)

        target_q_values_GUs = torch.empty(size=(n_agent_GU, agent_GUs[0].BATCH_SIZE, n_action_GU), dtype=torch.float32)
        max_target_q_values_GUs = torch.empty(size=(n_agent_GU, agent_GUs[0].BATCH_SIZE, 1),
                                              dtype=torch.float32)
        q_targets_GUs = torch.empty(size=(n_agent_GU, agent_GUs[0].BATCH_SIZE, 1), dtype=torch.float32)
        q_predicts_GUs = torch.empty(size=(n_agent_GU, agent_GUs[0].BATCH_SIZE, n_action_GU), dtype=torch.float32)
        a_q_values_GUs = torch.empty(size=(n_agent_GU, agent_GUs[0].BATCH_SIZE, 1), dtype=torch.float32)
        # loss_GUs = torch.empty(size=(n_agent_GU, 1), dtype=torch.float32)
        loss_GUs = []

        for i in range(n_agent_GU):

            """Experience replay"""
            batch_current_obs_GUs[i], batch_a_GUs[i], batch_r[i], batch_next_obs_GUs[i] = agent_GUs[
                i].memo.sample()  # update batch-size amounts of Q

            """Compute Targets"""
            target_q_values_GUs[i] = agent_GUs[i].target_net(batch_next_obs_GUs[i])
            # print(i)
            max_target_q_values_GUs[i] = target_q_values_GUs[i].max(dim=1, keepdim=True)[i][0]
            q_targets_GUs[i] = batch_r[i] + agent_GUs[i].GAMMA * max_target_q_values_GUs[i]

            """Compute Loss"""
            q_predicts_GUs[i] = agent_GUs[i].online_net(batch_current_obs_GUs[i])
            a_q_values_GUs[i] = torch.gather(input=q_predicts_GUs[i], dim=1, index=batch_a_GUs[i])  # ?
            loss_GUs.append(nn.functional.smooth_l1_loss(a_q_values_GUs[i], q_targets_GUs[i]))
            print(loss_GUs)

            """Gradient Descent"""
            agent_GUs[i].optimizer.zero_grad()
            # loss_GUs[i].backward(retain_graph=True)
            loss_GUs[i].backward()
            agent_GUs[i].optimizer.step()

            """Update target network"""
            if (episode_i * n_time_step) % agent_GUs[i].TARGET_UPDATE_FREQUENCY == 0:
                agent_GUs[i].target_net.load_state_dict(agent_GUs[i].online_net.state_dict())  # ?

        """UAV"""
        # UAVs batch storage space
        batch_current_obs_UAVs = torch.empty(size=(n_agent_UAV, agent_UAVs[0].BATCH_SIZE, dim_obs_UAVs),
                                             dtype=torch.float32)
        batch_a_UAVs = torch.randint(0, n_action_UAV, size=(n_agent_UAV, agent_UAVs[0].BATCH_SIZE, 1),
                                     dtype=torch.int64)
        batch_r = torch.empty(size=(n_agent_UAV, agent_UAVs[0].BATCH_SIZE, 1), dtype=torch.float32)
        batch_next_obs_UAVs = torch.empty(size=(n_agent_UAV, agent_UAVs[0].BATCH_SIZE, dim_obs_UAVs),
                                          dtype=torch.float32)

        target_q_values_UAVs = torch.empty(size=(n_agent_UAV, agent_UAVs[0].BATCH_SIZE, n_action_UAV),
                                           dtype=torch.float32)
        max_target_q_values_UAVs = torch.empty(size=(n_agent_UAV, agent_UAVs[0].BATCH_SIZE, 1),
                                               dtype=torch.float32)
        q_targets_UAVs = torch.empty(size=(n_agent_UAV, agent_UAVs[0].BATCH_SIZE, 1), dtype=torch.float32)
        q_predicts_UAVs = torch.empty(size=(n_agent_UAV, agent_UAVs[0].BATCH_SIZE, n_action_UAV), dtype=torch.float32)
        a_q_values_UAVs = torch.empty(size=(n_agent_UAV, agent_UAVs[0].BATCH_SIZE, 1), dtype=torch.float32)
        # loss_UAVs = torch.empty(size=(n_agent_UAV, 1), dtype=torch.float32)

        for i in range(n_agent_UAV):

            """Experience replay"""
            batch_current_obs_UAVs[i], batch_a_UAVs[i], batch_r[i], batch_next_obs_UAVs[i] = agent_UAVs[
                i].memo.sample()  # update batch-size amounts of Q

            """Compute Targets"""
            target_q_values_UAVs[i] = agent_UAVs[i].target_net(batch_next_obs_UAVs[i])
            max_target_q_values_UAVs[i] = target_q_values_UAVs[i].max(dim=1, keepdim=True)[i][0]
            q_targets_UAVs[i] = batch_r[i] + agent_UAVs[i].GAMMA * max_target_q_values_UAVs[i]

            """Compute Loss"""
            q_predicts_UAVs[i] = agent_UAVs[i].online_net(batch_current_obs_UAVs[i])
            a_q_values_UAVs[i] = torch.gather(input=q_predicts_UAVs[i], dim=1, index=batch_a_UAVs[i])  # ?
            loss_UAV = nn.functional.smooth_l1_loss(a_q_values_UAVs[i], q_targets_UAVs[i])

            """Gradient Descent"""
            agent_UAVs[i].optimizer.zero_grad()
            loss_UAV.backward()
            agent_UAVs[i].optimizer.step()

            """Update target network"""
            if (episode_i * n_time_step) % agent_UAVs[i].TARGET_UPDATE_FREQUENCY == 0:
                agent_UAVs[i].target_net.load_state_dict(agent_UAVs[i].online_net.state_dict())  # ?

        """Print the training progress"""
        print("Step: {}".format(episode_i))
        print("Avg reward: {}".format(np.mean(REWARD_BUFFER)))
