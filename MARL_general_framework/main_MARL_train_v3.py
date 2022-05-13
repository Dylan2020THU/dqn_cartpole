# Born time: 2022-5-10
# Latest update: 2022-5-12
# RL Training Phase
# Hengxi
import itertools

import gym
from agent_marl_v3 import Agent_GU, Agent_UAV
import numpy as np
import random
import torch
import torch.nn as nn

from env_SAGI import Env_SAGI

MIN_REPLAY_SIZE = 1000
BUFFER_SIZE = 500000
GRADIENT_DESCENT = 50
TARGET_UPDATE_FREQUENCY = 100

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

    a_GUs = np.empty(shape=n_agent_GU)  # to initialize the action space of all
    a_UAVs = np.empty(shape=n_agent_UAV)
    # for episode_i in range(n_episode):
    LOSS_BUFFER_GUs = []
    LOSS_BUFFER_UAVs = []
    REWARD_BUFFER = []

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
                    a_UAVs[j] = agent_UAVs[j].online_net.act(current_obs_UAVs[j])

            """Env transition"""
            next_obs_GUs, next_obs_UAVs, r = env.step(a_GUs, a_UAVs)  # same reward: cooperative game
            current_obs_GUs = next_obs_GUs
            current_obs_UAVs = next_obs_UAVs
            reward_per_episode += r

            env.update()

            """Start learning"""
            """GU"""
            """Add memories"""
            for i in range(n_agent_GU):
                agent_GUs[i].memo.add_memo(current_obs_GUs[i], a_GUs[i], r, next_obs_GUs[i])

                """Start Gradient Step"""
                if time_step_i % GRADIENT_DESCENT == 0:
                    loss_batch_GU = agent_GUs[i].q_learning_batch()
                    LOSS_BUFFER_GUs.append(loss_batch_GU)
                    # if i == 0:
                    #     print("Episode:", episode_i, 'Time step:', time_step_i, 'GU:', i, 'Loss:', loss_batch_GU)
                    #     pass

                """Update target network"""
                if time_step_i % TARGET_UPDATE_FREQUENCY == 0:
                    agent_GUs[i].update_target_q_network()
                    # if i == 0:
                    #     print("GU------Target Q network is updated")
                    #     pass

            """UAV"""
            """Add memories"""
            for j in range(n_agent_UAV):
                agent_UAVs[j].memo.add_memo(current_obs_UAVs[j], a_UAVs[j], r, next_obs_UAVs[j])

                """Start Gradient Step"""
                if time_step_i % GRADIENT_DESCENT == 0:
                    loss_batch_UAV = agent_UAVs[j].q_learning_batch()
                    LOSS_BUFFER_UAVs.append(loss_batch_UAV)
                    # if j == 0:
                    #     print("Episode:", episode_i, 'Time step:', time_step_i, 'UAV:', j, 'Loss:', loss_batch_UAV)
                    #     pass

                """Update target network"""
                if time_step_i % TARGET_UPDATE_FREQUENCY == 0:
                    agent_UAVs[j].update_target_q_network()
                    # if j == 0:
                    #     print("UAV------Target Q network is updated")
                    #     pass

        REWARD_BUFFER.append(reward_per_episode)
        """Print the training progress"""
        # print("Avg reward: {}".format(np.mean(REWARD_BUFFER)))
        print("Episode:", episode_i, "Reward:", reward_per_episode)
