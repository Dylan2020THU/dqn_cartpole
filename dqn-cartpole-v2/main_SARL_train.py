# Born time: 2022-5-10
# Latest update: 2022-5-11
# RL Training Phase
# Hengxi
import itertools

import gym
from agent import Agent
import numpy as np
import random
import torch
import torch.nn as nn

MIN_REPLAY_SIZE = 1000
BUFFER_SIZE = 500000

env = gym.make("CartPole-v1")

num_class_1_agent = 10

s = env.reset()

# Generate agents
dim_state = len(s)
n_action = env.action_space.n

"""Generate agents"""
# agents_class1 = []
# for i in range(num_class_1_agent):
#     agent = Agent(idx=i,
#                   n_input=dim_state,
#                   n_output=n_action,
#                   mode='train')
#     # agent = Agent(idx=i,
#     #               n_input=len(obs)+len(obs)+len(env.action_space),
#     #               n_output=len(env.action_space),
#     #               mode='train')
#     agents_class1.append(agent)

agent = Agent(idx=0,
              n_input=dim_state,
              n_output=n_action,
              mode='train')

# Main Training Loop

n_episode = 100000
n_time_step = 100

EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000

s = env.reset()
episode_reward = 0
REWARD_BUFFER = []
# for episode_i in range(n_episode):
for episode_i in itertools.count():
    epsilon = np.interp(episode_i, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])  # interpolation

    random_sample = random.random()
    if random_sample <= epsilon:
        a = env.action_space.sample()
    else:
        a = agent.online_net.act(s)

    s_, r, done, info = env.step(a)  # info will not be used
    agent.memo.add_memo(s, a, r, done, s_)
    s = s_
    episode_reward += r

    if done:
        s = env.reset()
        REWARD_BUFFER.append(episode_reward)

    # After solved, watch it play
    if len(REWARD_BUFFER) >= 100:
        if np.mean(REWARD_BUFFER) >= 200000:
            while True:
                a = agent.online_net.act(s)
                s, r, done, info = env.step(a)
                env.render()

                if done:
                    env.reset()

    # Start Gradient Step
    batch_s, batch_a, batch_r, batch_done, batch_s_ = agent.memo.sample()  # update batch-size amounts of Q

    # Compute Targets
    target_q_values = agent.target_net(batch_s_)
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]  # ?

    targets = batch_r + agent.GAMMA * (1 - batch_done) * max_target_q_values

    # Compute Loss
    q_values = agent.online_net(batch_s)

    a_q_values = torch.gather(input=q_values, dim=1, index=batch_a)  # ?

    loss = nn.functional.smooth_l1_loss(a_q_values, targets)

    # Gradient Descent
    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()

    # Update target network
    # if agent.memo.count % agent.TARGET_UPDATE_FREQUENCY == 0:
    if episode_i % agent.TARGET_UPDATE_FREQUENCY == 0:
        agent.target_net.load_state_dict(agent.online_net.state_dict())  # ?

    # Print the training progress
    #     print("Step: {}".format(agent.memo.count))
        print("Step: {}".format(episode_i))
        print("Avg reward: {}".format(np.mean(REWARD_BUFFER)))
