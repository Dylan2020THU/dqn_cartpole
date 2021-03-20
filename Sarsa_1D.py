# One-dimensional Sarsa-based Agent Searching for the Precious
# 2021-03-20
# Information Building 1111, TBSI, Shenzhen

import numpy as np
import pandas as pd
import time

np.random.seed()

N_states = 6  # the length of the path
# state_termial = N_states-1
state_termial = 'terminal'
ACTIONS = ['Left', 'Right']
epsilon = 0.9  # the probability to venture is 0.1
alpha = 0.1  # the learning rate
discount = 0.9  # the discount rate
max_episodes = 10  # the maximum episode
update_time = 0.1


def build_sarsa_table(n_states, actions):
    sarsa_table = pd.DataFrame(np.zeros((n_states + 1, len(actions))), columns=actions)
    sarsa_table.loc['terminal'] = [0,0]
    return sarsa_table


def choose_action(state, sarsa_table):
    state_actions = sarsa_table.loc[state, :]
    if (np.random.uniform() > epsilon) or (state_actions.all() == 0):
        action = np.random.choice(ACTIONS)
    else:
        action = state_actions.idxmax()
    return action


def feedback_from_env(s, a):
    if a == 'Right':
        if s == N_states - 2:
            s_ = state_termial
            R = 1
        else:
            s_ = s + 1
            R = 0
    else:
        R = 0
        if s == 0:
            s_ = s
        else:
            s_ = s - 1
    return s_, R


def update_env(s, episode, step_counter):
    env_list = ['-'] * (N_states - 1) + ['T']
    if s == state_termial:
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('{}'.format(interaction))
        time.sleep(2)
    else:
        env_list[s] = 'o'
        interaction = ''.join(env_list)
        print('{}'.format(interaction))
        time.sleep(update_time)


def sarsa_learning():
    sarsa_table = build_sarsa_table(N_states, ACTIONS)
    for episode in range(max_episodes):
        s = 0
        step_counter = 0
        is_terminated = False
        update_env(s, episode, step_counter)
        while not is_terminated:
            a = choose_action(s, sarsa_table)
            s_, R = feedback_from_env(s, a)
            sarsa_predict = sarsa_table.loc[s, a]
            if s_ != state_termial:
                sarsa_real = R + discount * sarsa_table.loc[s_, a]  # difference with Q-learning: .loc is used for char
            else:
                sarsa_real = R
                is_terminated == True

            sarsa_table.loc[s, a] += alpha * (sarsa_real - sarsa_predict)
            # print(sarsa_table)
            s = s_
            update_env(s, episode, step_counter + 1)
            step_counter += 1
            if s == state_termial:
                break
    return sarsa_table


if __name__ == "__main__":
    sarsa_table = sarsa_learning()
    print('\n')
    print(sarsa_table)
