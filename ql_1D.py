# One-dimensional Agent Searching for Precious
# 2020-12-25
# Information Building 1111, TBSI, Shenzhen

import numpy as np
import pandas as pd
import time

np.random.seed()

N_states = 6
# state_termial = N_states-1
state_termial = 'terminal'
ACTIONS = ['Left', 'Right']
epsilon = 0.9
alpha = 0.1
discount = 0.9
max_episodes = 15
update_time = 0.1


def build_q_table(n_states, actions):
    q_table = pd.DataFrame(np.zeros((n_states + 1, len(actions))), columns=actions)
    q_table.loc['terminal'] = [0,0]
    return q_table


def choose_action(state, q_table):
    state_actions = q_table.loc[state, :]
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


def q_learning():
    q_table = build_q_table(N_states, ACTIONS)
    for episode in range(max_episodes):
        s = 0
        step_counter = 0
        is_terminated = False
        update_env(s, episode, step_counter)
        while not is_terminated:
            a = choose_action(s, q_table)
            s_, R = feedback_from_env(s, a)
            q_predict = q_table.loc[s, a]
            if s_ != state_termial:
                q_real = R + discount * q_table.iloc[s_, :].max()
            else:
                q_real = R
                is_terminated == True

            q_table.loc[s, a] += alpha * (q_real - q_predict)
            # print(q_table)
            s = s_
            update_env(s, episode, step_counter + 1)
            step_counter += 1
            if s == state_termial:
                break
    return q_table


if __name__ == "__main__":
    q_table = q_learning()
    print('\n')
    print(q_table)
    # q_table = build_q_table(N_states, ACTIONS)
