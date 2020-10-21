import numpy as np
import pandas as pd
import time

N_states = 12 #the length
ACTIONS = ['left', 'right', 'freeze']

alpha = 0.1 #learning rate
Lambda = 0.9 #discount factor
max_episodes = 100 #numbers of episodes
fresh_time = 0.01
max_epsilon = 1.0
min_epsilon = 0.08
decay = 0.9
epsilon = max_epsilon

#type(states) = int
#type(action) = list
#初始化qtable,行数为states，列数为len(action)的零矩阵
def build_a_qtable():
    q_table = pd.DataFrame(np.zeros((N_states, len(ACTIONS))),
                           columns = ACTIONS
                           )
    return q_table

#使用epsilon贪心算法进行动作选择，取随机数
def epsilon_greedy(qtable, state, epsilon):

    a_state = qtable.iloc[state, :]
    if np.random.rand() > epsilon:
        #采用最大奖励动作
        action_name = a_state.idxmax()
    else:
        #随机采取动作
        action_name = np.random.choice(ACTIONS)
    return action_name

#动作对环境的反馈
def env_feedback(action_name, state):
    if action_name == "left":
        R = -100
        if state == 0:
            next_state = 0
        else:
            next_state = state -1
    if action_name == "freeze":
        R = 0
        if state == N_states - 1:
            next_state = 'terminal'
        else:
            next_state = state
    if action_name == "right":
        R = 100
        if state == N_states - 2:
            next_state = 'terminal'
        else:
            next_state = state + 1
    return next_state, R


#创建游戏环境 "_____T"
def env(episode, step_counter, state):
    env_list = (N_states-1)*"_" + "T"
    if state == 'terminal':
        interaction = 'Episode %s: total_steps = %s' %(episode+1, step_counter)
        print('\r{}'.format(interaction), end ='')
        time.sleep(4)
        print('\r                                 ', end='')
    else:
        env_list = list(env_list)
        env_list[state] = 'O'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        #time.sleep(fresh_time)

#rl部分，主要为Qtable的更新
def rl():
    global epsilon
    q_table = build_a_qtable()
    for episode in range(max_episodes):
        step_counter = 0
        state = 0
        is_terminal = False
        env(episode, step_counter, state)
        while not is_terminal:
            epsilon = max(epsilon * decay, min_epsilon)
            action = epsilon_greedy(q_table, state, epsilon) #根据qtable的值选取当前state的动作
            #print(action)
            next_state, R = env_feedback(action, state) #根据当前state做出的动作预测出下一步的状态和得到的汇报
            q_predict = q_table.loc[state, action]
            if next_state != 'terminal':
                q_target = R + Lambda*q_table.values[next_state, :].max()
            else:
                q_target = R
                is_terminal = True
            q_table.loc[state, action] += alpha * (q_target - q_predict)
            state = next_state
            env(episode, step_counter+1, state)
            step_counter += 1
            #print(q_table)
    return q_table



if __name__ =="__main__":
    q_table = rl()
    print(q_table)
