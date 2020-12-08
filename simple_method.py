from MaTris.gym_matris_v2 import MatrisEnv
import numpy as np
import math


def occlusion_penalty(p, c):
    def foo(board):
        if board.sum() > 0:
            occlusion_height = np.argmax(board, axis=0)
            occlusion_height[occlusion_height == 0] = board.shape[0] # 22
            occluded_height = board.shape[0] - np.argmin(np.flipud(board), axis=0)
            occlusion_score = (occluded_height - occlusion_height).sum()
            return occlusion_score
        else:
            return 0


    occlusion_delta = foo(c[0]) - foo(p[0])
    if occlusion_delta > 0:
        occlusion_delta = occlusion_delta * 20 + 300

    return -1 * occlusion_delta

reward_functions = [
    occlusion_penalty
]


env = MatrisEnv(no_display=False, reward_functions=reward_functions)
state = env.reset()
alpha = 0.01
gamma = 0.95
curr_state = np.zeros([4])
next_state = np.zeros([4])
weight = np.array([-5, -5, -5, -30])

epsilon = 0.8
epsilon_final = 0.05
epsilon_decay = 0.01

lines = 0
ref_height = 0

for f in range(10000):
    if epsilon > epsilon_final:
        epsilon -= epsilon_decay
    env.render()
    max_val = -np.inf
    greedy_act = 0
    rand_sel = np.random.rand() < epsilon
    if rand_sel:
        greedy_act = env.action_space.sample()
        next_state = np.array(env.peek_step(greedy_act))
    else:
        for i in range(env.action_space.n):
            tmp = np.array(env.peek_step(i))
            val = (tmp * weight).sum()
            if val > max_val:
                max_val = val
                greedy_act = i
                next_state = tmp

    state, reward, done, info = env.step(greedy_act)

    line_inc = info['lines'] - lines
    lines = info['lines']
    one_step_reward = 10 * line_inc ** 2 #- (next_state[0] - ref_height)
    ref_height = next_state[0]
    # weight update
    weight = weight + alpha * weight * (one_step_reward - curr_state + gamma * next_state)


    print(f"T: {f} | Reward: {reward} | Weight {weight}")
    curr_state = next_state

    if done:
        env.reset()
        lines = 0
        ref_height = 0
        
print("Game over!")