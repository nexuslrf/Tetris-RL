from MaTris.gym_matris_v2 import MatrisEnv
import numpy as np
import math
import utils.board_utils as bu

def get_simple_state(c):
    return np.array([
            bu.num_hidden_boxes(c),
            bu.num_hidding_boxes(c),
            bu.num_closed_boxes(c),
            bu.num_closed_regions(c),
            bu.num_shared_edges(c),
            bu.board_height(c),
            bu.board_box_height(c),
            bu.board_ave_height(c),
            bu.board_quadratic_uneveness(c),
            bu.board_line_score(c),
            bu.hidding_boxes_score(c),
            bu.closed_boxes_score(c),
            bu.closed_regions_score(c),
            bu.shared_edges_score(c),
            bu.board_height_score(c),
            bu.boxes_in_a_line_score(c),
            bu.board_box_height_score(c),
            bu.board_ave_height_score(c),
            bu.board_quadratic_uneveness_score(c),
    ])

def max_q_action(env, w):
    policy_act = 0
    max_val = -np.inf
    for i in range(env.action_space.n):
        tmp_state, _, _, _ = env.peak_step_srdi(i)
        tmp_simple_state = get_simple_state(tmp_state)
        val = (tmp_simple_state * w).sum()
        if val > max_val:
            max_val = val
            policy_act = i
            next_simple_state = tmp_simple_state
    return policy_act, next_simple_state

env = MatrisEnv(no_display=False)
state = env.reset()
alpha = 0.01
gamma = 0.95
curr_state = np.zeros(env.observation_space.shape)
next_state = np.zeros(env.observation_space.shape)
curr_simple_state = get_simple_state(curr_state)
next_simple_state = get_simple_state(next_state)

weight = np.random.randn(*curr_simple_state.shape) * 10 + 5

epsilon = 0.8
epsilon_final = 0.05
epsilon_decay = 0.01

lines = 0
ref_height = 0

for f in range(10000):
    if epsilon > epsilon_final:
        epsilon -= epsilon_decay
    env.render()
    policy_act = 0
    rand_sel = np.random.rand() < epsilon
    if rand_sel:
        policy_act = env.action_space.sample()
        next_state, _, _, _ = env.peak_step_srdi(policy_act)
        next_simple_state = get_simple_state(next_state)
    else:
        policy_act, next_simple_state = max_q_action(env, weight)

    next_state, reward, done, info = env.step(policy_act)
    reward *= 10
    if done:
        reward = -2000

    curr_simple_state = next_simple_state

    # weight update
    _, max_next_simple_state = max_q_action(env, weight)
    target = reward + gamma * (max_next_simple_state*weight).sum()
    diff =  target - (curr_simple_state*weight).sum()
    weight = weight + alpha * diff * curr_simple_state

    print(f"T: {f} | Reward: {reward} |")

    if done:
        env.reset()
        lines = 0
        ref_height = 0
        
print("Game over!")