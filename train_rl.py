import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from timeit import default_timer as timer
from datetime import timedelta
import numpy as np
import random
import os

from MaTris.gym_matris import MatrisEnv
from MaTris.actions import ACTIONS
from agents.DQN import Model
from utils.hyperparameters import Config
from utils.ReplayMemory import ExperienceReplayMemory

def print_board(c):
    n, m = c.shape
    for i in range(n+2):
        for j in range(m+2):
            if (i == 0 or i == n+1) and (j == 0 or j == m+1):
                ch = '+'
            elif i == 0 or i == n+1:
                ch = '-'
            elif j == 0 or j == m+1:
                ch = '|'
            else:
                ch = '* '[c[i-1, j-1] == 0]
            print(ch, end="")
        print()

def penalize_hidden_boxes(p, c):
    def num_hidden_boxes(c):
        n, m = c.shape
        cnt = 0
        for j in range(m):
            hidden = False
            for i in range(n):
                if c[i, j] != 0:
                    hidden = True
                if hidden and c[i, j] == 0:
                    cnt += 1
        # print_board(c)
        # print(f"#Hidden: {cnt}")
        return cnt
    return -1 * (num_hidden_boxes(c[0]) - num_hidden_boxes(p[0]))

def penalize_closed_boxes(p, c):
    def num_closed_boxes(c):
        n, m = c.shape
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        visited = set()
        queue = [(0, j) for j in range(m) if c[0, j] == 0]
        while len(queue) > 0:
            u = queue.pop()
            for d in directions:
                v = (u[0] + d[0], u[1] + d[1])
                if v[0] < 0 or v[0] >= n or v[1] < 0 or v[1] >= m:
                    continue
                if c[v] == 0 and v not in visited:
                    visited.add(v)
                    queue.append(v)
        closed = n * m - c.sum() - len(visited)
        # print_board(c)
        # print(f"#Closed: {closed}")
        return closed
    return -2 * (num_closed_boxes(c[0]) - num_closed_boxes(p[0]))

def encourage_shared_edges(p, c):
    def num_shared_edges(c, h):
        n, m = c.shape
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        cnt = 0
        for i in range(n):
            for j in range(m):
                if h[i, j] != 0:
                    for d in directions:
                        ii, jj = i + d[0], j + d[1]
                        if ii < 0 or ii >= n or jj < 0 or jj >= m:
                            continue
                        if c[ii, jj] != 0:
                            cnt += 1
        # print_board(h)
        # print_board(c)
        # print(f"#Shared edges: {cnt}")
        return cnt
    return 1 * (num_shared_edges(c[0], c[2]) - num_shared_edges(p[0], p[2]))


def penalize_higher_boxes(p, c):
    def boxes_height(c):
        n, m = c.shape
        # print_board(c)
        for i in range(n):
            for j in range(m):
                if c[i, j] != 0:
                    # print(f"#Height: {n-i}")
                    return n - i
        # print(f"#Height: {0}")
        return 0
    return -5 * (boxes_height(c[0]) - boxes_height(p[0]))

reward_functions = [
    penalize_closed_boxes,
    penalize_hidden_boxes,
    encourage_shared_edges,
    penalize_higher_boxes
]

if __name__ == "__main__":
    config = Config()
    config.USE_NOISY_NETS = True
    config.USE_PRIORITY_REPLAY = True
    env = MatrisEnv(no_display=False, real_tick=False, reward_functions=reward_functions)
    model = Model(env=env, config=config)

    episode_reward = 0
    lines = 0
    observation = env.reset()
    for frame_idx in range(1, config.MAX_FRAMES + 1):
        # if frame_idx % 1000 == 0:
        env.render()
        epsilon = config.epsilon_by_frame(frame_idx)

        action = model.get_action(observation, epsilon)
        prev_observation=observation
        observation, reward, done, info = env.step(action)
        observation = None if done else observation

        model.update(prev_observation, action, reward, observation, frame_idx)
        episode_reward += reward
        lines = info['lines']

        print(f"T: {frame_idx:5} | Action: {ACTIONS[action][0]:11} | Reward: {reward:7.3f} | Episode reward {episode_reward:7.3f}| Lines: {lines}")

        if done:
            observation = env.reset()
            model.save_reward(episode_reward)
            episode_reward = 0
            lines = 0
            
    if not os.path.exists('saved_agents/'):
        os.makedirs('saved_agents')
    model.save_w()
    env.close()

