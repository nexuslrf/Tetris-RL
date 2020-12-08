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
    return cnt

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
    return closed

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
                        if ii != -1:
                            cnt += 1  # shared with boarder
                        continue
                    if c[ii, jj] != 0:
                        cnt += 1
    return cnt

def boxes_height(c):
    n, m = c.shape
    for i in range(n):
        for j in range(m):
            if c[i, j] != 0:
                return n - i
    return 0

def board_base_score(c):
    n, m = c.shape
    layer2score = [d / 10.0 for d in range(n)]
    score = 0
    for i in range(n):
        for j in range(m):
            if c[i, j] != 0:
                score += layer2score[i]
    return score

def print_observation(ob):
    if ob is None:
        print()
        print(' --- GAME OVER ---')
        print()
        return
    n, m = ob[0].shape
    for i in range(n+2):
        for j in range(m+2):
            if (i == 0 or i == n+1) and (j == 0 or j == m+1):
                ch = '+'
            elif i == 0 or i == n+1:
                ch = '-'
            elif j == 0 or j == m+1:
                ch = '|'
            else:
                if ob[0, i-1, j-1] != 0:
                    ch = '.'
                elif ob[1, i-1, j-1] != 0:
                    ch = '~'
                elif ob[2, i-1, j-1] != 0:
                    ch = 'x'
                else:
                    ch = ' '
            print(ch, end="")
        print()
    print(f'Base score: {board_base_score(ob[0]+ob[1]):4.1f} | Height: {boxes_height(ob[0] + ob[1])} | Hidden boxes: {num_hidden_boxes(ob[0]+ob[1])} | Closed boxes: {num_closed_boxes(ob[0]+ob[1])} | Shared edges: {num_shared_edges(ob[0], ob[1])}')


def penalize_hidden_boxes(p, c):
    return -1 * (num_hidden_boxes(c[0] + c[1]) - num_hidden_boxes(p[0] + p[1]))

def penalize_closed_boxes(p, c):
    return -2 * (num_closed_boxes(c[0] + c[1]) - num_closed_boxes(p[0] + p[1]))

def encourage_shared_edges(p, c):
    return 1 * (num_shared_edges(c[0], c[1]) - num_shared_edges(p[0], p[1]))

def penalize_higher_boxes(p, c):
    return -5 * (boxes_height(c[0] + c[1]) - boxes_height(p[0] + p[1]))

def encourage_lower_layers(p, c):
    return board_base_score(c[0] + c[1]) - board_base_score(p[0] + p[1])

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
    env = MatrisEnv(no_display=True, real_tick=False, reward_functions=reward_functions)
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

        print_observation(observation)
        print(f"T: {frame_idx:5} | Action: {ACTIONS[action][0]:11} | Reward: {reward:7.3f} | Episode reward {episode_reward:7.3f}| Lines: {lines} | Epsilon {epsilon:.3f}")

        if done:
            observation = env.reset()
            model.save_reward(episode_reward)
            episode_reward = 0
            lines = 0
            
    if not os.path.exists('saved_agents/'):
        os.makedirs('saved_agents')
    model.save_w()
    env.close()

