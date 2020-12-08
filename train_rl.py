import curses
import os
from curses import wrapper
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from timeit import default_timer as timer
from datetime import timedelta
import numpy as np
import random
import os

from MaTris.gym_matris import MatrisEnv
from MaTris.actions import ACTIONS
from networks.network_bodies import TetrisBodyV2
from agents.DQN import Model
from utils.hyperparameters import Config
from utils.ReplayMemory import ExperienceReplayMemory

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

def num_hidding_boxes(c):
    n, m = c.shape
    cnt = 0
    for j in range(m):
        hidden = False
        for i in range(n-1, -1, -1):
            if c[i, j] == 0:
                hidden = True
            if hidden and c[i, j] != 0:
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
    layer2score = [d ** 1.5 / 10.0 for d in range(n)]
    score = 0
    for i in range(n):
        for j in range(m):
            if c[i, j] != 0:
                score += layer2score[i]
    return score

def print_observation(ob, stdcsr=None):
    if ob is None:
        ob = np.zeros(shape=(3, 21, 10), dtype=np.int)
    n, m = ob[0].shape
    if stdcsr:
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)
    for i in range(n+2):
        for j in range(m+2):
            clr = 1
            if (i == 0 or i == n+1) and (j == 0 or j == m+1):
                pch = '+'
                if stdcsr:
                    if i == 0 and j == 0:
                        ch = curses.ACS_ULCORNER
                    elif i == 0 and j == m+1:
                        ch = curses.ACS_URCORNER
                    elif i == n+1 and j == 0:
                        ch = curses.ACS_LLCORNER
                    else:
                        ch = curses.ACS_LRCORNER
            elif i == 0 or i == n+1:
                pch = '-'
                if stdcsr:
                    ch = curses.ACS_HLINE
            elif j == 0 or j == m+1:
                pch = '|'
                if stdcsr:
                    ch = curses.ACS_VLINE
            else:
                if ob[0, i-1, j-1] != 0:
                    pch = '.'
                    if stdcsr:
                        ch = curses.ACS_BLOCK
                elif ob[1, i-1, j-1] != 0:
                    pch = '~'
                    if stdcsr:
                        ch = curses.ACS_BLOCK
                        clr = 2
                elif ob[2, i-1, j-1] != 0:
                    pch = 'x'
                    if stdcsr:
                        ch = curses.ACS_BLOCK
                        clr = 3
                else:
                    pch = ' '
                    ch = ' '
            if stdcsr:
                stdcsr.addch(ch, curses.color_pair(clr))
            else:
                print(pch, end="")
        if stdcsr:
            stdcsr.addstr('\n')
        else:
            print()
    if stdcsr:
        stdcsr.addstr(f'Base score: {board_base_score(ob[0]+ob[2]):6.1f} | Height: {boxes_height(ob[0] + ob[2]):2} | Hidden boxes: {num_hidden_boxes(ob[0]+ob[2]):2} | Closed boxes: {num_closed_boxes(ob[0]+ob[2]):2} | Shared edges: {num_shared_edges(ob[0], ob[2])}\n')
    else:
        print(f'Base score: {board_base_score(ob[0]+ob[2]):4.1f} | Height: {boxes_height(ob[0] + ob[2])} | Hidden boxes: {num_hidden_boxes(ob[0]+ob[2])} | Closed boxes: {num_closed_boxes(ob[0]+ob[2])} | Shared edges: {num_shared_edges(ob[0], ob[2])}')



def penalize_hidden_boxes(p, c):
    return -1 * (num_hidden_boxes(c[0] + c[2]) - num_hidden_boxes(p[0] + p[2]))

def penalize_hidding_boxes(p, c):
    return -1 * (num_hidden_boxes(c[0] + c[2]) - num_hidden_boxes(p[0] + p[2]))

def penalize_closed_boxes(p, c):
    return -2 * (num_closed_boxes(c[0] + c[2]) - num_closed_boxes(p[0] + p[2]))

def encourage_shared_edges(p, c):
    return 1 * (num_shared_edges(c[0], c[2]) - num_shared_edges(p[0], p[2]))

def penalize_higher_boxes(p, c):
    return -5 * (boxes_height(c[0] + c[2]) ** 1.5 - boxes_height(p[0] + p[2]) ** 1.5)

def encourage_lower_layers(p, c):
    return board_base_score(c[0] + c[2]) - board_base_score(p[0] + p[2])

reward_functions = [
    penalize_closed_boxes,
    # penalize_hidden_boxes,
    penalize_hidding_boxes,
    encourage_shared_edges,
    penalize_higher_boxes,
    encourage_lower_layers
]

def main(stdcsr=None):
    if stdcsr:
        stdcsr.clear()  # clear screen

    config = Config()
    config.USE_NOISY_NETS = True
    config.USE_PRIORITY_REPLAY = True
    env = MatrisEnv(no_display=True, real_tick=False, reward_functions=reward_functions)
    model = Model(env=env, config=config, body=TetrisBodyV2)

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

        model.append_to_replay(prev_observation, action, reward, observation)
        if frame_idx % config.TRAIN_FREQ == 0:
            model.update(prev_observation, action, reward, observation, frame_idx)
        episode_reward += reward
        lines = info['lines']

        if stdcsr:
            stdcsr.clear()
            print_observation(observation, stdcsr)
            stdcsr.addstr(f"T: {frame_idx:5} | Action: {ACTIONS[action][0]:11} | Reward: {reward:7.3f} | Episode reward {episode_reward:7.3f}| Lines: {lines} | Epsilon {epsilon:.3f}\n")
            stdcsr.refresh()
        else:
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


if __name__ == "__main__":
    use_text_gui = True
    if use_text_gui:
        wrapper(main)
    else:
        main()