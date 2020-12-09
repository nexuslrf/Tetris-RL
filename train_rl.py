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
from utils.board_utils import num_closed_boxes, num_hidden_boxes, num_shared_edges, num_hidding_boxes, board_base_score, board_height, print_observation
from utils.board_utils import penalize_closed_boxes, penalize_hidden_boxes, penalize_hidding_boxes, penalize_higher_boxes, encourage_lower_layers, encourage_shared_edges


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