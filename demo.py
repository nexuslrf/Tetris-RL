import os
import time
import json
from functools import partial
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import random
from torch import nn
from utils.hyperparameters import Config
from utils.board_utils import print_observation
from networks.network_bodies import TetrisBodyV2, TetrisBodyV3
from MaTris.gym_matris_v2 import MatrisEnv
from MaTris.gym_matris_v2 import ACTIONS
from value_iteration import ExperienceReplayMemory, ValueNetwork, Agent

use_text_gui = False
pygame_gui = True
try:
    from curses import wrapper
except:
    use_text_gui = False
    print("your env does not support curses package")

def main(stdcsr=None):
    def log(s, end="\n"):
        if stdcsr:
            stdcsr.addstr(s + end)
        else:
            print(s, end=end)
    def refresh():
        if stdcsr:
            stdcsr.refresh()
    env = MatrisEnv(no_display=not pygame_gui, real_tick=False)
    config = Config()
    model_path = "saved_agents/value_iteration_model.pth" # "saved_agents/V2/agent_11000/model.pth"
    body_list = [TetrisBodyV2]
    agent = Agent(env=env, config=config, body=body_list[0], use_target=False, static_policy=True, use_data_parallel=False)
    # load model 
    ckpt = torch.load(model_path)
    # agent.model.load_state_dict(ckpt)
    agent.model.load_state_dict({k.replace('module.',''):v for k,v in ckpt.items()}) 

    episode_reward = 0
    rounds = 0
    lines = 0
    epsilon = 0
    observation = env.reset()

    start_time = time.time()
    for frame_idx in range(1, config.MAX_FRAMES + 1):
        env.render()
        prev_observation=observation
        action = agent.get_action(epsilon)
        observation, reward, done, info = env.step(action)
        observation = None if done else observation

        episode_reward += reward
        rounds += 1
        lines = info['lines']
        score = info['score']
        if done:
            print("done")

        if not pygame_gui:
            print_observation(observation, stdcsr)
            current_value = agent.observation_value(prev_observation)
            next_value = agent.observation_value(observation)
            log("[{:5}/{} {:.0f} secs] State value: {:<5.1f}  Target value: {:<5.1f} ({:=5.1f} + {:=5.1f})  Action: {:<2}".format(
                frame_idx, config.MAX_FRAMES, time.time() - start_time, current_value, next_value+reward, reward, next_value, action))
            log("Game: {}  Round: {}  Episode reward: {:<5.1f}  Cleared lines: {:<4}  Loss: {:<.1f}  Epsilon: {:<.3f}".format(
                len(agent.lines), rounds, episode_reward, lines, agent.losses[-1][1] if len(agent.losses) > 0 else 0.0, epsilon))
            refresh()

        if done:
            observation = env.reset()
            assert observation is not None
            agent.append_episode_reward(frame_idx, episode_reward)
            agent.append_rounds(frame_idx, rounds)
            agent.append_lines(frame_idx, lines)
            game = len(agent.lines)
            episode_reward = 0
            rounds = 0
            lines = 0
    env.close()


if __name__ == '__main__':
    
    if use_text_gui:
        wrapper(main)
    else:
        main()


