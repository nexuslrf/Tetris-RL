from timeit import default_timer as timer
from datetime import timedelta
import numpy as np
import random
import os

from MaTris.gym_matris import MatrisEnv
from agents.DQN import Model
from utils.hyperparameters import Config
from utils.ReplayMemory import ExperienceReplayMemory

if __name__ == "__main__":
    config = Config()
    config.USE_NOISY_NETS = True
    config.USE_PRIORITY_REPLAY = True
    env = MatrisEnv(no_display=True)
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

        print(f"T: {frame_idx} | Reward: {reward} | Lines: {lines}")
        # if lines < info['lines']:
        #     env.render()

        lines = info['lines']
        
        if done:
            observation = env.reset()
            model.save_reward(episode_reward)
            episode_reward = 0
            lines = 0
            
            # if np.mean(model.rewards[-10:]) > 19:
            #     # plot(frame_idx, model.rewards, model.losses, model.sigma_parameter_mag, timedelta(seconds=int(timer()-start)))
            #     break

        # if frame_idx % 10000 == 0:
        #     plot(frame_idx, model.rewards, model.losses, model.sigma_parameter_mag, timedelta(seconds=int(timer()-start)))
    if not os.path.exists('saved_agents/'):
        os.makedirs('saved_agents')
    model.save_w()
    env.close()

