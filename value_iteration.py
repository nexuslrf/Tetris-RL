import os
import time
import json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch import optim
import numpy as np
from curses import wrapper
import random
from torch import nn
from utils.hyperparameters import Config
from utils.board_utils import print_observation
from MaTris.actions import ACTIONS
from networks.network_bodies import TetrisBodyV2
from MaTris.gym_matris import MatrisEnv
from utils.board_utils import penalize_closed_boxes, penalize_hidden_boxes, penalize_hidding_boxes, penalize_higher_boxes, encourage_lower_layers, encourage_shared_edges, encourage_boxex_in_a_line



class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0

    def push(self, transition):
        if self.index == len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.index] = transition
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class ValueNetwork(nn.Module):
    def __init__(self, input_shape, body):
        super(ValueNetwork, self).__init__()
        self.body = body(input_shape)
        self.fc = nn.Linear(self.body.feature_size(), 1)

    def forward(self, x):
        return self.fc(self.body(x))


class Agent(object):
    def __init__(self, env=None, config=None, body=TetrisBodyV2, static_policy=False):
        self.env = env
        self.device = config.device
        self.observation_shape = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.memory = ExperienceReplayMemory(capacity=config.EXP_REPLAY_SIZE)
        self.model = ValueNetwork(self.observation_shape, body=body)
        self.target_model = ValueNetwork(self.observation_shape, body=body)
        self.static_policy = static_policy
        self.episode_rewards = []
        self.losses = []

        self.batch_size = config.BATCH_SIZE
        self.lr = config.LR
        self.gamma = config.GAMMA


        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        if self.static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.eval()
    
    def get_action(self, observation, epsilon):
        with torch.no_grad():
            self.model.eval()
            # assert np.allclose(observation, self.env.game.matris.get_state())
            if np.random.random() >= epsilon or self.static_policy:
                action_value_pairs = []
                for action in range(self.num_actions):
                    next_state, reward, done, _ = self.env.peak_step(action)
                    if not done:
                        next_state = torch.tensor([next_state], device=self.device, dtype=torch.float)
                        action_value_pairs.append((action, reward + float(self.model(next_state).item())))
                if len(action_value_pairs) > 0:
                    return max(action_value_pairs, key=lambda x:x[1])[0]
                else:
                    return random.choice(range(self.num_actions))
            else:
                return np.random.randint(0, self.num_actions)

    def append_to_replay(self, prev_observation, action, reward, observation):
        self.memory.push((prev_observation, action, reward, observation))

    def append_episode_reward(self, reward):
        self.episode_rewards.append(reward)

    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1.0 - cond)

    def update(self):
        self.model.train()
        batch_transaction = self.memory.sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state = zip(*batch_transaction)
        input_shape = (-1,) + self.observation_shape
        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(input_shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.bool)
        lst = [s for s in batch_next_state if s is not None]
        if len(lst) > 0:
            non_final_next_states = torch.tensor(lst, device=self.device, dtype=torch.float).view(input_shape)
            empty_next_state_values = False
        else:
            non_final_next_states = None
            empty_next_state_values = True
    
        current_v_values = self.model(batch_state)

        with torch.no_grad():
            target_state_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                target_state_values[non_final_mask] = self.target_model(non_final_next_states)
            expected_v_values = batch_reward + (self.gamma * target_state_values)
        
        diff = (expected_v_values - current_v_values)
        loss = self.huber(diff).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.append_loss(loss.item())
    
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def append_loss(self, loss):
        self.losses.append(loss)

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(dirname, "model.pth"))
        torch.save(self.optimizer.state_dict(), os.path.join(dirname, "optimizer.pth"))
        with open(os.path.join(dirname, 'losses.json'), 'w') as f:
            json.dump(self.losses, f)
        with open(os.path.join(dirname, 'episode_rewards.json'), 'w') as f:
            json.dump(self.episode_rewards, f)

# reward_functions = []
reward_functions = [
    penalize_closed_boxes,
    penalize_hidden_boxes,
    penalize_hidding_boxes,
    encourage_shared_edges,
    penalize_higher_boxes,
    encourage_lower_layers,
    encourage_boxex_in_a_line
]

def main(stdcsr=None):
    def log(s, end="\n"):
        if stdcsr:
            stdcsr.addstr(s + end)
            stdcsr.refresh()
        else:
            print(s, end=end)
    env = MatrisEnv(no_display=True, real_tick=False, reward_functions=reward_functions)
    config = Config()
    # config.LEARN_START = 130
    # config.TRAIN_FREQ = 25
    # config.TARGET_NET_UPDATE_FREQ = 50
    # config.MAX_FRAMES = 200
    # config.BATCH_SIZE = 1
    agent = Agent(env=env, config=config, body=TetrisBodyV2)

    episode_reward = 0
    lines = 0
    observation = env.reset()
    for frame_idx in range(1, config.MAX_FRAMES + 1):
        env.render()
        epsilon = config.epsilon_by_frame(frame_idx)

        action = agent.get_action(observation, epsilon)
        prev_observation=observation
        observation, reward, done, info = env.step(action)
        observation = None if done else observation

        agent.append_to_replay(prev_observation, action, reward, observation)
        if frame_idx >= config.LEARN_START and frame_idx % config.TRAIN_FREQ == 0:
            agent.update()
        
        if frame_idx >= config.LEARN_START and frame_idx % config.TARGET_NET_UPDATE_FREQ == 0:
            agent.update_target_network()

        episode_reward += reward
        lines = info['lines']
        if done:
            print("done")

        print_observation(observation, stdcsr)
        log(f"T: {frame_idx:5} | Action: {ACTIONS[action][0]:11} | Reward: {reward:7.3f} | Episode reward {episode_reward:7.3f}| Lines: {lines} | Epsilon {epsilon:.3f}", end='\n')
        log(f'Losses: {agent.losses[-1] if len(agent.losses) > 0 else 0.0:5.2f}')

        if done:
            observation = env.reset()
            assert observation is not None
            agent.append_episode_reward(episode_reward)
            episode_reward = 0
            lines = 0
            
    agent.save('./saved_agent')
    env.close()


if __name__ == '__main__':
    use_text_gui = True
    if use_text_gui:
        wrapper(main)
    else:
        main()

