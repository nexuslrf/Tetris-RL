import os
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

from MaTris.gym_matris import MatrisEnv
from networks.network_bodies import TetrisBodyV2
from utils.hyperparameters import Config
from utils.board_utils import print_observation
import utils.board_utils as bu
try:
    from curses import wrapper
except:
    print("your env does not support curses package")

render = True
test = False
config = Config()
device = config.device
config.BATCH_SIZE = 10
config.GAMMA = 0.99

class PGbaseline(nn.Module):
    def __init__(self, input_shape, body=TetrisBodyV2, num_actions=2):
        super(PGbaseline, self).__init__()
        self.net_body = body(input_shape)
        
        in_features = self.net_body.feature_size()
        self.action_head = nn.Linear(in_features, num_actions) # action 1: static, action 2: move up, action 3: move down
        self.value_head = nn.Linear(in_features, 1)

        self.num_actions = num_actions
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.net_body(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values


    def select_action(self, x):
        probs, state_value = self.forward(x)
        m = Categorical(probs)
        action = m.sample()

        self.saved_log_probs.append((m.log_prob(action), state_value))
        return action

class PGbaselineV2(nn.Module):
    def __init__(self, input_shape, body=TetrisBodyV2, num_actions=2):
        super(PGbaselineV2, self).__init__()
        self.net_body = body(input_shape)
        
        in_features = self.net_body.feature_size()
        self.rot_head = nn.Linear(in_features, 4) # action 1: static, action 2: move up, action 3: move down
        self.pos_head = nn.Linear(in_features, 10) 
        self.value_head = nn.Linear(in_features, 1)

        self.num_actions = num_actions
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.net_body(x))
        rot_scores = self.rot_head(x)
        pos_scores = self.pos_head(x)
        state_values = self.value_head(x)
        return F.softmax(rot_scores, dim=-1), F.softmax(pos_scores, dim=-1), state_values


    def select_action(self, x):
        rot_probs, pos_probs, state_value = self.forward(x)
        rot_m = Categorical(rot_probs)
        rot = rot_m.sample()
        pos_m = Categorical(pos_probs)
        pos = pos_m.sample()
        action = rot * 10 + pos

        self.saved_log_probs.append((rot_m.log_prob(rot) + pos_m.log_prob(pos), state_value))
        return action

def finish_episode(policy, optimizer):
    R = 0
    policy_loss = []
    value_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + config.GAMMA * R
        rewards.insert(0, R)
    # turn rewards to pytorch tensor and standardize
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
    rewards = rewards.to(device)
    for (log_prob, value), reward in zip(policy.saved_log_probs, rewards):
        advantage = reward - value
        policy_loss.append(- log_prob * advantage)         # policy gradient
        value_loss.append(F.smooth_l1_loss(value, reward)) # value function approximation
    optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum()
    value_loss = torch.stack(value_loss).sum()
    loss = policy_loss + value_loss
    loss = loss.to(device)
    loss.backward()
    optimizer.step()

    # clean rewards and saved_actions
    del policy.rewards[:]
    del policy.saved_log_probs[:]

reward_functions = [
    # bu.occlusion_penalty
]


def main(stdcsr=None):
    def log(s, end="\n"):
        if stdcsr:
            stdcsr.addstr(s + end)
        else:
            print(s, end=end)
    def refresh():
        if stdcsr:
            stdcsr.refresh()

    env = MatrisEnv(no_display=True, reward_functions=reward_functions)
    state_shape = env.observation_space.shape
    # print(state_shape)
    max_lines = 0
    # built policy network
    policy = PGbaseline(state_shape, num_actions=env.action_space.n).to(device)

    # check & load pretrain model
    if os.path.isfile('pgb_params.pkl'):
        print('Load PGbaseline Network parametets ...')
        policy.load_state_dict(torch.load('pgb_params.pkl'))

    # construct a optimal function
    optimizer = optim.Adam(policy.parameters(), lr=config.LR)

    # Main loop
    running_reward = None
    reward_sum = 0
    for i_episode in count(1):
        state = env.reset()
        prev_x = None
        for t in range(10000):
            if render: env.render()
            cur_x = state
            x = cur_x - prev_x if prev_x is not None else np.zeros(state_shape)
            prev_x = cur_x
            x = torch.tensor([x], device=device, dtype=torch.float)
            action = policy.select_action(x)
            state, reward, done, info = env.step(action)
            reward_sum += reward

            policy.rewards.append(reward)

            max_lines = max(max_lines, info['lines'])
            print_observation(state, stdcsr)
            log(f"Ep: {i_episode} | Reward: {reward:<5.1f} |  Reward_sum: {reward_sum::<5.1f}")
            log(f"Lines: {info['lines']:4d} | Max_Cleared_line: {max_lines:4d}")
            refresh()

            if done:
                # tracking log
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                log('Policy Gradient with Baseline ep %03d done. reward: %f. reward running mean: %f' % (i_episode, reward_sum, running_reward))
                reward_sum = 0
                break


        # use policy gradient update model weights
        if i_episode % config.BATCH_SIZE == 0 and test == False:
            finish_episode(policy, optimizer)

        # Save model in every 50 episode
        if i_episode % 50 == 0 and test == False:
            print('ep %d: model saving...' % (i_episode))
            torch.save(policy.state_dict(), 'pgb_params.pkl')

if __name__ == '__main__':
    use_text_gui = True
    if use_text_gui:
        wrapper(main)
    else:
        main()
