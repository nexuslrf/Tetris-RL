import os
import time
import json
from functools import partial
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch import optim
import numpy as np
from curses import wrapper
import random
from torch import nn
from utils.hyperparameters import Config
from utils.board_utils import penalize_closed_regions, print_observation
from networks.network_bodies import TetrisBodyV2, TetrisBodyV3
from MaTris.gym_matris_v2 import MatrisEnv
from MaTris.gym_matris_v2 import ACTIONS
from utils.board_utils import penalize_closed_boxes, penalize_hidden_boxes, penalize_hidding_boxes, penalize_higher_boxes, encourage_lower_layers, encourage_shared_edges, encourage_boxex_in_a_line
from utils.board_utils import board_height_score, hidden_boxes_score, hidding_boxes_score, closed_boxes_score, shared_edges_score, boxes_in_a_line_score, board_box_height_score
from utils.board_utils import penalize_ave_height, penalize_quadratic_uneveness

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
        self.body = body(input_shape, num_actions=None)
        in_features = self.body.feature_size()
        if in_features == 1:
            self.fc = nn.Sequential()
        else:
            self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.fc(self.body(x))


class TetrisHeirsticBody(nn.Module):
    score_functions = [
        board_height_score, 
        hidden_boxes_score, 
        hidding_boxes_score, 
        closed_boxes_score, 
        # shared_edges_score, 
        boxes_in_a_line_score, 
        # board_box_height_score
    ]
    def __init__(self, input_shape):
        super(TetrisHeirsticBody, self).__init__()
        self.out_features = len(self.score_functions)
        self.fc = nn.Linear(in_features=self.out_features, out_features=1)

    def preprocess(self, x):
        device = x.device
        bs = x.shape[0]
        x = x.cpu().numpy().astype(np.int)
        # assert isinstance(x, np.ndarray)
        # x.astype(np.int)
        scores = np.zeros(shape=(bs, self.out_features), dtype=np.int)
        for i in range(bs):
            for j in range(self.out_features):
                scores[i, j] = self.score_functions[j](x[i])
        return torch.tensor(scores, dtype=torch.float, device=device)

    def forward(self, x):
        # x = self.preprocess(x)
        # x = self.fc(x)
        # x = torch.sum(x, dim=1, keepdim=True)
        return torch.zeros((x.size(0), 1), dtype=torch.float, device=x.device)
    
    def feature_size(self):
        return self.fc.out_features


class Agent(object):
    def __init__(self, env=None, config=None, body=TetrisBodyV2, static_policy=False, use_target=True, use_data_parallel=True):
        self.env: MatrisEnv = env
        self.device = config.device
        self.observation_shape = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.use_data_parallel = use_data_parallel
        self.memory = ExperienceReplayMemory(capacity=config.EXP_REPLAY_SIZE)
        self.model = ValueNetwork(self.observation_shape, body=body).to(self.device)
        if use_data_parallel:
            self.model = nn.DataParallel(self.model)
        self.static_policy = static_policy
        self.episode_rewards = []
        self.losses = []
        self.rounds = []
        self.lines = []

        self.batch_size = config.BATCH_SIZE
        self.lr = config.LR
        self.gamma = config.GAMMA
        self.use_target = use_target

        if use_target:
            self.target_model = ValueNetwork(self.observation_shape, body=body).to(self.device)
            self.target_model.load_state_dict(self.model.state_dict())
            if use_data_parallel:
                self.target_model = nn.DataParallel(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # if self.static_policy:
        #     self.model.eval()
        #     self.target_model.eval()
        # else:
        #     self.model.train()
        #     self.target_model.eval()
    
    def get_action(self, epsilon):
        target_model = self.target_model if self.use_target else self.model
        with torch.no_grad():
            self.model.eval()
            target_model.eval()
            # assert np.allclose(observation, self.env.game.matris.get_state())
            if np.random.random() >= epsilon or self.static_policy:
                # next_states = []
                # rewards = []
                # for action in range(self.num_actions):
                #     next_state, reward, done, _ = self.env.peak_step_srdi(action)
                #     if not done:
                #         next_states.append(torch.tensor([next_state], device=self.device, dtype=torch.float))
                #         rewards.append(torch.tensor([reward], device=self.device, dtype=torch.float))
                #     else:
                #         rewards.append(torch.tensor([reward], device=self.device, dtype=torch.float))
                #         next_states.append(torch.zeros_like(torch.tensor([observation]), device=self.device, dtype=torch.float))
                # next_states = torch.cat(next_states, dim=0)
                # rewards = torch.cat(rewards, dim=0).unsqueeze(1)
                # expected_values = rewards + target_model(next_states)
                batch_next_state, batch_reward, batch_done, batch_info = zip(*self.env.peak_actions())
                expected_values = torch.tensor(batch_reward, dtype=torch.float, device=self.device)
                if not all(batch_done):
                    valid_next_state = [state for done, state in zip(batch_done, batch_next_state) if not done]
                    batch_done = torch.tensor(batch_done, dtype=torch.bool)
                    batch_next_state = torch.tensor(valid_next_state, dtype=torch.float, device=self.device)
                    expected_values[~batch_done] += target_model(batch_next_state).squeeze()
                action = torch.argmax(expected_values, dim=0).squeeze().item()
                return action
            else:
                return np.random.randint(0, self.num_actions)
    
    def append_to_replay(self, prev_observation, action, reward, observation):
        self.memory.push((prev_observation, action, reward, observation))

    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1.0 - cond)

    def update(self, step=0):
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
            target_model = self.target_model if self.use_target else self.model
            target_state_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                target_state_values[non_final_mask] = target_model(non_final_next_states)
            expected_v_values = batch_reward + (self.gamma * target_state_values)
        
        diff = (expected_v_values - current_v_values)
        loss = self.huber(diff).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.model.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.append_loss(step, loss.item())
    
    def update_target_network(self):
        if self.use_target:
            self.target_model.load_state_dict(self.model.state_dict())

    def append_episode_reward(self, step, reward):
        self.episode_rewards.append((step, reward))
    
    def observation_value(self, observation):
        if observation is None:
            return 0.0
        with torch.no_grad():
            self.model.eval()
            return self.model(torch.tensor([observation], dtype=torch.float, device=self.device)).squeeze().item()
        
    def append_loss(self, step, loss):
        self.losses.append((step, loss))
    
    def append_rounds(self, step, rounds):
        self.rounds.append((step, rounds))
    
    def append_lines(self, step, lines):
        self.lines.append((step, lines))

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(dirname, "model.pth"))
        if self.use_target:
            torch.save(self.target_model.state_dict(), os.path.join(dirname, "target_model.pth"))
        torch.save(self.optimizer.state_dict(), os.path.join(dirname, "optimizer.pth"))
        for name, data in zip(['losses', 'rounds', 'lines', 'episode_rewards'], [self.losses, self.rounds, self.lines, self.episode_rewards]):
            with open(os.path.join(dirname, f'{name}.json'), 'w') as f:
                json.dump(data, f)
            with open(os.path.join(dirname, f'{name}.txt'), 'w') as f:
                f.write("\n".join(" ".join(str(v) for v in l) for l in data))

# reward_functions = []
reward_functions = [
    # penalize_closed_boxes,
    penalize_hidden_boxes,
    penalize_hidding_boxes,
    penalize_closed_regions,
    # encourage_shared_edges,
    penalize_higher_boxes,
    encourage_lower_layers,
    encourage_boxex_in_a_line,
    penalize_ave_height,
    penalize_quadratic_uneveness
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
    env = MatrisEnv(no_display=True, real_tick=False, reward_functions=reward_functions)
    config = Config()
    # config.MAX_FRAMES = 20
    config.EXP_REPLAY_SIZE = 50000
    config.BATCH_SIZE = 2048
    config.LEARN_START = 2048
    config.TRAIN_FREQ = 1
    config.TARGET_NET_UPDATE_FREQ = 100
    config.SAVE_FREQ = 100
    config.LR = 2e-3
    config.epsilon_start = 0.0
    config.epsilon_final = 0.0
    # config.MAX_FRAMES = 200
    # config.BATCH_SIZE = 1
    body_list = [TetrisBodyV2, TetrisHeirsticBody]
    agent = Agent(env=env, config=config, body=body_list[0], use_target=False)

    episode_reward = 0
    rounds = 0
    lines = 0
    observation = env.reset()
    start_time = time.time()
    for frame_idx in range(1, config.MAX_FRAMES + 1):
        env.render()
        epsilon = config.epsilon_by_frame(frame_idx)

        action = agent.get_action(epsilon)
        prev_observation=observation
        observation, reward, done, info = env.step(action)
        observation = None if done else observation

        agent.append_to_replay(prev_observation, action, reward, observation)
        if frame_idx >= config.LEARN_START and frame_idx % config.TRAIN_FREQ == 0:
            agent.update(frame_idx)

        if frame_idx >= config.LEARN_START and frame_idx % config.TARGET_NET_UPDATE_FREQ == 0:
            agent.update_target_network()

        episode_reward += reward
        rounds += 1
        lines = info['lines']
        if done:
            print("done")

        print_observation(observation, stdcsr)
        current_value = agent.observation_value(prev_observation)
        next_value = agent.observation_value(observation)
        log("[{:5}/{} {:.0f} secs] State value: {:<5.1f}  Target value: {:<5.1f} ({:=5.1f} + {:=5.1f})  Action: {:<2}".format(
            frame_idx, config.MAX_FRAMES, time.time() - start_time, current_value, next_value+reward, reward, next_value, action))
        log("Rounds: {:<4}  Episode reward: {:<5.1f}  Cleared lines: {:<4}  Loss: {:<.1f}  Epsilon: {:<.3f}".format(
            rounds, episode_reward, lines, agent.losses[-1][1] if len(agent.losses) > 0 else 0.0, epsilon))
        refresh()

        if done:
            observation = env.reset()
            assert observation is not None
            agent.append_episode_reward(frame_idx, episode_reward)
            agent.append_rounds(frame_idx, rounds)
            agent.append_lines(frame_idx, lines)
            episode_reward = 0
            rounds = 0
            lines = 0
        
        if frame_idx % config.SAVE_FREQ == 0:
            agent.save(f'./saved_agents/agent_{frame_idx}')
            
    agent.save('./saved_agents/final')
    env.close()


if __name__ == '__main__':
    use_text_gui = True
    if use_text_gui:
        wrapper(main)
    else:
        main()

