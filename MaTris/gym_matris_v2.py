import os
import sys
import gym
from gym import error, spaces, utils
from gym.utils import seeding

from multiprocessing import Pool

this_dir = os.path.dirname(__file__)
# Add lib to PYTHONPATH
lib_path = os.path.join(this_dir)
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

from matris import *
import numpy as np
import itertools
import copy

"""
Redesign the action, every step the tetrimino will drop to the bottom
the action will choose the rotation and horizontal position
rotation: [0,1,2,3] | position: [-4,-3,-2,-1,0,+1,+2,+3,+4,+5]
after drop rotation and translation (optional)
rotation: [0,1,2,3] | position: [-3,-2,-1,0,+1,+2,+3]
"""
ROTATION = [0,1,2,3] # clock-wise
POSITION = [-4,-3,-2,-1,0,+1,+2,+3,+4,+5]
POST_MOV = [-1,0,1]
ACTIONS = [combo for combo in itertools.product(ROTATION, POSITION, POST_MOV)]
def generate_action_seq(act):
    rot, pos, p_pos = act
    act_list = ["forward"] * rot if rot < 3 else ["reverse"]
    act_list = act_list + ['left'] * -pos if pos < 0 else act_list + ['right'] * pos
    act_list.append('bottom drop')
    act_list = act_list + ['left'] * -p_pos if p_pos < 0 else act_list + ['right'] * p_pos
    act_list.append('hard drop')
    return act_list

class MatrisEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    action_list = ACTIONS

    def __init__(self, no_display=True, real_tick=False, reward_functions=None, mp_pool=0):
        if not no_display:
            pygame.init()
            pygame.display.set_caption("MaTris")

            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        else:
            self.screen = None
        self.game = Game()
        self.game.gym_init(self.screen)
        self.reward_functions = reward_functions

        self.real_tick = real_tick
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,22,10), dtype=np.int)
        self.mp_pool = mp_pool

    def step(self, action_id):
        self.game.matris.drop_bonus = False
        timepassed = self.game.clock.tick(50) if self.real_tick else 20
        previous_state = self.game.matris.get_state()
        act = generate_action_seq(self.action_list[action_id])
        reward = self.game.matris.step_update(act, timepassed/1000)
        done = self.game.matris.done
        state = self.game.matris.get_state()
        info = self.game.matris.get_info()

        previous_state[1:] = 0
        state[1:] = 0
        if self.reward_functions is not None:
            for reward_function in self.reward_functions:
                reward += reward_function(previous_state, state)

        reward /= 10

        return state, reward, done, info

    def peak_step_srdi(self, action_id):
        timepassed = self.game.clock.tick(50) if self.real_tick else 20
        previous_state = self.game.matris.get_state()
        self.game.matris.push_state()
        act = generate_action_seq(self.action_list[action_id])
        reward = self.game.matris.step_update(act, timepassed/1000, set_next=False)
        done = self.game.matris.done
        state = self.game.matris.get_state()
        info = self.game.matris.get_info()
        self.game.matris.pop_state()

        previous_state[1:] = 0
        state[1:] = 0
        if self.reward_functions:
            for reward_function in self.reward_functions:
                reward += reward_function(previous_state, state)

        reward /= 10
        return state, reward, done, info


    def peek(args):
        mat, action_id, reward_functions = args
        reward_functions = None
        act = generate_action_seq(MatrisEnv.action_list[action_id])
        reward = mat.step_update(act, 0.02, set_next=False)
        done = mat.done
        state = mat.get_state()
        info = mat.get_info()
        state[1:] = 0
        if reward_functions:
            for reward_function in reward_functions:
                reward += reward_function(previous_state, state)

        reward /= 10
        return state, reward, done, info
        # return action_id 

    def mp_peek_step(self):
        assert self.mp_pool is not None
        previous_state = self.game.matris.get_state()
        previous_state[1:] = 0
        self.game.matris.push_state()
        
        ret_lst = self.mp_pool.map(MatrisEnv.peek, 
            [(copy.copy(self.game.matris), i, self.reward_functions) for i in range(self.action_space.n)])

        self.game.matris.pop_state()

        return ret_lst
        


    def reset(self):
        self.game.gym_init(self.screen)
        return self.game.matris.get_state()

    def render(self, mode='human', close=False):
        try:
            pygame.event.get()
        except:
            pass
        if close:
            self.game.matris.gameover()
            return
        if self.screen and self.game.matris.needs_redraw:
            self.game.redraw()

    def close(self):
        self.game.matris.gameover(True)

if __name__ == "__main__":
    mp_pool = Pool(4)
    env = MatrisEnv(no_display=True, mp_pool=mp_pool)
    for i in range(1000):
        state, reward, done, info = env.step(env.action_space.sample())
        env.mp_peek_step()
        print(f"Reward: {reward}")
        if not done:
            env.render()
        else:
            env.reset()
            
    print("Game over!")