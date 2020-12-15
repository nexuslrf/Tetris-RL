import os
import sys
import gym
from gym import error, spaces, utils
from gym.utils import seeding

from multiprocessing import Pool, cpu_count

this_dir = os.path.dirname(__file__)
# Add lib to PYTHONPATH
lib_path = os.path.join(this_dir)
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

from MaTris.matris import *
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
# POST_MOV = [-1,0,1]
ACTIONS = [combo for combo in itertools.product(ROTATION, POSITION)]
def generate_action_seq(act):
    rot, pos = act
    act_list = ["forward"] * rot if rot < 3 else ["reverse"]
    act_list = act_list + ['left'] * -pos if pos < 0 else act_list + ['right'] * pos
    # act_list.append('bottom drop')
    # act_list = act_list + ['left'] * -p_pos if p_pos < 0 else act_list + ['right'] * p_pos
    act_list.append('hard drop')
    return act_list

class MatrisEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    action_list = ACTIONS
    # pools = Pool(cpu_count())
    def step_func_wrapper(args):
        matris, action_id, reward_functions = args
        return MatrisEnv.step_func(matris, action_id, 20, reward_functions)

    def step_func(matris: MatrisCore, action_id, timepassed, reward_functions=None):
        previous_state = matris.get_state()
        act = generate_action_seq(ACTIONS[action_id])
        reward = matris.step_update(act, timepassed/1000, set_next=True)
        done = matris.done
        state = matris.get_state()
        info = matris.get_info()

        previous_state[1:] = 0
        state[1:] = 0
        if reward_functions:
            for reward_function in reward_functions:
                reward += reward_function(previous_state, state)

        reward /= 10
        return state, reward, done, info

    def __init__(self, no_display=True, real_tick=False, reward_functions=None, mp_pool=None):
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
        return MatrisEnv.step_func(self.game.matris, action_id, timepassed, self.reward_functions)

    def peak_step(self, action_id):
        timepassed = self.game.clock.tick(50) if self.real_tick else 20
        self.game.matris.push_state()
        ret = MatrisEnv.step_func(self.game.matris, action_id, timepassed, self.reward_functions)
        self.game.matris.pop_state()
        return ret
    
    def peak_actions(self, action_id_list=None):
        if action_id_list is None:
            action_id_list = list(range(self.action_space.n))
        
        if self.mp_pool is not None:
            self.game.matris.push_state()
            ret = self.mp_pool.map(MatrisEnv.step_func_wrapper, \
                [(copy.deepcopy(self.game.matris), action_id, self.reward_functions) \
                    for action_id in action_id_list])
            self.game.matris.pop_state()
        else:
            ret = []
            for action_id in action_id_list:
                self.game.matris.push_state()
                tmp = MatrisEnv.step_func(self.game.matris, action_id, 20, self.reward_functions)
                ret.append(tmp)
                self.game.matris.pop_state()
        return ret
        
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