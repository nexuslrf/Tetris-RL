import os
import sys
import gym
from gym import error, spaces, utils
from gym.utils import seeding

this_dir = os.path.dirname(__file__)
# Add lib to PYTHONPATH
lib_path = os.path.join(this_dir)
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

from matris import *
import numpy as np
import itertools

"""
Redesign the action, every step the tetrimino will drop to the bottom
the action will choose the rotation and horizontal position
rotation: [0,1,2,3] | position: [-4,-3,-2,-1,0,+1,+2,+3,+4,+5]
after drop rotation and translation (optional)
rotation: [0,1,2,3] | position: [-3,-2,-1,0,+1,+2,+3]
"""
ROTATION = [0,1,2,3] # clock-wise
POSITION = [-4,-3,-2,-1,0,+1,+2,+3,+4,+5]
ACTIONS = [combo for combo in itertools.product(ROTATION, POSITION)]
def generate_action_seq(act):
    rot, pos = act
    act_list = ["forward"] * rot if rot < 3 else ["reverse"]
    act_list = act_list + ['left'] * -pos if pos < 0 else act_list + ['right'] * pos
    act_list.append('hard drop')
    return act_list

class MatrisEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, no_display=True, real_tick=False, reward_functions=None):
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
        self.action_list = ACTIONS
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,22,10), dtype=np.int)

    def step(self, action_id):
        self.game.matris.drop_bonus = False
        timepassed = self.game.clock.tick(50) if self.real_tick else 20
        previous_state = self.game.matris.get_state()
        act = generate_action_seq(self.action_list[action_id])
        reward = self.game.matris.step_update(act, timepassed/1000)
        done = self.game.matris.done
        state = self.game.matris.get_state()
        info = self.game.matris.get_info()

        if self.reward_functions is not None:
            for reward_function in self.reward_functions:
                reward += reward_function(previous_state, state)

        reward /= 1000

        return state, reward, done, info

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
    env = MatrisEnv(no_display=False)
    for i in range(1000):
        state, reward, done, info = env.step(env.action_space.sample())
        print(f"Reward: {reward}")
        if not done:
            env.render()
        else:
            env.reset()
            
    print("Game over!")