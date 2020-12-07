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
from actions import ACTIONS
import numpy as np

class MatrisEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, no_display=True, real_tick=False):
        if not no_display:
            pygame.init()
            pygame.display.set_caption("MaTris")

            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        else:
            self.screen = None
        self.game = Game()
        self.game.gym_init(self.screen)

        self.real_tick = real_tick
        self.action_list = ACTIONS
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(low=0, high=3, shape=(1,20,10), dtype=np.int)
        # self.observation_space = spaces.Dict({"board": spaces.Box(low=0, high=1, shape=(3,20,10), dtype=np.int), 
        #                                       "current": spaces.Box(low=0, high=7, shape=(1,), dtype=np.int),  
        #                                       "next": spaces.Box(low=0, high=7, shape=(1,), dtype=np.int),
        #                                       "hold": spaces.Box(low=0, high=7, shape=(1,), dtype=np.int),
        #                                     }) 

    def step(self, action_id):
        timepassed = self.game.clock.tick(50) if self.real_tick else 20
        reward = self.game.matris.step_update(self.action_list[action_id], timepassed/1000)
        done = self.game.matris.done
        state = self.game.matris.get_state()
        info = None
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