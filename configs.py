import argparse
import os
import sys
import math

class Config(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--epsilon_start", type=float, default=1.0)
        self.parser.add_argument("--epsilon_final", type=float, default=0.01)
        self.parser.add_argument("--epsilon_decay", type=int, default=30000)

        self.parser.add_argument("--gamma", type=float, default=0.99)
        self.parser.add_argument("--lr", type=float, default=1e-4)
        
        self.parser.add_argument("--tar_net_update_freq", type=int, default=1000)
        self.parser.add_argument("--exp_replay_size", type=int, default=32)

        self.parser.add_argument("--learn_start", type=int, default=10000)
        self.parser.add_argument("--max_frames", type=int, default=1000000)

    
    def parse(self):
        config = self.parser.parse_args()
        config.epsilon_by_frame = lambda frame_idx:\
             config.epsilon_final + (config.epsilon_start - config.epsilon_final) \
             * math.exp(-1. * frame_idx / config.epsilon_decay)
        return config

