import gymnasium as gym
from stormvogel.extensions.gym_grid import *
from stormvogel import *
import IPython.display as ipd
import numpy as np
import stormvogel.stormpy_utils.mapping as mapping
import environments
import random


class cartPole:
    def __init__(self):
        self.env = gym.make("CartPole-v0")

        # self.initial_calculations()
        observation, _ = self.env.env.env.reset()
        self.init = observation
        self.state = observation
        self.terminated = False
        
    def reset(self):
        observation, _  = self.env.env.env.reset()
        self.state = observation
        
    def step(self, action):
        # Check if crash, if so, move to next anyways and end episode
        old_state = self.state
        next_state, reward, terminated, truncated, info = self.env.env.env.step(action)
        self.terminated = terminated
        self.state = next_state

        return old_state, next_state, reward
    
    def is_done(self):
        return self.terminated
    
    def get_reward_function(self):
        return self.reward_model
    
    def get_transition_function(self):
        return self.transition_model
    
    def get_nb_actions(self):
        return 2
    
    def get_nb_states(self):
        pass
    
    def get_traps(self):
        pass