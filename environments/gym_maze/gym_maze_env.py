import gymnasium as gym
import IPython.display as ipd
import numpy as np
import environments
import random
import pandas as pd
import math
import discretization.grid.partition as prt
from collections import defaultdict
import sys
sys.path.append("../gym_maze/")

class maze:
    def __init__(self):
        self.env = gym.make("maze-sample-5x5-v0")
        
        # self.initial_calculations()
        observation, _ = self.env.reset()
        self.init = observation
        self.state = observation
        self.state_shape = [2]
        self.terminated = False
        self.goal = self.env.env.env.env.get_goal()
        self.partition_states()
    
    def reset(self):
        observation, _  = self.env.reset()
        self.state = observation
    
    def partition_states(self):
        nrPerDim = [5, 5]
    
        regionWidth = [
            5/nrPerDim[0],    # x position ∈ [0, 5]
            5/nrPerDim[1],      # cart velocity ∈ [-5, 0]
        ]
        
        origin = [2.5, -2.5]
        
        partition = prt.define_partition(
            dim=2,
            nrPerDim=nrPerDim,
            regionWidth=regionWidth,
            origin=origin
        )
        
        self.partition = partition
        self.nb_states = len(partition["center"])
    def state2region(self, state):
        idx = prt.state2region(state, self.partition, self.partition['c_tuple'])
        return idx[0]
    
    def step(self, action):
        old_state = self.state
        next_state, reward, done, truncated, info = self.env.step(action)
        self.terminated = done
        self.crashed = False # Edit when trap exists
        
        return old_state, next_state, reward
    
    def is_done(self):
        return self.terminated
    
    def check_crashed(self):
        return self.crashed
    
    def get_reward_function(self):
        pass
    
    def get_nb_actions(self):
        return 4
    
    def get_state_shape(self):
        return self.state_shape
    
    def get_state(self):
        return self.state
    
    def get_nb_states(self):
        return self.nb_states
    
    def get_traps(self):
        return []
    
    def get_goal_state(self):
        return []
    
    def get_init_state(self):
        print("our init state = ", self.init)
        return [0.0, 0.0], int(self.init)
    
class mazePolicy:
    def __init__(self, env, epsilon=0.1):
        self.nb_states = env.nb_states
        self.nb_actions = 2
        self.pi = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        self.env = env
        self.epsilon = epsilon
        
        self.compute_baseline()
        
    def compute_baseline(self):
        self.pi = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        # random baseline for now, might do something with this later
    