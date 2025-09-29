import gymnasium as gym
import IPython.display as ipd
import numpy as np
import environments
import random
import partition as prt
from collections import defaultdict

class cartPole:
    def __init__(self):
        self.env = gym.make("CartPole-v0")

        # self.initial_calculations()
        observation, _ = self.env.reset()
        self.init = observation
        self.state = observation
        self.terminated = False
        self.goal = [0.0, 0.0, 0.0, 0.0]
        self.partition_states()
        
    def reset(self):
        observation, _  = self.env.reset()
        self.state = observation
    
    def partition_states(self):
        # Non-terminal region definitions
        nrPerDim = [10, 10, 10, 10]

        regionWidth = [
            (2.4*2)/nrPerDim[0],    # cart position ∈ [-2.4, 2.4]
            (6.0)/nrPerDim[1],      # cart velocity ∈ [-3, 3]
            (0.2095*2)/nrPerDim[2], # pole angle ∈ [-0.2095, 0.2095]
            (7.0)/nrPerDim[3]       # pole angular velocity ∈ [-3.5, 3.5]
        ]

        origin = [0.0, 0.0, 0.0, 0.0]

        partition = prt.define_partition(
            dim=4,
            nrPerDim=nrPerDim,
            regionWidth=regionWidth,
            origin=origin
        )

        # Add one extra region for terminal states
        terminal_region_idx = len(partition["center"])
        partition["terminal_idx"] = terminal_region_idx
        
        self.partition = partition
        self.nb_states = len(partition["center"]) + 1
        
    def state2region(self, state):
        # Check terminal condition
        if abs(state[0]) > 2.4 or abs(state[2]) > 0.2095:
            return self.partition["terminal_idx"]

        # Clip unbounded dimensions
        state = state.copy()
        state[1] = np.clip(state[1], -3, 3)       # cart velocity
        state[3] = np.clip(state[3], -3.5, 3.5)   # pole angular velocity

        # Map to region
        idx = prt.state2region(state, self.partition, self.partition['c_tuple'])
        if idx == None:
            return self.partition["terminal_idx"]
        return idx[0]
        
    def step(self, action):
        old_state = self.state
        old_region = self.state2region(old_state)
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.terminated = terminated or truncated
        self.state = next_state
        next_region = self.state2region(next_state)

        return old_region, next_region, reward
    
    def is_done(self):
        return self.terminated
    
    def get_reward_function(self):
        reward_function = defaultdict(float)
        for state in range(self.nb_states):
            for next_state in range(self.nb_states):
                if next_state != self.partition["terminal_idx"]:
                    reward_function[(state, next_state)] = 1
            # reward_function[(state, self.partition["terminal_idx"])] = 0
        # R[:, nb_states-1] = FALL_REWARD
        # reward_function = {key: value for key, value in reward_function.items() if value != 0.0}
        return reward_function
    
    def get_transition_function(self):
        return self.transition_model
    
    def get_nb_actions(self):
        return 2
    
    def get_state(self):
        return self.state
    
    def get_nb_states(self):
        return self.nb_states
    
    def get_traps(self):
        return self.partition["terminal_idx"]
    
    def get_goal_state(self):
        return self.state2region(self.goal)
    
    def get_init_state(self):
        return self.state2region(self.init)
    
    def regions_left_of_origin(self):
        """
        Return indices of regions where *all* continuous states 
        have cart position < 0.
        """
        # Dimension 0 = cart position
        mask = self.partition["upp"][:, 0] < 0
        left_indices = np.where(mask)[0]
        return left_indices
    
    def regions_right_of_origin(self):
        """
        Return indices of regions where *all* continuous states 
        have cart position > 0.
        """
        # Dimension 0 = cart position
        mask = self.partition["low"][:, 0] > 0
        right_indices = np.where(mask)[0]
        return right_indices
    
class cartPolePolicy:
    def __init__(self, env, epsilon=0.1):
        self.nb_states = env.nb_states
        self.nb_actions = 2
        self.pi = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        self.env = env
        self.epsilon = epsilon
        
        self.compute_baseline()
        
    def compute_baseline(self):
        pi = np.zeros((self.nb_states, self.nb_actions))
        left_states = self.env.regions_left_of_origin()
        right_states = self.env.regions_right_of_origin()
        
        for state in range(len(pi)):
            if state in left_states:
                pi[state][1] = 1.0
            elif state in right_states:
                pi[state][0] = 1.0
            else:
                pi[state][0] = 0.5
                pi[state][1] = 0.5
                
        self.pi = (1 - self.epsilon) * pi + self.epsilon * self.pi
                
     