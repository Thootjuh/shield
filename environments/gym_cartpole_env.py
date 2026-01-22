import gymnasium as gym
import IPython.display as ipd
import numpy as np
import environments
import random
import discretization.grid.partition as prt
from collections import defaultdict


class cartPole:
    def __init__(self):
        self.env = gym.make("CartPole-v0")

        # self.initial_calculations()
        observation, _ = self.env.reset()
        self.init = observation
        self.state = observation
        self.state_shape = [4]
        self.terminated = False
        self.goal = [0.0, 0.0, 0.0, 0.0]
        self.partition_states()
        
    def reset(self):
        observation, _  = self.env.reset()
        self.state = observation
    
    def partition_states(self):
        # Non-terminal region definitions
        nrPerDim = [6, 6, 4, 6]
        # nrPerDim = [10, 10, 10, 10]
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
        # old_region = self.state2region(old_state)
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.terminated = terminated and not truncated
        if self.terminated:
            reward = 0
        self.truncated = truncated
        self.done = terminated or truncated
        self.state = next_state
        # next_region = self.state2region(next_state)

        return old_state, next_state, reward
    
    def is_terminated(self):
        return self.terminated
    
    def is_truncated(self):
        return self.truncated
    
    def is_done(self):
        return self.terminated
    
    def check_crashed(self):
        return self.crashed
    
    def get_reward_function(self):
        reward_function = defaultdict(float)
        for state in range(self.nb_states):
            for next_state in range(self.nb_states):
                # if next_state != self.partition["terminal_idx"]:
                reward_function[(state, next_state)] = 1
            # reward_function[(state, self.partition["terminal_idx"])] = 0
        # R[:, nb_states-1] = FALL_REWARD
        # reward_function = {key: value for key, value in reward_function.items() if value != 0.0}
        return reward_function
    
    def get_transition_function(self):
        return self.transition_model
    
    def get_nb_actions(self):
        return 2
    
    def get_state_shape(self):
        return self.state_shape
    
    def get_state(self):
        return self.state
    
    def get_nb_states(self):
        return self.nb_states
    
    def get_traps(self):
        return self.partition["terminal_idx"]
    
    def get_goal_state(self):
        states = np.arange(0, self.nb_states)
        states = states[states!=self.partition["terminal_idx"]] 
        assert self.partition["terminal_idx"] not in states
        # print("FAAAAAAAAAAA")
        return states
        # return [self.state2region(self.goal)]
    
    def get_init_state(self):
        return self.init, self.state2region(self.init)
    
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
    
    def get_successor_states(self, state, action):
        dim = 4
        dt = 0.02
        A = np.array([
            [1, dt, 0, 0],
            [0, 1, 0.02 * 9.8 * 0.1 / 1.1, 0],
            [0, 0, 1, dt],
            [0, 0, 0.02 * 9.8 * 1.1 / 0.5, 1]
        ])
        B = np.array([[0], [dt / 1.1], [0], [dt * 1 / 0.5]])
        K = np.array([10.0, 3.0, 200.0, 20.0])
        A_cl = A - B @ K.reshape(1, -1)

        U_prime_values = [-2.0, 2.0]
        
        def noise_sampler():
            return np.random.normal(0, [0.01, 0.01, 0.01, 0.02], size=4)
        N = 1000
        samples = []
        
        for _ in range(N):
            samples.append(noise_sampler())
        succ = prt.successor_states(state, action, A_cl, B, U_prime_values, samples, self, 1000)
        return succ

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
    
    def compute_baseline_size(self, states):
        pi = np.zeros((states, self.nb_actions))
        # left_states = self.env.regions_left_of_origin()
        # right_states = self.env.regions_right_of_origin()
        
        for state in range(len(pi)):
            # if state in left_states:
            #     pi[state][1] = 1.0
            # elif state in right_states:
            #     pi[state][0] = 1.0
            # else:
            pi[state][0] = 0.5
            pi[state][1] = 0.5
                
        # pi_b = (1 - self.epsilon) * pi + self.epsilon * self.pi
        return pi
        
                
     