import gymnasium as gym
import IPython.display as ipd
import numpy as np
import environments
import random
import math
import discretization.grid.partition as prt

class gymIce:
    def __init__(self):
        env = gym.make("FrozenLakeContinuous-v0", render_mode="rgb_array", map_name = "8x8", is_slippery=True)
        # self.sv_model = gymnasium_grid_to_stormvogel(self.env)
        self.env = env.env.env
        self.partition_states()
        self.initial_calculations()
    
        observation, _ = self.env.reset()
        self.state_shape = [2]
        self.init = observation
        self.state = observation
        self.nb_actions = 4
        self.terminated = False
        
        
    def reset(self):
        observation, _  = self.env.reset()
        self.state = observation
    
    def set_random_state(self):
        possible_states = [s for s in range(self.nb_states) if s not in self.traps and s != self.goal]    
        random_state = random.choice(possible_states)
        self.env.s = random_state
        self.state = self.env._state_to_continuous(random_state)

    def partition_states(self):
        nrPerDim = [8, 8]
    
        regionWidth = [
            8/nrPerDim[0],    # x position ∈ [0, 8]
            8/nrPerDim[1],    # y position ∈ [0, 8]
        ]
        
        origin = [4.0, 4.0]
        
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
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.terminated = terminated
        self.truncated = truncated 
        self.state = next_state
        return old_state, next_state, reward
    
    def is_done(self):
        return self.terminated or self.truncated
    def is_truncated(self):
        return self.truncated
    def is_terminated(self):
        return self.terminated
    
    # from storm vogel
    def get_target_state(self, env):
        """Calculate the target state for an env. Works for FrozenLake and Cliffwalking"""
        return env.observation_space.n - 1

    # from storm vogel
    def to_coordinate(self, s, env):
        """Calculate the state's coordinates. Works for FrozenLake, Cliffwalking, and Taxi"""
        num_states = env.observation_space.n
        grid_size = int(math.sqrt(num_states))
        x_target = int(s % grid_size)
        y_target = int(s // grid_size)
        return x_target, y_target

    def initial_calculations(self):

        self.reward_model = {}
        self.traps = []
        self.goal = []

        for region, center in enumerate(self.partition["center"]):
            x, y = center
            col = int(x)
            row = int(y)

            tile = self.env.get_tile_from_continuous_state([x,y])
            reward = 0
            # Trap regions
            if tile == "H":
                self.traps.append(region)
                reward = self.env.get_trap_reward()

            # Goal regions
            elif tile == "G":
                reward = self.env.get_goal_reward()
                self.goal.append(region)
                
            elif tile=="F" or tile=="F":
                reward = self.env.get_step_reward()
                
            if reward != 0:
                for prev_region, _ in enumerate(self.partition["center"]):
                    self.reward_model[(prev_region, region)] = reward

                

    def get_reward_function(self):
        return self.reward_model
    
    # def get_reward_function_no_neg(self):
        # return self.reward_model_no_neg
    
    def get_transition_function(self):
        return self.transition_model
   
    def get_state(self):
        return self.state
    
    def get_nb_states(self):
        return self.nb_states
    
    def get_nb_actions(self):
        return self.nb_actions
    
    def get_goal_state(self):
        return self.goal
    
    def get_init_state(self):
        return self.init, self.state2region(self.init)
    
    def get_traps(self):
        print("traps are ", self.traps)
        return self.traps
    
    def get_state_shape(self):
        return self.state_shape
    
    def get_baseline_policy(self, epsilon):
        pi_r = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        pi_b = np.zeros((self.nb_states, self.nb_actions))
        pi_b[:,1] = 0.5
        pi_b[:,2] = 0.5
        
        pi = (1-epsilon) * pi_b + epsilon * pi_r
        return pi_r
    
    def compute_baseline_policy_size(self, size, epsilon):
        pi_r = np.ones((size, self.nb_actions)) / self.nb_actions
        pi_b = np.zeros((size, self.nb_actions))
        pi_b[:,1] = 0.5
        pi_b[:,2] = 0.5
        
        pi = (1-epsilon) * pi_b + epsilon * pi_r
        return pi_r