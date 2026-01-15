import gymnasium as gym
from environments.MountainCarCrashWrapper import MountainCarCrashWrapper
import IPython.display as ipd
import numpy as np
import environments
import random
import discretization.grid.partition as prt
from collections import defaultdict

CRASH_BOUND = -1.0
MOTOR_FAILURE = 0.65
# Note: none of the algorithms ever crash
# Note: Algorithms barely ever reach the goal
# Note: The shield has a value of 0.0 or 1.0 (or close to it) for all* states
# Note: Maybe try to expand the environment: Add more crash option
class crashingMountainCar:
    def __init__(self):
        # create environment
        self.env = MountainCarCrashWrapper(gym.make("MountainCar-v0"), CRASH_BOUND, MOTOR_FAILURE)
        observation, _ = self.env.reset()
        self.state_shape = [2]
        self.init = observation
        self.state = observation
        self.terminated = False
        # self.goal = [0.0, 0.0, 0.0, 0.0]
        self.partition_states()
    
    def set_random_state(self):
        pos_low, pos_high = -0.9, 0.5
        vel_low, vel_high = -0.05, 0.05
        # pos_low, pos_high = 0.45, 0.49
        # vel_low, vel_high = 0.055, 0.059
        position = np.random.uniform(pos_low, pos_high)
        velocity = np.random.uniform(vel_low, vel_high)  
        
        if position <= pos_low and velocity < 0:
            velocity = 0
        if position >= pos_high and velocity > 0:
            velocity = 0
        
        self.env.set_state(np.array([position, velocity], dtype=np.float32))
        # self.env.state = 
        self.state = np.array([position, velocity], dtype=np.float32)
        # print(self.state)
        
    def reset(self):
        observation, _  = self.env.reset()
        self.state = observation
        
    def partition_states(self):
        # Non-terminal region definitions
        # State: [position, velocity]
            nrPerDim = [15, 14]

            # Define region widths based on known limits of MountainCar
            # position ∈ [-1.0, 0.5], velocity ∈ [-0.07, 0.07]
            regionWidth = [
                (0.5 - (CRASH_BOUND)) / nrPerDim[0],  # position width
                (MOTOR_FAILURE - (-1*MOTOR_FAILURE)) / nrPerDim[1] # velocity width
            ]

            # Choose origin at the center of the space
            origin = [-0.25, 0.0]

            # Build the partition (using your partition helper)
            partition = prt.define_partition(
                dim=2,
                nrPerDim=nrPerDim,
                regionWidth=regionWidth,
                origin=origin
            )

            # Add terminal regions
            goal_region_idx  = len(partition["center"])
            crash_region_idx = goal_region_idx + 1

            partition["goal_idx"]  = goal_region_idx
            partition["crash_idx"] = crash_region_idx

            # Store and record total number of discrete regions (including terminals)
            self.partition = partition
            self.nb_states = len(partition["center"]) + 2
            # print(self.nb_states, ": NB STATES")
            
    def state2region(self, state):
        # Check terminal and goal condition
        position, velocity = state
        
        if position > 0.5:
            return self.partition["goal_idx"]
        
        if position < -CRASH_BOUND or abs(velocity) > MOTOR_FAILURE:
            # print("died")
            # print(position)
            # print(velocity)
            return self.partition["crash_idx"]

        # Clip unbounded dimensions
        state = state.copy()
        state[0] = np.clip(state[0], -1.0, 0.5)   # position bounds
        state[1] = np.clip(state[1], -1*MOTOR_FAILURE, MOTOR_FAILURE) # velocity bounds

        # Map to region
        idx = prt.state2region(state, self.partition, self.partition["c_tuple"])
        if idx is None:
            # print("It actually goes wrong here tihi")
            return self.partition["crash_idx"]
        return idx[0]
    
    def step(self, action):

        old_state = self.state

        # old_region = self.state2region(old_state)
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.terminated = terminated or truncated
        self.crashed = terminated
        self.state = next_state
        
        if self.terminated and (reward == -250.0 or reward == 100):
            print("finished the episode without truncation")
        if self.terminated:
            print("done:", reward)
        # print("taking step, reached :", next_state)
        # if self.state2region(next_state) == self.partition["crash_idx"]:
            # print("kill yourself IRL")
            # print(next_state)
        # next_region = self.state2region(next_state)

        return old_state, next_state, reward
    
    def check_crashed(self):
        return self.crashed
    def is_done(self):
        
        return self.terminated
    
    
    def get_reward_function(self):
        reward_function = defaultdict(float)
        for state in range(self.nb_states):
            for next_state in range(self.nb_states):
                if next_state != self.partition["crash_idx"] and next_state != self.partition["goal_idx"] :
                    reward_function[(state, next_state)] = -1
                if next_state == self.partition["crash_idx"]:
                    reward_function[(state, next_state)] = self.env.crash_penalty
                if next_state == self.partition["goal_idx"]:
                    reward_function[(state, next_state)] = self.env.success_reward
        return reward_function
                    
    def get_transition_function(self):
        return self.transition_model
    
    def get_nb_actions(self):
        return 3
    
    def get_state(self):
        return self.state
    
    def get_nb_states(self):
        return self.nb_states
    
    def get_state_shape(self):
        return self.state_shape
    
    def get_traps(self):
        return self.partition["crash_idx"]
    
    def get_goal_state(self):
        return self.partition["goal_idx"]
    
    def get_init_state(self):
        return self.init, self.state2region(self.init)
    
    def get_successor_states(self, state, action):
        dim = 2
        dt = 1.0
        A = np.array([
            [1.0, dt],
            [0.0, 1.0]
        ])
        B = np.array([[0.01], [0.01]])
        U_prime_values = [-1.0, 0.0, 1.0]
        K = np.array([-0.5, 0.5])
        
        A_cl = A - B @ K.reshape(1, -1)

        def noise_sampler():
            return np.random.normal(0, [0.01, 0.001], size=dim)
        N = 1000
        samples = [noise_sampler() for _ in range(N)]
        
        for _ in range(N):
            samples.append(noise_sampler())
        succ = prt.successor_states(state, action, A_cl, B, U_prime_values, samples, self, 1000)
        return succ
    
class crashingMountainCarPolicy:
    def __init__(self, env, epsilon=0.1):
        self.nb_states = env.nb_states
        self.nb_actions = env.get_nb_actions()
        self.pi = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        self.env = env
        self.epsilon = epsilon
        
        self.compute_baseline()
        
    def compute_baseline(self):
        pi = np.zeros((self.nb_states, self.nb_actions))
        
        for state in range(len(pi)):
            if state == self.env.get_traps() or state == self.env.get_goal_state():
                pi[state][1] = 1.0
            else: 
                vel_low = self.env.partition["low"][state, 1]
                vel_high = self.env.partition["upp"][state, 1]
                vel_center = (vel_low + vel_high) / 2.0
                
                if vel_center < 0.055: 
                    pi[state][2] = 1.0
                else: 
                    pi[state][0] = 0.5
                    pi[state][1] = 0.5
                    
                
        self.pi = (1 - self.epsilon) * pi + self.epsilon * self.pi
        # self.pi = s
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