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

MAZE = "maze-sample-5x5-v0"
STANDARD_REWARD = -0.04
class maze:
    def __init__(self):
        self.env = gym.make(MAZE)
        
        # self.initial_calculations()
        observation = self.env.reset()
        # print("bbbbbbbbbbbbbbbbbbbb", observation)
        self.init = observation
        self.state = observation
        self.state_shape = [2]
        self.terminated = False
        self.goal = self.env.env.env.env.get_goal()
        self.partition_states()
    
    def reset(self):
        observation  = self.env.reset()
        self.state = observation
    
    def set_random_state(self):
        self.state = self.env.env.env.env.set_random_state()
        # print(self.state)
        
    def partition_states(self):
        nrPerDim = [5, 5]
    
        regionWidth = [
            5/nrPerDim[0],    # x position ∈ [0, 5]
            5/nrPerDim[1],      # cart velocity ∈ [5, 0]
        ]
        
        origin = [2.5, 2.5]
        
        partition = prt.define_partition(
            dim=2,
            nrPerDim=nrPerDim,
            regionWidth=regionWidth,
            origin=origin
        )
        
        self.partition = partition
        self.nb_states = len(partition["center"])
        
    def state2region(self, state):
        # print("AAAAAAAAAAAAAAAA")
        # print(state)
        # print(prt)
        # print("BBBBBBBBBBBBBB")
        # print(self.partition['c_tuple'])
        idx = prt.state2region(state, self.partition, self.partition['c_tuple'])
        return idx[0]
    
    def step(self, action):
        old_state = self.state
        next_state, reward, done, truncated, info = self.env.step(action)
        self.state = next_state
        self.terminated = done
        self.truncated = truncated # Edit when trap exists
        # print("s", old_state)
        # print("ns", next_state)
        return old_state, next_state, reward
    
    def is_done(self):
        return self.terminated or self.truncated
    def is_truncated(self):
        return self.truncated
    def is_terminated(self):
        return self.terminated
    
    def check_crashed(self):
        return self.crashed
    
    def get_reward_function(self):
        reward_matrix = np.zeros((self.nb_states, self.nb_states))
        goal = self.get_goal_state()
        traps = self.get_traps()
        for ns in range(len(reward_matrix)):
            if ns in goal:
                reward_matrix[:, ns] = 10
            elif ns in traps:
                reward_matrix[:, ns] = -10
            else:
                reward_matrix[:, ns] = STANDARD_REWARD
        return reward_matrix
    
    def get_nb_actions(self):
        return 4
    
    def get_state_shape(self):
        return self.state_shape
    
    def get_state(self):
        return self.state
    
    def get_nb_states(self):
        return self.nb_states
    
    
    def _regions_intersecting_cell(self, cell_coord, cell_size=1.0):
        """
        Return list of partition state indices whose region intersects
        the environment cell defined by top-left coordinate `cell_coord`.
        """
        partition = self.partition

        cell_low = np.array(cell_coord, dtype=float)
        cell_upp = cell_low + cell_size

        low = partition['low']
        upp = partition['upp']

        intersecting = []
        for i in range(partition['nr_regions']):
            # Intersection check per dimension
            if np.all((low[i] < cell_upp) & (upp[i] > cell_low)):
                intersecting.append(i)

        return intersecting
    
    def get_traps(self):
        traps = self.env.env.env.env.get_trap()

        trap_states = []
        for trap_cell in traps:
            trap_states.extend(self._regions_intersecting_cell(trap_cell))
        
        return list(set(trap_states))
    
    def get_goal_state(self):
        goal = self.env.env.env.env.get_goal()

        goal_states = self._regions_intersecting_cell(goal)
        return goal_states
    
    def get_init_state(self):
        # print("our init state = ", self.init)
        return [0.0, 0.0], self.state2region(self.init)
    
from discretization.MRL.mdp_utils import SolveMDP

class mazePolicy:
    def __init__(self, env, epsilon=0.1):
        # P, R = self.get_maze_MDP(MAZE)
        self.nb_states = env.get_nb_states()
        self.nb_actions = 4
        self.pi = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        self.env = env
        self.epsilon = epsilon
        # _, self.pi_star = SolveMDP(P, R, 0.98, 1e-10, True, "max")
        # self.pi = (1 - self.epsilon) * self.pi_star + self.epsilon * self.pi
        self.compute_baseline()
        
    # def opp_action(self, a):
    #     if a == 0:
    #         return 1
    #     elif a == 1:
    #         return 0
    #     elif a == 2:
    #         return 3
    #     elif a == 3:
    #         return 2    
    
    # def get_maze_MDP(self, maze):
    #     # initialize maze
    #     env = gym.make(maze)
    #     obs = env.reset()
    #     l = env.unwrapped.maze_size[0]

    #     # initialize matrices
    #     a = 4
    #     P = np.zeros((a, l * l + 1, l * l + 1))
    #     R = np.zeros((a, l * l + 1))
    #     # P = np.zeros((a, l*l, l*l))
    #     # R = np.zeros((a, l*l))

    #     # store clusters seen and cluster/action pairs seen in set
    #     c_seen = set()
    #     ca_seen = set()

    #     # initialize env, set reward of original
    #     obs = env.reset()
    #     ogc = int(obs[0] + obs[1] * l)
    #     reward = -1 / (l * l)

    #     while len(ca_seen) < (4 * l * l - 4):
    #         # update rewards for new cluster
    #         if ogc not in c_seen:
    #             for i in range(a):
    #                 # R[i, ogc] = reward
    #                 R[i, ogc] = reward
    #             c_seen.add(ogc)

    #         stop = False
    #         for i in range(a):
    #             if (ogc, i) not in ca_seen:
    #                 ca_seen.add((ogc, i))
    #                 # print(len(ca_seen))
    #                 new_obs, reward, done, truncated, info = env.step(i)

    #                 ogc_new = int(new_obs[0] + new_obs[1] * l)
    #                 # update probabilities
    #                 P[i, ogc, ogc_new] = 1
    #                 # print('updated', ogc, ogc_new, done)
    #                 if not done:
    #                     if ogc != ogc_new:
    #                         P[self.opp_action(i), ogc_new, ogc] = 1
    #                         # print('updated', ogc_new, ogc)
    #                         ca_seen.add((ogc_new, self.opp_action(i)))
    #                     # print(len(ca_seen))
    #                 ogc = ogc_new
    #                 # print('new ogc', ogc, done)

    #                 if done:
    #                     # set next state to sink
    #                     for i in range(a):
    #                         P[i, ogc_new, l * l] = 1
    #                         P[i, l * l, l * l] = 1
    #                         R[i, l * l] = 0
    #                         R[i, ogc_new] = 1
    #                     obs = env.reset()
    #                     ogc = int(obs[0] + obs[1] * l)

    #                 stop = True
    #             if stop:
    #                 break

    #         # if all seen already, take random step
    #         if not stop:
    #             action = env.action_space.sample()
    #             new_obs, reward, done, truncated, info = env.step(action)

    #             ogc = int(new_obs[0] + new_obs[1] * l)
    #             # print('trying random action', ogc)
    #             if done:
    #                 obs = env.reset()
    #                 ogc = int(obs[0] + obs[1] * l)
    #     env.close()

    #     return P, R
    
    def compute_baseline(self):
        
        # Go right and down
        # Somehow figure out where the walls are?
        # Base this on the original statespace?
        pi = np.zeros((self.nb_states, self.nb_actions))
        for state in range(len(pi)):
            pi[state][1] = 0.5
            pi[state][2] = 0.5
        self.pi = (1 - self.epsilon) * pi + self.epsilon * self.pi
        
    def compute_baseline_size(self, size):
        pi_r = np.ones((size, self.nb_actions)) / self.nb_actions
        pi_b = np.zeros((size, self.nb_actions))
        for state in range(len(pi_b)):
            pi_b[state][1] = 0.5
            pi_b[state][2] = 0.5
            
        pi = (1 - self.epsilon) * pi_b + self.epsilon * pi_r
        
        return pi
    