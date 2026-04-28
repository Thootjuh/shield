import IPython.display as ipd
import numpy as np
import environments
import random
import discretization.grid.partition as prt
from collections import defaultdict
from collections.abc import Mapping
import copy

ACTION_TRANSLATOR = {
    'UP': np.array([0, 0.1]),
    'RIGHT': np.array([0.1, 0]),
    'DOWN': np.array([0, -0.1]),
    'LEFT': np.array([-0.1, 0]),
}


# Standard
GOAL_REWARD = 10
TRAP_REWARD = -10
STEP_REWARD = 0

## Variant
# GOAL_REWARD = 10
# TRAP_REWARD = -1
# STEP_REWARD = -0.1


class RewardDict(Mapping):
    def __init__(self, nb_states, terminal_idx, goal_idx, reward_per_next):
        self.nb_states = nb_states
        self.terminal_idx = terminal_idx
        self.goal_idx = goal_idx
        self.reward_per_next = reward_per_next
            
    def __getitem__(self, key):
        state, next_state = key

        if next_state == self.goal_idx:
            return GOAL_REWARD
        if next_state == self.terminal_idx:
            return TRAP_REWARD

        return STEP_REWARD

    def __iter__(self):
        raise RuntimeError("Full iteration over N^2 reward table is infeasible.")

    def __len__(self):
        return self.nb_states * self.nb_states

class MovingObstacles:
    def __init__(self):
        self.state = [0, 0, 0, 0] #Agent X-pos, Agent Y-pos, Vertical offset of obstacle 1, Horizontal offset of obstacle 2
        self.init = copy.deepcopy(self.state)
        self.obstacle_start = [[0.3, 0.6], [0.4, 0.3]]
        self.goal_region = [[[0.35, 0.35], [0.5, 0.5]], [[0.85, 0.85],[1.0, 1.0]]]
        self.obstacle_width = 0.1
        # self.goal_region = [[[0.35, 0.5],[1.0, 1.0]]]
        self.partition_states()
        self.trap = []
        self.goal = []
        self.terminated = False
        self.nb_actions = len(ACTION_TRANSLATOR)
        self.state_shape= [len(self.state)]
        
    def reset(self):
        self.state = [0, 0, 0, 0]
        self.terminated = False
        
    def partition_states(self):
            # Non-terminal region definitions for LunarLander
            # (coarse discretization example; adjust later)
            nrPerDim = [10, 10, 4, 4]
            # Approximate observation bounds used for partitioning
            regionWidth = [
                1.0 / nrPerDim[0],        # x position ∈ [0, 1.0]
                1.0 / nrPerDim[1],        # y position ∈ [0, 1.0]
                0.42 / nrPerDim[2],       # x velocity ∈ [-0.2, 0.2]
                0.42 / nrPerDim[3],       # y velocity ∈ [-0.2, 0.2]

            ]

            origin = [
                0.5, 
                0.5,
                0.0,
                0.0,
            ]

            partition = prt.define_partition(
                dim=4,
                nrPerDim=nrPerDim,
                regionWidth=regionWidth,
                origin=origin
            )
            
            # Add one extra region for trap states and one for goal states
            terminal_region_idx = len(partition["center"])
            partition["terminal_idx"] = terminal_region_idx
            
            terminal_region_idx = len(partition["center"])+1
            partition["goal_idx"] = terminal_region_idx
            
            self.partition = partition
            self.nb_states = len(partition["center"]) + 2
    
    def state2region(self, state):
        # Check if agent overlaps with trap -> trap_idx
        if self.check_trap(state):
            return self.partition["terminal_idx"]
        
        # Check if agent overlaps with goal -> goal_idx
        if self.check_goal(state):
            return self.partition["goal_idx"]
    
    
        # Return normal state2region
        idx = prt.state2region(state, self.partition, self.partition['c_tuple'])
        return idx[0]

            
    def step(self, action):
        if self.terminated:
            print("You have already terminated, this should not be printed ever ever ever ever")
        # Move agent
        old_state = copy.deepcopy(self.state)
        action_coordinates = list(ACTION_TRANSLATOR.values())[action]
        # print(action_coordinates)
        new_x = min(1.0, max(0.0, old_state[0]+action_coordinates[0]+random.uniform(-0.025, 0.025)))
        new_y = min(1.0, max(0.0, old_state[1]+action_coordinates[1]+random.uniform(-0.025, 0.025)))
        # Move Obstacles
        movement = random.uniform(-0.05, 0.05)
        new_offset_1 = min(0.2, max(-0.2, old_state[2]+movement))
        movement = random.uniform(-0.05, 0.05)
        new_offset_2 = min(0.2, max(-0.2, old_state[2]+movement))
        
        new_state = [new_x, new_y, new_offset_1, new_offset_2]
        self.state = new_state
        # print(new_state)
        # Check if agent is on an obstacle or goal (if both obstacle and goal, its still considered a failure)
        if self.check_trap(new_state):
            self.terminated = True
            return old_state, new_state, TRAP_REWARD
        elif self.check_goal(new_state):
            self.terminated = True
            return old_state, new_state, GOAL_REWARD
        return old_state, new_state, STEP_REWARD
    
    def check_trap(self, state):
        agent_x, agent_y = state[0], state[1]

        # Trap 1
        trap1_x = self.obstacle_start[0][0]
        trap1_y = self.obstacle_start[0][1] + state[2]

        # Trap 2
        trap2_x = self.obstacle_start[1][0] + state[3]
        trap2_y = self.obstacle_start[1][1]

        half_width = self.obstacle_width/2
        
        def in_square(ax, ay, ox, oy):
            return (ox - half_width <= ax <= ox + half_width and
                oy - half_width <= ay <= oy + half_width)
            
        # Check both traps
        if in_square(agent_x, agent_y, trap1_x, trap1_y):
            return True
        if in_square(agent_x, agent_y, trap2_x, trap2_y):
            return True

        return False

    
    def check_goal(self, state):
        agent_x, agent_y = state[0], state[1]

        for region in self.goal_region:
            (x_min, y_min), (x_max, y_max) = region

            if x_min <= agent_x <= x_max and y_min <= agent_y <= y_max:
                return True

        return False
    
    def is_done(self):
        return self.terminated
    
    def get_reward_from_centre(self, center):
        pass
    
    def get_reward_function(self):
        return RewardDict(
            nb_states=self.nb_states,
            terminal_idx=self.partition["terminal_idx"],
            goal_idx=self.partition["goal_idx"],
            reward_per_next=None,
        )
    
    def get_nb_actions(self):
        return self.nb_actions
    
    def get_state_shape(self):
        return self.state_shape
    
    def get_state(self):
        return self.state
    
    def get_nb_states(self):
        return self.nb_states
    
    def get_traps(self):
        return self.partition["terminal_idx"]
    
    def get_goal_state(self):
        return self.partition["goal_idx"]
    
    def get_init_state(self):
        return self.init, self.state2region(self.init)
    
    
    
class MovingObstaclesPolicy:
    def __init__(self, env, epsilon):
        self.nb_states = env.nb_states
        self.nb_actions = env.nb_actions
        self.pi_r= np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        self.env = env
        self.epsilon = epsilon
        
        self.pi = self.get_baseline_policy()
    
    def get_baseline_policy(self):
        pi_b = np.zeros((self.nb_states, self.nb_actions))
        pi_b[:,0] = 0.5
        pi_b[:,1] = 0.5
        
        pi = (1-self.epsilon) * pi_b + self.epsilon * self.pi_r
        return pi
    
    def compute_baseline_size(self, size):
        pi_r= np.ones((size, self.nb_actions)) / self.nb_actions
        pi_b = np.zeros((size, self.nb_actions))
        pi_b[:,0] = 0.5
        pi_b[:,1] = 0.5
        
        pi = (1-self.epsilon) * pi_b + self.epsilon * pi_r
        return pi