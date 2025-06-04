import gymnasium as gym
from stormvogel.extensions.gym_grid import *
from stormvogel import *
import IPython.display as ipd
import numpy as np
import stormvogel.stormpy_utils.mapping as mapping
import environments
import random

class gymTaxi:
    def __init__(self):
        self.env = gym.make("CustomTaxi-v0", render_mode="rgb_array")
        # self.sv_model = gymnasium_grid_to_stormvogel(self.env)
        self.initial_calculations()
        observation, _ = self.env.env.env.reset()
        self.init = observation
        self.state = observation
        self.terminated = False
        
        # print(self.env.env.env.encode(0, 0, 0, 0))
        # print(self.env.env.env.encode(0, 0, 0, 1))
        # print(self.env.env.env.encode(0, 0, 0, 2))
        # print(self.env.env.env.encode(0, 0, 0, 3))
        # print(self.env.env.env.encode(0, 4, 1, 0))
        # print(self.env.env.env.encode(0, 4, 1, 1))
        # print(self.env.env.env.encode(0, 4, 1, 2))
        # print(self.env.env.env.encode(0, 4, 1, 3))
        # print(self.env.env.env.encode(4, 0, 2, 0))
        # print(self.env.env.env.encode(4, 0, 2, 1))
        # print(self.env.env.env.encode(4, 0, 2, 2))
        # print(self.env.env.env.encode(4, 0, 2, 3))
        # print(self.env.env.env.encode(4, 3, 3, 0))
        # print(self.env.env.env.encode(4, 3, 3, 1))
        # print(self.env.env.env.encode(4, 3, 3, 2))
        # print(self.env.env.env.encode(4, 3, 3, 3))
        
        # print(self.goal_states)
        # for (state,action,next_state) in self.transition_model.keys():
        #     if state != self.nb_states-1:
        #         taxi_row, taxi_col, pass_loc, dest_idx = self.env.env.env.decode(state)
        #         # prob = self.transition_model[(state,action,next_state)]
        #         print(f"State: {state}: ({taxi_row},{taxi_col}), pass at {pass_loc}, dest at {dest_idx}), taking action {action} takes you to: {next_state}")
        # print("in ", state, "using ", action, "to ", next_state)
    
    def pick_initial_state(self):
        return self.env.env.env.pick_initial_state()  
    def reset(self):
        observation, _  = self.env.env.env.reset()
        self.state = observation
    
    def step(self, action):
        # Check if crash, if so, move to next anyways and end episode
        old_state = self.state
        next_state, reward, terminated, truncated, info = self.env.env.env.step(action)
        self.terminated = terminated
        self.state = next_state
        if old_state >= 500:
            print(old_state, " to ", self.state)
            print("Took action from unreachable state")
        return old_state, next_state, reward
    
    def is_done(self):
        # if self.terminated:
        #     # if self.state == self.nb_states-1:
        #     #     print("Died!!")
        #     if self.state == self.nb_states-2:
        #         print("correct drop off")
        #     # elif self.state ==self.nb_states-3:
        #     #     print("wrong drop off, passanger angry")
        return self.terminated
    
    
    def initial_calculations(self):
        P = self.env.env.env.P
        self.reward_model = {}
        self.transition_model = {}
        self.nb_states = len(P)
        self.nb_actions = len(P[0])
        self.goal_states = []
        for state in P:
            if self.env.env.env.isGoalState(state):
                self.goal_states.append(state)
            if state < self.nb_states-3:
                for action in P[state]:
                    for prob, next_state, reward, done in P[state][action]:
                        self.transition_model[(state, action, next_state)] = prob
                        self.reward_model[(state, next_state)] = reward

        print("goal States are: ", self.goal_states)
        self.valid_states = []
        for s in range(self.nb_states-3):
            _, _, pass_loc, dest_loc = self.env.env.env.decode(s)
            if pass_loc != dest_loc and s not in self.goal_states:
                self.valid_states.append(s)
      
    def set_random_state(self):          
        state = np.random.choice(self.valid_states)
        self.state = state
        self.env.env.env.set_state(state)
        
    def get_reward_function(self):
        for reward in self.reward_model.values():
            if reward > 15:
                print(reward)
        return self.reward_model
    
    def get_transition_function(self):
        return self.transition_model
                
    def get_baseline_policy_old(self, epsilon):
        pi_r = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        
        stormpy_model = mapping.stormvogel_to_stormpy(self.sv_model)
        prop = stormpy.parse_properties('Rmax=? [S]')
        # res = model_checking(self.model, f)
        res = stormpy.model_checking(stormpy_model, prop[0], extract_scheduler=True)
        scheduler = res.scheduler
        
        pi_sched = np.full((self.nb_states, self.nb_actions), 0)   
        for next_state in range(self.nb_states):
            choice = scheduler.get_choice(next_state).get_deterministic_choice()
            pi_sched[next_state][choice] = 1
        
        pi = (1-epsilon) * pi_sched + epsilon * pi_r        
        return pi    
    
    def get_baseline_policy(self, epsilon):
        pi_r = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        
        possible_destinations = []
        for i in range(4):
            loc_row, loc_col = self.env.env.env.locs[i]
            possible_destinations.append((loc_row, loc_col))
        
        pi_b = np.zeros((self.nb_states, self.nb_actions))
        for state in range(len(pi_b)-1):
            taxi_row, taxi_col, pass_loc, dest_loc = self.env.env.env.decode(state)
            dest_row, dest_col = self.env.env.env.locs[dest_loc]
            
            if (taxi_row, taxi_col) in possible_destinations:
                if pass_loc < 4:
                    pass_row, pass_col = self.env.env.env.locs[pass_loc]
                    if taxi_row == pass_row and taxi_col == pass_col:
                        # print(f"STATE: {state}, where {taxi_row} == {pass_row}, {taxi_col} == {pass_col}")
                        pi_b[state][4] = 1
                        # pi_b[state][0:4] = 0.125
                    else:
                        pi_b[state][0] = 0.25
                        pi_b[state][1] = 0.25
                        pi_b[state][2] = 0.25
                        pi_b[state][3] = 0.25
                elif taxi_row == dest_row and taxi_col == dest_col and pass_loc == 4:
                    pi_b[state][5] = 1
                else:
                    pi_b[state][0] = 0.25
                    pi_b[state][1] = 0.25
                    pi_b[state][2] = 0.25
                    pi_b[state][3] = 0.25
            else:
                pi_b[state][0] = 0.25
                pi_b[state][1] = 0.25
                pi_b[state][2] = 0.25
                pi_b[state][3] = 0.25
            # print("state = ", pi_b[state], " which sums to ", sum(pi_b[state]))
        # print(pi_b)
        pi_b[self.nb_states-1][:] = 1/self.nb_actions
        pi_b[self.nb_states-2][:] = 1/self.nb_actions
        pi_b[self.nb_states-3][:] = 1/self.nb_actions
        pi = (1-epsilon) * pi_b + epsilon * pi_r  
        # for state in range(len(pi_b)):
        #     print("state ", state, ": ", pi_b[state] ) 
        # print(pi)     
        return pi
    
    def get_nb_states(self):
        return self.nb_states
    
    def get_nb_actions(self):
        return self.nb_actions
    # def get_traps(self):
    #     return self.trap_states
    def get_goal(self):
        return self.goal_states
    
    def get_init_state(self):
        return self.init
    
    def get_state(self):
        return self.state
    
    
class gymIce:
    def __init__(self):
        self.env = gym.make("FrozenLakeCustom-v0", render_mode="rgb_array", map_name = "8x8", is_slippery=True)
        self.sv_model = gymnasium_grid_to_stormvogel(self.env)
        self.initial_calculations()
        observation, _ = self.env.env.env.reset()
        self.init = observation
        self.state = observation
        self.goal = get_target_state(self.env)
        self.terminated = False
        
    def reset(self):
        observation, _  = self.env.env.env.reset()
        self.state = observation
    
    def set_random_state(self):
        possible_states = [s for s in range(self.nb_states) if s not in self.traps and s != self.goal]    
        random_state = random.choice(possible_states)
        self.env.env.env.s = random_state
        self.state = random_state
        print(self.env.env.env.s, "==", self.state )
        
    def step(self, action):
        old_state = self.state
        next_state, reward, terminated, truncated, info = self.env.env.env.step(action)
        self.terminated = terminated
        self.state = next_state
        return old_state, next_state, reward
    
    def is_done(self):
        #     else:
        #         print("Fell :(, self.state = )", self.state)
        return self.terminated
    
    def initial_calculations(self):
        # print("goal: ", self.env.env.env.desc[0][0])
        P = self.env.env.env.P
        self.reward_model = {}
        self.transition_model = {}
        self.nb_states = len(P)
        self.nb_actions = len(P[0])
        self.traps = []
        for state in P:
            for action in P[state]:
                for prob, next_state, reward, done in P[state][action]:
                    self.transition_model[(state, action, next_state)] = prob
                    self.reward_model[(state, next_state)] = reward
            col, row = to_coordinate(state, self.env.env.env)
            if self.env.env.env.desc[row][col] == b"H":
                self.traps.append(state)

    def get_reward_function(self):
        return self.reward_model
   
    def get_transition_function(self):
        # print(self.transition_model)
        return self.transition_model
   
    def get_nb_states(self):
        return self.nb_states
    
    def get_nb_actions(self):
        return self.nb_actions
    
    def get_goal_state(self):
        return self.goal
    
    def get_init_state(self):
        return self.init
    
    def get_traps(self):
        return self.traps
    def get_baseline_policy(self, epsilon):
        pi_r = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        
        # stormpy_model = mapping.stormvogel_to_stormpy(self.sv_model)
        # prop = stormpy.parse_properties('Rmax=? [S]')
        # # res = model_checking(self.model, f)
        # res = stormpy.model_checking(stormpy_model, prop[0], extract_scheduler=True)
        # scheduler = res.scheduler
        
        # pi_sched = np.full((self.nb_states, self.nb_actions), 0)   
        # for next_state in range(self.nb_states):
        #     choice = scheduler.get_choice(next_state).get_deterministic_choice()
        #     pi_sched[next_state][choice] = 1
        
        # pi = (1-epsilon) * pi_sched + epsilon * pi_r        
        return pi_r  