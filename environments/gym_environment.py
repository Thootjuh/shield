import gymnasium as gym
from stormvogel.extensions.gym_grid import *
from stormvogel import *
import IPython.display as ipd
import numpy as np
import stormvogel.stormpy_utils.mapping as mapping
import environments

class gymTaxi:
    def __init__(self):
        self.env = gym.make("CustomTaxi-v0", render_mode="rgb_array")
        self.sv_model = gymnasium_grid_to_stormvogel(self.env)
        self.initial_calculations()
        observation, _ = self.env.env.env.reset()
        self.init = observation
        self.state = observation
        self.terminated = False
        # print(self.goal_states)
        for (state,action,next_state) in self.transition_model.keys():
            taxi_row, taxi_col, pass_loc, dest_idx = self.env.env.env.decode(state)
                # prob = self.transition_model[(state,action,next_state)]
            print(f"State: {state}: ({taxi_row},{taxi_col}), pass at {pass_loc}, dest at {dest_idx}), taking action {action} takes you to: {next_state}")
        print("in ", state, "using ", action, "to ", next_state)
            
    def reset(self):
        observation, _  = self.env.env.env.reset()
        self.state = observation
    
    def step(self, action):
        if self.state == 500 or self.state in self.goal_states:
            print("WTF IS ER HIER NOU WEER AAN DE FUCKING HAND DIT KLOPT VOOR GEEN METER!!")
        # Check if crash, if so, move to next anyways and end episode
        old_state = self.state
        next_state, reward, terminated, truncated, info = self.env.env.env.step(action)
        self.terminated = terminated
        self.state = next_state
        if next_state in self.goal_states:
            print("WE DID IT!! WE REACHED A GOAL!! BE PROUD!!")
        return old_state, next_state, reward
    
    def is_done(self):
        return self.terminated
    
    
    def initial_calculations(self):
        P = self.env.env.env.P
        self.reward_model = {}
        self.transition_model = {}
        self.nb_states = len(P)
        self.nb_actions = len(P[0])
        self.goal_states = []
        for state in P:
            for action in P[state]:
                for prob, next_state, reward, done in P[state][action]:
                    self.transition_model[(state, action, next_state)] = prob
                    self.reward_model[(state, next_state)] = reward
            if self.env.env.env.isGoalState(state):
                self.goal_states.append(state)  
      

            
    def get_reward_function(self):
        for reward in self.reward_model.values():
            if reward > 15:
                print(reward)
        return self.reward_model
    
    def get_transition_function(self):
        return self.transition_model
                
    def get_baseline_policy(self, epsilon):
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
        
    def get_baseline_policy_old(self, epsilon):
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
                    pi_b[state][4] = 0.5
                    pi_b[state][0:4] = 0.125
                elif taxi_row == dest_row and taxi_col == dest_col and pass_loc == 4:
                    pi_b[state][5] = 1
                else:
                    pi_b[state][0:4] = 0.25
            else:
                pi_b[state][0:4] = 0.25
            # print("state = ", pi_b[state], " which sums to ", sum(pi_b[state]))
        # print(pi_b)
        pi_b[self.nb_states-1][:] = 1/self.nb_actions
        pi = (1-epsilon) * pi_b + epsilon * pi_r        
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