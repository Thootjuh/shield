import stormpy
import numpy as np
from collections import defaultdict
from dtmc_builder import dtmcBuilderWetChicken, dtmcBuilderRandomMDPs, dtmcBuilderFrozenLake

class evaluator:
    def __init__(self, P, pi, prop, n_states, n_actions, init, goal, traps, env_name):
        self.init = init
        self.n_states = n_states
        self.n_actions = n_actions
        self.prop = prop
        if isinstance(P, dict):
            self.transition_dynamics = P
        else:
            self.transition_dynamics = self.dense_to_sparse_transitions(P)
        self.policy = pi
        self.env_name = env_name
        self.goal=goal 
        self.traps = traps
        self.reshape_P()
     
    def dense_to_sparse_transitions(self, P):
        """
        Convert a dense transition array P[state, action, next_state]
        into a sparse dictionary:
            {(s, a, s_next): prob}
        Only non-zero probabilities are kept.
        """
        sparse = {}

        n_states, n_actions, n_next = P.shape

        for s in range(n_states):
            for a in range(n_actions):
                for s_next in range(n_next):
                    prob = P[s, a, s_next]
                    if prob != 0.0:   # skip zeros
                        sparse[(s, a, s_next)] = prob

        return sparse  
       
    def reshape_P(self):
        self.transitions_reshaped = defaultdict(list)
        for (s, a, next_s), p in self.transition_dynamics.items():
            self.transitions_reshaped[(s, a)].append((next_s, p)) 
    
    def construct_DTMC(self):
        if self.env_name == "wet_chicken":
            self.model = dtmcBuilderWetChicken(self.transition_dynamics, self.policy, self.n_states, self.n_actions, self.init, self.goal, self.traps).build_model()
        if self.env_name == "random_mdps":
            self.model = dtmcBuilderRandomMDPs(self.transition_dynamics, self.policy, self.n_states, self.n_actions, self.init, self.goal, self.traps).build_model()
        if self.env_name == "frozen_lake":
            self.model = dtmcBuilderFrozenLake(self.transition_dynamics, self.policy, self.n_states, self.n_actions, self.init, self.goal, self.traps).build_model()
    def invoke_storm(self):
        # invoke storm to check the probability of satisfying the prob from the start state
        # properties = stormpy.parse_properties(self.prop, self.model)
        
        properties = stormpy.parse_properties(self.prop)
        result = stormpy.model_checking(self.model, properties[0])
        return result

    def find_success_prob(self):
        # Use the DTMC to find the probability of succeeding
        result = self.invoke_storm()
        return result.get_values()[self.init]