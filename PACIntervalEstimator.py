import numpy as np
from collections import defaultdict
import math

class PACIntervalEstimator:
    def __init__(self, structure, error_tolerance, trajectories, nb_actions, alpha=2, precision=1e-8):
        self.nb_actions = nb_actions
        self.alpha = alpha
        self.precision = precision
        self.structure = structure
        self.error_tolerance = error_tolerance
        self.trajectories = trajectories
        self.count()
        self.state_action_state_pairs = self.calculate_intervals()
    # This function goes over the structure and looks at all possible state-action-state triples 
    def calculate_intervals(self):
        state_action_state_pairs = defaultdict(list) # np.zeros((self.structure.size, 3))
        for state in range(len(self.structure)):
            for action in range(len(self.structure[state])):
                for next_state in self.structure[state][action]:
                    triplet = [state, action, next_state]
                    interval = self.get_transition_interval(triplet)
                    state_action_state_pairs[(state, action, next_state)] = interval
        return state_action_state_pairs                    
    
    def count(self):
        state_action_counts = defaultdict(int)
        state_action_next_state_counts = defaultdict(int)

        for trajectory in self.trajectories:
            for a, s, ns, _ in trajectory:
                state_action_counts[(s, a)] += 1
                state_action_next_state_counts[(s, a, ns)] += 1
        
        self.state_action_counts = state_action_counts
        self.state_action_state_counts = state_action_next_state_counts
        
    # This function calculates the interval around the mode
    def get_transition_interval(self, triplet):
        point = self.mode(triplet)
        confidence_interval = self.computePACBound(triplet)
        lower_bound = max(point - confidence_interval, self.precision)
        upper_bound = min(point + confidence_interval, 1-self.precision)
        # print(f"({lower_bound},{upper_bound})")
        # print("the size of the interval is ", upper_bound-lower_bound)
        return lower_bound, upper_bound
    
    def mode(self, triplet):
        state, action, next_state = triplet
        num = self.state_action_state_counts[(state, action, next_state)] + self.alpha - 1
        denum = 0
        count = 0
        successors = self.structure[state][action]
        for successor in successors:
            alph = self.state_action_state_counts[(state, action, successor)]+self.alpha
            denum += alph
            count += 1
        denum -= count
        return num/denum
    

    # This function calculates the pac bound
    def computePACBound(self, triplet):
        state, action, next_state = triplet
        m = self.structure.size 
        n  = self.state_action_counts[(state, action)] + self.alpha * self.nb_actions #Grows with number of observations
        alph = (self.error_tolerance*(1/m))/self.nb_actions
        delta = math.sqrt(math.log(2/alph)/(2*n ))
        return delta
    
    def get_intervals(self):
        return self.state_action_state_pairs
    
    def get_specific_interval(self, state, action, next_state):
        return self.state_action_state_pairs[(state, action, next_state)]
        