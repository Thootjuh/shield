import numpy as np
from collections import defaultdict
import math

class PACIntervalEstimator:
    def __init__(self, structure, error_tolerance, trajectories, nb_actions, alpha=2, precision=1e-8):
        """
        Initializes the PACIntervalEstimator.

        Parameters
        ----------
        structure : np.ndarray
            Transition structure where structure[s][a] is a list of possible next states from state s with action a.
        error_tolerance : float
            Desired error tolerance for PAC bounds.
        trajectories : list[list[tuple]]
            Observed transitions. Each trajectory is a list of (action, state, next_state, reward) tuples.
        nb_actions : int
            The number of actions available per state.
        alpha : float, optional
            Smoothing parameter for Bayesian estimation, default is 2.
        precision : float, optional
            Minimum and maximum bound for interval values, default is 1e-8.
        """
        self.nb_actions = nb_actions
        self.alpha = alpha
        self.precision = precision
        self.structure = structure
        self.error_tolerance = error_tolerance
        self.trajectories = trajectories
        self.count()
        self.state_action_state_pairs = self.calculate_intervals()
    
    def calculate_intervals(self):
        """
        Computes intervals for all possible (state, action, next_state) triplets defined in the structure.

        Returns
        -------
        dict
            A dictionary mapping (state, action, next_state) to a (lower_bound, upper_bound) interval tuple.
        """
        state_action_state_pairs = defaultdict(list) # np.zeros((self.structure.size, 3))
        for state in range(len(self.structure)):
            for action in range(len(self.structure[state])):
                for next_state in self.structure[state][action]:
                    triplet = [state, action, next_state]
                    interval = self.get_transition_interval(triplet)
                    state_action_state_pairs[(state, action, next_state)] = interval
        return state_action_state_pairs
    
    def count(self):
        """
        Counts the occurrences of (state, action) and (state, action, next_state) pairs in the given trajectories.

        Updates
        -------
        self.state_action_counts : dict
            Count of (state, action) pairs.
        self.state_action_state_counts : dict
            Count of (state, action, next_state) triplets.
        """
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
        """
        Computes a PAC confidence interval around the mode for a given (state, action, next_state) triplet.

        Parameters
        ----------
        triplet : list[int]
            A list representing the [state, action, next_state] triplet.

        Returns
        -------
        tuple[float, float]
            A tuple containing the lower and upper bounds of the estimated interval.
        """
        point = self.mode(triplet)
        confidence_interval = self.computePACBound(triplet)
        lower_bound = max(point - confidence_interval, self.precision)
        upper_bound = min(point + confidence_interval, 1-self.precision)
        return lower_bound, upper_bound
    
    def mode(self, triplet):
        """
        Computes the mode (point estimate) of the transition probability for the given triplet
        using Bayesian smoothing.

        Parameters
        ----------
        triplet : list[int]
            A list representing the [state, action, next_state] triplet.

        Returns
        -------
        float
            The estimated mode of the transition probability.
        """
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
        """
        Computes the PAC-bound (confidence radius) for a transition probability estimate.

        Parameters
        ----------
        triplet : list[int]
            A list representing the [state, action, next_state] triplet.

        Returns
        -------
        float
            The PAC confidence bound for the transition probability.
        """
        state, action, next_state = triplet
        m = self.structure.size 
        n  = self.state_action_counts[(state, action)] + self.alpha * self.nb_actions #Grows with number of observations
        alph = (self.error_tolerance*(1/m))/self.nb_actions
        delta = math.sqrt(math.log(2/alph)/(2*n ))
        return delta
    
    def get_intervals(self):
        """
        Returns all computed PAC intervals for (state, action, next_state) triplets.

        Returns
        -------
        dict
            A dictionary of all PAC intervals.
        """
        return self.state_action_state_pairs
    
    def get_specific_interval(self, state, action, next_state):
        """
        Retrieves the specific PAC interval for a given (state, action, next_state) triplet.

        Parameters
        ----------
        state : int
            The source state.
        action : int
            The action taken from the source state.
        next_state : int
            The resulting state after taking the action.

        Returns
        -------
        tuple[float, float]
            The PAC interval for the specified triplet.
        """
        return self.state_action_state_pairs[(state, action, next_state)]
        