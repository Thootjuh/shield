from collections import defaultdict
import math

class PACIntervalEstimatorAbs:
    def __init__(self, structure, error_tolerance, trajectories, nb_actions, precision=1e-8):
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
        # self.nb_states = min(100, len(structure))
        self.nb_states = len(structure)
        self.precision = precision
        self.structure = structure
        self.error_tolerance = error_tolerance
        self.trajectories = trajectories
        self._count()
        self._build_model()
        self.state_action_state_pairs = self.calculate_intervals()
    
    def _count(self):
        """
        Counts the state-action pairs and state-action-triplets and stores them.
        """

        batch_trajectory = [val for sublist in self.trajectories for val in sublist]

        self.count_state_action_state = defaultdict(int)
        self.count_state_action = defaultdict(int)
        for [action, state, next_state, _] in batch_trajectory:
            self.count_state_action_state[(int(state), action, int(next_state))] += 1
            self.count_state_action[(int(state), action)] += 1
    
    def _build_model(self):
        """
        Estimates the transition probabilities from the given data.
        """
        self.transition_model = {}

        for (s, a, s_prime), count in self.count_state_action_state.items():
            denom = self.count_state_action.get((s, a), 0)

            if denom == 0:
                continue  # Avoid division by zero; unseen (s,a) pairs are skipped

            prob = count / denom
            self.transition_model[(s, a, s_prime)] = prob
            
    def calculate_intervals(self):
        """
        Computes intervals for all possible (state, action, next_state) triplets defined in the structure.

        Returns
        -------
        dict
            A dictionary mapping (state, action, next_state) to a (lower_bound, upper_bound) interval tuple.
        """
        state_action_state_pairs = defaultdict(list)
        # for state in range(len(self.structure)):
        #     for action in range(len(self.structure[state])):
        #         for next_state in self.structure[state][action]:
        for (state, action, next_state) in self.transition_model.keys():
            triplet = [state, action, next_state]
            interval = self.get_transition_interval(triplet)
            state_action_state_pairs[(state, action, next_state)] = interval
        return state_action_state_pairs     
    
    
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
        state, action, next_state = triplet
        point = self.transition_model[(state, action, next_state)]
        confidence_interval = self.computePACBound(triplet)
        lower_bound = max(point - confidence_interval, self.precision)
        upper_bound = min(point + confidence_interval, 1-self.precision)
        return lower_bound, upper_bound
    
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
        s = self.nb_states
        count = self.count_state_action[(state, action)]+5
        if count == 0:
            count = 1
        bound = math.sqrt((8/count)*(s*math.log(2)+math.log(1/self.error_tolerance)))
        bound = math.sqrt((8/count)*math.log((2*s*self.nb_actions*2^s)/self.error_tolerance))
        # n  = self.state_action_counts[(state, action)] + self.alpha * self.nb_actions #Grows with number of observations
        # alph = (self.error_tolerance*(1/m))/self.nb_actions
        # delta = math.sqrt(math.log(2/alph)/(2*n ))
        return bound
    
    def get_intervals(self):
        """
        Returns all computed PAC intervals for (state, action, next_state) triplets.

        Returns
        -------
        dict
            A dictionary of all PAC intervals.
        """
        return self.state_action_state_pairs