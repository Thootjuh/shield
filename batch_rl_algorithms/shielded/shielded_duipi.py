import numpy as np

from batch_rl_algorithms.shielded.shielded_batch_rl_algorithm import shieldedBatchRLAlgorithm 


class shield_DUIPI(shieldedBatchRLAlgorithm):
    # Algorithm implemented following 'Uncertainty Propagation for Efficient Exploration in Reinforcement Learning'
    # by Alexander Hans and Steffen Udluft; a small modification has been added see the Master's thesis
    # 'Evaluation of Safe Policy Improvement with Soft Baseline Bootstrapping'
    NAME = 'shield-DUIPI'

    def __init__(self, pi_b, gamma, nb_states, nb_actions, data, R, episodic, shield, bayesian, xi, alpha_prior=0.1,
                 zero_unseen=True, max_nb_it=100, checks=False, speed_up_dict=None, estimate_baseline = True):
        """
        :param pi_b: numpy matrix with shape (nb_states, nb_actions), such that pi_b(s,a) refers to the probability of
        choosing action a in state s by the behavior policy
        :param gamma: discount factor
        :param nb_states: number of states of the MDP
        :param nb_actions: number of actions available in each state
        :param data: the data collected by the behavior policy, which should be a list of [state, action, next_state,
         reward] sublists
        :param R: reward matrix as numpy array with shape (nb_states, nb_states), assuming that the reward is deterministic w.r.t. the
         previous and the next states
        :param episodic: boolean variable, indicating whether the MDP is episodic (True) or non-episodic (False)
        :param zero_unseen: boolean variable, indicating whether the estimated model should guess set all transition
        probabilities to zero for a state-action pair which has never been visited (True) or to 1/nb_states (False)
        :param max_nb_it: integer, indicating the maximal number of times the PE and PI step should be executed, if
        convergence is not reached
        :param checks: boolean variable indicating if different validity checks should be executed (True) or not
        (False); this should be set to True for development reasons, but it is time consuming for big experiments
        :param speed_up_dict: a dictionary containing pre-calculated quantities which can be reused by many different
        algorithms, this should only be used for big experiments; for DUIPI this should only contain
        the following:
        'count_state_action': numpy array with shape (nb_states, nb_actions) indicating the number of times a s
        tate-action pair has been visited
        'count_state_action_state': numpy array with shape (nb_states, nb_actions, nb_states) indicating the number of
        times a state-action-next-state triplet has been visited
        :param bayesian: boolean variable, indicating whether the estimation of the variance of the estimation of the
        transition probabilities should be done bayesian (True) using the Dirichlet distribution as a prior or
        frequentistic (False)
        :param xi: hyper-parameter of DUIPI, the higher xi is, the stronger is the influence of the variance
        :param alpha_prior: float variable necessary if bayesian=True, usually between 0 and 1
        """
        self.xi = xi
        self.alpha_prior = alpha_prior
        self.bayesian = bayesian
        super().__init__(pi_b, gamma, nb_states, nb_actions, data, R, episodic, shield, zero_unseen, max_nb_it, checks,
                         speed_up_dict, estimate_baseline)
        self.variance_q = np.zeros([self.nb_states, self.nb_actions])
        self.pi = 1 / self.nb_actions * np.ones([self.nb_states, self.nb_actions])
        
        self.shield_actions()
        self.mask = self.mask & self.allowed

    def _initial_calculations(self):
        """
        Starts all the calculations which can be done before the actual training.
        """
        self._prepare_R_and_variance_R()
        self._prepare_P_and_variance_P()
        self._compute_mask()


    def _prepare_R_and_variance_R(self):
        """
        Estimates the reward matrix and its variance from a sparse R_state_state dictionary:
        {(s, s'): reward}.
        Broadcasts the reward across all actions for a given state-next_state pair.
        """
        self.R_state_action_state = np.zeros((self.nb_states, self.nb_actions, self.nb_states))

        for (s, s_prime), reward in self.R_state_state.items():
            self.R_state_action_state[s, :, s_prime] = reward  # Assign reward for all actions at (s, s')

        self.variance_R = np.zeros((self.nb_states, self.nb_actions, self.nb_states))

    def _prepare_P_and_variance_P(self):
        """
        Estimates the transition model and its variance using sparse count and transition dictionaries.
        Supports Bayesian and frequentist cases.
        """
        self.variance_P = np.zeros((self.nb_states, self.nb_actions, self.nb_states))

        if self.bayesian:
            # Bayesian update from sparse count dictionary
            transition_model = {}
            count_sums = {}  # Holds alpha_d_0 = sum over s' for each (s, a)
            
            # First pass: compute alpha_d and alpha_d_0
            for (s, a, s_prime), count in self.count_state_action_state.items():
                alpha_d = count + self.alpha_prior
                transition_model[(s, a, s_prime)] = alpha_d
                count_sums[(s, a)] = count_sums.get((s, a), 0) + alpha_d

            # Normalize to get probabilities and compute variance
            self.transition_model = {}
            for (s, a, s_prime), alpha_d in transition_model.items():
                alpha_d_0 = count_sums[(s, a)]
                prob = alpha_d / alpha_d_0
                self.transition_model[(s, a, s_prime)] = prob

                var = (alpha_d * (alpha_d_0 - alpha_d)) / (alpha_d_0**2 * (alpha_d_0 + 1))
                self.variance_P[s, a, s_prime] = var

        else:
            # Frequentist case
            self._build_model()  # Must build self.transition_model from sparse data
            for (s, a, s_prime), prob in self.transition_model.items():
                count = self.count_state_action.get((s, a), 0)
                if count > 1:
                    var = prob * (1 - prob) / (count - 1)
                else:
                    var = 1 / 4  # max variance for binary distribution with no/1 sample
                self.variance_P[s, a, s_prime] = var

            # Default variance for unvisited state-action pairs
            for s in range(self.nb_states):
                for a in range(self.nb_actions):
                    if (s, a) not in self.count_state_action:
                        self.variance_P[s, a, :] = 1 / 4

        self._check_if_valid_transitions()

    def _compute_mask(self):
        """
        Compute the mask which indicates which state-pair has never been visited.
        """
        # self.mask = self.count_state_action > 0
        self.mask = np.full((self.nb_states, self.nb_actions), False, dtype=bool)
        for (state, action), value in self.count_state_action.items():
            if value > 0:
                self.mask[state,action] = True

    def _policy_evaluation(self):
        """
        Evaluates the current policy self.pi and calculates its variance, using a sparse transition model.
        """
        # Value function: v(s) = sum_a pi(s,a) * q(s,a)
        self.v = np.einsum('ij,ij->i', self.pi, self.q)

        # Variance of v(s): weighted sum of variance_q(s,a)
        self.variance_v = np.einsum('ij,ij->i', self.pi ** 2, self.variance_q)

        # Initialize q and variance_q to zero
        self.q = np.zeros((self.nb_states, self.nb_actions))
        self.variance_q = np.zeros((self.nb_states, self.nb_actions))

        for (s, a, s_prime), prob in self.transition_model.items():
            r = self.R_state_action_state[s, a, s_prime]
            v_sp = self.v[s_prime]
            self.q[s, a] += prob * (r + self.gamma * v_sp)

            # Variance terms
            r_term = r + self.gamma * v_sp
            var1 = (self.gamma ** 2) * (prob ** 2) * self.variance_v[s_prime]
            var2 = (r_term ** 2) * self.variance_P[s, a, s_prime]
            var3 = (prob ** 2) * self.variance_R[s, a, s_prime]

            self.variance_q[s, a] += var1 + var2 + var3

        # Replace any NaNs or infs due to numerical issues
        self.variance_q = np.nan_to_num(self.variance_q, nan=np.inf, posinf=np.inf)

 
    def _policy_improvement(self):
        """
        Updates the current policy self.pi.
        """
        q_uncertainty_and_mask_corrected = self.q - self.xi * np.sqrt(self.variance_q)
        # The extra modification to avoid unobserved state-action pairs
        q_uncertainty_and_mask_corrected[~self.mask] = - np.inf

        best_action = np.argmax(q_uncertainty_and_mask_corrected, axis=1)
        for state in range(self.nb_states):
            d_s = np.minimum(1 / self.nb_it, 1 - self.pi[state, best_action[state]])
            self.pi[state, best_action[state]] += d_s
            for action in range(self.nb_actions):
                if action == best_action[state]:
                    continue
                elif self.pi[state, best_action[state]] == 1:
                    self.pi[state, action] = 0
                else:
                    self.pi[state, action] = self.pi[state, action] * (1 - self.pi[state, best_action[state]]) / (
                            1 - self.pi[state, best_action[state]] + d_s)
