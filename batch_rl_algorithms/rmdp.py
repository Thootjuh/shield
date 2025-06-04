import numpy as np
from batch_rl_algorithms.batch_rl_algorithm import BatchRLAlgorithm
from collections import defaultdict
from scipy.sparse import dok_matrix, eye
from scipy.sparse.linalg import spsolve
class WorstCaseRMDP(BatchRLAlgorithm):
    NAME = 'WorstCaseRMDP'
    def __init__(self, pi_b, gamma, nb_states, nb_actions, data, R, episodic, intervals,
                 zero_unseen=True, max_nb_it=500, checks=False, speed_up_dict=None, estimate_baseline=False):
        """
        :param transition_intervals: Dictionary mapping (s, a) pairs to a list of (s', [P_min, P_max]).
        """
        self.intervals = intervals
        super().__init__(pi_b, gamma, nb_states, nb_actions, data, R, episodic, zero_unseen, max_nb_it, checks, speed_up_dict, estimate_baseline)
        print("starting RMDP")
        
    def build_worst_case_model(self):
        """
        Computes the worst-case action-value function for the current policy using a sparse transition model.
        """
        V_pessimistic = np.max(self.q, axis=1)

        # Estimate value for each (s, s') using sparse rewards
        next_state_values = {
            s_prime: self.R_state_state.get((0, s_prime), 0.0) + self.gamma * V_pessimistic[s_prime]
            for s_prime in range(self.nb_states)
        }

        # Copy transition model
        P_pessimistic = self.transition_model.copy()

        for s in range(self.nb_states):
            for a in range(self.nb_actions):
                # Collect next states with non-zero transition probability
                possible_next_states = [
                    s_prime for (s2, a2, s_prime) in self.transition_model
                    if s2 == s and a2 == a
                ]

                if len(possible_next_states) <= 1:
                    continue

                # Worst next state = lowest value
                worst_next_state = min(possible_next_states, key=lambda s_: next_state_values[s_])
                remaining_next_states = [s_ for s_ in possible_next_states if s_ != worst_next_state]

                # Interval bounds for worst case transition
                bounds = self.intervals.get((s, a, worst_next_state), (0.0, 1.0))
                key_worst = (s, a, worst_next_state)
                original_prob = P_pessimistic.get(key_worst, 0.0)
                mass_added = bounds[1] - original_prob
                P_pessimistic[key_worst] = original_prob + mass_added

                # Reduce probability from better states
                mass_subtracted = 0.0
                remaining_next_states = sorted(remaining_next_states, key=lambda s_: next_state_values[s_], reverse=True)

                for best_next_state in remaining_next_states:
                    if mass_subtracted >= mass_added:
                        break
                    key_best = (s, a, best_next_state)
                    prob = P_pessimistic.get(key_best, 0.0)
                    bounds = self.intervals.get(key_best, (0.0, 1.0))
                    min_bound = bounds[0]
                    mass_moved = min(mass_added - mass_subtracted, prob - min_bound)
                    P_pessimistic[key_best] = prob - mass_moved
                    mass_subtracted += mass_moved

        self.P_pessimistic = P_pessimistic
    
    def _compute_R_state_action(self):
        """
        Applies the estimated transition probabilities and the reward matrix with shape (nb_states, nb_states) to
        estimate a new reward matrix in the shape (nb_states, nb_actions) such that self.R_state_action[s, a] is the
        expected reward when choosing action a in state s in the estimated MDP.
        """
        self.build_worst_case_model()
        result = defaultdict(float)

        for (i, j, k), p_val in self.P_pessimistic.items():
            r_val = self.R_state_state.get((i, k), 0.0)
            result[(i, j)] += p_val * r_val

        # Convert result to dense NumPy array

        self.R_state_action = np.zeros((self.nb_states, self.nb_actions))

        for (i, j), val in result.items():
            self.R_state_action[i, j] = val
        # print(f"estimated R_state_action = {self.R_state_action}")      
    
    def _policy_evaluation(self):
        """
        Computes the action-value function for the current policy self.pi using a sparse P_pessimistic.
        """
        nb_sa = self.nb_states * self.nb_actions
        old_q = self.q.copy()
        started = True
        nb_it = 0

        while started or np.linalg.norm(self.q - old_q) > 1e-3 and nb_it < (self.max_nb_it / 100):
            started = False
            nb_it += 1

            self._compute_R_state_action()  # Also builds P_pessimistic

            # Build sparse M matrix: (I - γ * Pπ)
            M = dok_matrix((nb_sa, nb_sa))
            for s in range(self.nb_states):
                for a in range(self.nb_actions):
                    sa_index = s * self.nb_actions + a
                    M[sa_index, sa_index] = 1.0  # Identity part

                    for s_prime in range(self.nb_states):
                        for a_prime in range(self.nb_actions):
                            pi_val = self.pi[s_prime, a_prime]
                            prob = self.P_pessimistic.get((s, a, s_prime), 0.0)
                            if prob != 0:
                                sa_prime_index = s_prime * self.nb_actions + a_prime
                                M[sa_index, sa_prime_index] -= self.gamma * prob * pi_val

            # Solve linear system
            b = self.R_state_action.reshape(nb_sa)
            M_csr = M.tocsr()
            q_flat = spsolve(M_csr, b)
            self.q = q_flat.reshape(self.nb_states, self.nb_actions)

