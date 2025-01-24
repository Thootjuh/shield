import numpy as np
from batch_rl_algorithms.batch_rl_algorithm import BatchRLAlgorithm

class WorstCaseRMDP(BatchRLAlgorithm):
    NAME = 'WorstCaseRMDP'
    def __init__(self, pi_b, gamma, nb_states, nb_actions, data, R, episodic, intervals,
                 zero_unseen=True, max_nb_it=500, checks=False, speed_up_dict=None, estimate_baseline=False):
        """
        :param transition_intervals: Dictionary mapping (s, a) pairs to a list of (s', [P_min, P_max]).
        """
        self.intervals = intervals
        super().__init__(pi_b, gamma, nb_states, nb_actions, data, R, episodic, zero_unseen, max_nb_it, checks, speed_up_dict, estimate_baseline)
    
    def build_worst_case_model(self):
        """
        Computes the worst case action-value function for the current policy self.pi.
        """
        # q_pessimistic[self.mask_unseen] = 1 / (1 - self.gamma) * self.r_min
        V_pessimistic = np.max(self.q, axis=1)
        next_state_values = self.R_state_state[0] + self.gamma * V_pessimistic
        P_pessimistic = self.transition_model.copy()
        # Now find the worst P:
        for state in range(self.nb_states):
            for action in range(self.nb_actions):
                possible_next_states = np.nonzero(self.transition_model[state, action])[0]
                if len(possible_next_states) <= 1:
                    break
                # print("next states = ", possible_next_states)
                worst_next_state = min(possible_next_states, key=lambda i: next_state_values[i])
                remaining_next_states = possible_next_states[possible_next_states!=worst_next_state] # Delete the worst state from the remaining next states
                bounds = self.intervals[(state, action, worst_next_state)]
                mass_added = bounds[1] - P_pessimistic[state, action, int(worst_next_state)]
                P_pessimistic[state, action, worst_next_state] += mass_added #Maximize the probability of the transition to the worst possible next state
                mass_subtracted = 0
                
                remaining_next_states = sorted(remaining_next_states, key=lambda i: next_state_values[i], reverse=True)
                for best_next_state in remaining_next_states:
                    if mass_added == mass_subtracted:
                        break
                    # Note: It can happen that the best_next_state is the same as the worst_next_state, i.e. if all
                    # states are equally good, then this just removes the probability mass and we end up not
                    # changing P_pessimistic at all, which is fine if every state's value is the same
                    bounds = self.intervals[(state, action, best_next_state)]
                    
                    mass_moved = min(mass_added-mass_subtracted, P_pessimistic[state, action, best_next_state]-bounds[0])
                    # mass_in_move = np.min(
                    #     [mass_added - mass_subtracted, P_pessimistic[state, action, best_next_state]])
                    P_pessimistic[state, action, best_next_state] -= mass_moved
                    mass_subtracted += mass_moved
        self.P_pessimistic = P_pessimistic
    
    def _compute_R_state_action(self):
        """
        Applies the estimated transition probabilities and the reward matrix with shape (nb_states, nb_states) to
        estimate a new reward matrix in the shape (nb_states, nb_actions) such that self.R_state_action[s, a] is the
        expected reward when choosing action a in state s in the estimated MDP.
        """
        self.build_worst_case_model()
        self.R_state_action = np.einsum('ijk,ik->ij', self.P_pessimistic, self.R_state_state)
        # print(f"estimated R_state_action = {self.R_state_action}")      
    
    def _policy_evaluation(self):
        """
        Computes the action-value function for the current policy self.pi.
        """
        nb_sa = self.nb_actions * self.nb_states
        old_q = self.q.copy()
        started = True
        nb_it = 0
        while started or np.linalg.norm(self.q - old_q) > 10 ** (-3) and nb_it < (self.max_nb_it/100):
            # print(nb_it)
            started = False
            nb_it += 1
            self._compute_R_state_action() # Also builds P_worst_case
            M = np.eye(nb_sa) - self.gamma * np.einsum('ijk,kl->ijkl', self.P_pessimistic, self.pi).reshape(nb_sa, nb_sa)
            old_q = self.q.copy()
            self.q = np.linalg.solve(M, self.R_state_action.reshape(nb_sa)).reshape(self.nb_states, self.nb_actions)

