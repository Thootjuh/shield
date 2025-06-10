from batch_rl_algorithms.shielded.shielded_batch_rl_algorithm import shieldedBatchRLAlgorithm
import numpy as np
class shieldedBaseline(shieldedBatchRLAlgorithm):
    NAME = 'shielded_baseline'
    def __init__(self, pi_b, gamma, nb_states, nb_actions, data, R, episodic, shield, zero_unseen=True, max_nb_it=100,
                 checks=False, speed_up_dict=None, estimate_baseline=False, shield_baseline=False, shield_data=False):
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
        algorithms, this should only be used for big experiments; for the standard algorithms this should only contain
        the following:
        'count_state_action': numpy array with shape (nb_states, nb_actions) indicating the number of times a s
        tate-action pair has been visited
        'count_state_action_state': numpy array with shape (nb_states, nb_actions, nb_states) indicating the number of
        times a state-action-next-state triplet has been visited 
        """
        self.shield = shield
        self.pi_b = pi_b
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.pi = self.pi_b.copy()




        if estimate_baseline:
            self.pi_b = self.estimate_baseline()
            # self.pi_b = self.modifyPolicyWithShield(pi_b.copy())
            self.pi = self.pi_b.copy()
            

                   
    def modifyPolicyWithShield(self, policy):
        for i, state in enumerate(policy):
            allowed_actions = self.shield.get_safe_actions_from_shield(i)
            temp = np.zeros(len(state))
            for action in allowed_actions:
                temp[action] = state[action]
            total_mass = np.sum(temp)
            for j in range(len(temp)):
                policy[i][j] = temp[j]/total_mass
        return policy 
    
    def fit(self):
        """
        Starts the actual training by reiterating between self._policy_evaluation() and self._policy_improvement()
        until convergence of the action-value function or the maximal number of iterations (self.max_nb_it) is reached.
        :return:
        """
        self.pi = self.modifyPolicyWithShield(self.pi.copy())