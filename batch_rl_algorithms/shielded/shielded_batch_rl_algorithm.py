from batch_rl_algorithms.batch_rl_algorithm import BatchRLAlgorithm
import numpy as np
class shieldedBatchRLAlgorithm(BatchRLAlgorithm):
    def __init__(self, pi_b, gamma, nb_states, nb_actions, data, R, episodic, shield, zero_unseen=True, max_nb_it=5000,
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
        self.gamma = gamma
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.zero_unseen = zero_unseen
        self.episodic = episodic
        self.data = data
        if shield_data:
            self.data = self._modify_data()
        self.max_nb_it = max_nb_it
        self.pi = self.pi_b.copy()
        self.q = np.zeros([nb_states, nb_actions])
        if isinstance(R, dict):
            self.R_state_state = R
        else:
            self.R_state_state = self.reward_function_to_dict(R)
        self.checks = checks
        self.speed_up_dict = speed_up_dict
        self.shield_actions()
        
        if self.speed_up_dict:
            self.count_state_action = self.speed_up_dict['count_state_action']
            self.count_state_action_state = self.speed_up_dict['count_state_action_state']
        else:
            self._count()
        if estimate_baseline:
            self.pi_b = self.estimate_baseline()
            # self.pi_b = self.modifyPolicyWithShield(pi_b.copy())
            self.pi = self.pi_b.copy()
        if shield_baseline:
            self.pi_b = self.modifyPolicyWithShield(pi_b.copy())
            self.pi = self.pi_b.copy()
        self._initial_calculations()

    
    def _modify_data(self):
        safe_data = []
        if self.episodic:
            batch_trajectory = [val for sublist in self.data for val in sublist]
        else:
            batch_trajectory = self.data.copy()
        for [action, state, next_state, reward] in batch_trajectory:
            safe_actions = self.shield.get_safe_actions_from_shield(state)
            if action in safe_actions:
                safe_data.append([[action,state,next_state,reward]])
        return safe_data
    
    def shield_actions(self):
        self.allowed = np.full((self.nb_states, self.nb_actions), False, dtype=bool)
        for s in range(len(self.allowed)):
            allowed_actions = self.shield.get_safe_actions_from_shield(s)
            for a in allowed_actions:
                self.allowed[s][a]=True 
                   
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