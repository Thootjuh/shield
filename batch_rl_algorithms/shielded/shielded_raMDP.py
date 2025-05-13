import numpy as np

from batch_rl_algorithms.shielded.shielded_batch_rl_algorithm import shieldedBatchRLAlgorithm


class Shield_RaMDP(shieldedBatchRLAlgorithm):
    # Algorithm from 'Safe policy improvement by minimizing robust baseline regret' by Marek Petrik, Yinlam Chow and
    # Mohammad Ghavamzadeh, which is also equivalent to MBIE-EB from 'An analysis of model-based Interval
    # Estimation for Markov Decision Processes' by Alexander L. Strehl and Michael L. Littman
    NAME = 'shield-RaMDP'

    def __init__(self, pi_b, gamma, nb_states, nb_actions, data, R, episodic, shield, kappa, zero_unseen=True, max_nb_it=5000,
                 checks=False, speed_up_dict=None, estimate_baseline=False):
        '''
        :param pi_b: numpy matrix with shape (nb_states, nb_actions), such that pi_b(s,a) refers to the probability of
        choosing action a in state s by the behavior policy
        :param gamma: discount factor
        :param nb_states: number of states of the MDP
        :param nb_actions: number of actions available in each state
        :param data: the data collected by the behavior policy, which should be a list of [state, action, next_state,
         reward] sublists
        :param R: reward matrix with shape (nb_states, nb_states), assuming that the reward is deterministic w.r.t. the
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
        :param kappa: hyper-parameter of RaMDP
        '''
        self.kappa = kappa
        if isinstance(R, dict):
            self.r_min = min(R.values())
        else:
            self.r_min = np.min(R)
        super().__init__(pi_b, gamma, nb_states, nb_actions, data, R, episodic, shield, zero_unseen, max_nb_it, checks,
                         speed_up_dict=speed_up_dict, estimate_baseline=estimate_baseline)

        # print("raMDP")
        # print(self.allowed.shape)
        
        

    def _compute_R_state_action(self):
        '''
        Applies the estimated transition probabilities and the reward matrix with shape (nb_states, nb_states) to
        estimate a new reward matrix in the shape (nb_states, nb_actions) such that self.R_state_action[s, a] is the
        expected reward when choosing action a in state s in the estimated MDP. Additionally, it adjusts the reward
        matrix by the rules of RaMDP. It also applies a maximum negative reward to the actions that are not allowed by the shield
        :return:
        '''
        super()._compute_R_state_action()
    def _compute_R_state_action(self):
        '''
        Applies the estimated transition probabilities and the reward matrix with shape (nb_states, nb_states) to
        estimate a new reward matrix in the shape (nb_states, nb_actions) such that self.R_state_action[s, a] is the
        expected reward when choosing action a in state s in the estimated MDP. Additionally, it adjusts the reward
        matrix by the rules of RaMDP.
        :return:
        '''
        super()._compute_R_state_action()
        # Apply penalty only to known (s, a) pairs
        for (s, a), count in self.count_state_action.items():
            if count > 0:
                self.R_state_action[s, a] -= self.kappa / np.sqrt(count)

        # Set minimum reward for unseen (s, a) pairs
        min_reward = np.min(list(self.R_state_state.values())) * (1 / (1 - self.gamma))
        for s in range(self.nb_states):
            for a in range(self.nb_actions):
                if (s, a) not in self.count_state_action:
                    self.R_state_action[s, a] = min_reward
        self.R_state_action[~self.allowed] = self.r_min * (1 / (1 - self.gamma))
        # print("in shielded")
        # print(self.R_state_action)
       
    def _policy_improvement(self):
        """
        Updates the current policy self.pi (Here: greedy update).
        """
        
        self.q_shield = self.q.copy()
        self.q_shield[~self.allowed] = -np.inf
        self.pi = np.zeros([self.nb_states, self.nb_actions])
        for s in range(self.nb_states):    
            self.pi[s, np.argmax(self.q_shield[s, :])] = 1
        # print(self.pi)
            
             
