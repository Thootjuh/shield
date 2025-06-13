import numpy as np 
from collections import defaultdict
from scipy.sparse import eye, lil_matrix
from scipy.sparse.linalg import spsolve

class BatchRLAlgorithm:
    # Base class for all batch RL algorithms, which implements the general framework and the PE and PI step for
    # Dynamic Programming following 'Reinforcement Learning - An Introduction' by Sutton and Barto and
    # https://github.com/RomainLaroche/SPIBB. Additionally, it also implements the estimations of the transition
    # probabilities and reward matrix and some validation checks.
    def __init__(self, pi_b, gamma, nb_states, nb_actions, data, R, episodic, zero_unseen=True, max_nb_it=100,
                 checks=False, speed_up_dict=None, estimate_baseline = False):
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
        self.pi_b = pi_b
        self.pi = self.pi_b.copy()
        self.gamma = gamma
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.data = data
        self.zero_unseen = zero_unseen
        self.episodic = episodic
        self.max_nb_it = max_nb_it
        
        self.q = np.zeros([nb_states, nb_actions])
        if isinstance(R, dict):
            self.R_state_state = R
        else:
            self.R_state_state = self.reward_function_to_dict(R)
        self.checks = checks
        self.speed_up_dict = speed_up_dict
        if self.speed_up_dict:
            self.count_state_action = self.speed_up_dict['count_state_action']
            self.count_state_action_state = self.speed_up_dict['count_state_action_state']
        else:
            self._count()
        if estimate_baseline:
            self.pi_b = self.estimate_baseline()
            self.pi = self.pi_b.copy()
        self._initial_calculations()

            
    def array_to_dict(self, arr):
        sparse_dict = {}

        # Iterate over the 3D array and add non-zero elements to the dictionary
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    if arr[i, j, k] != 0:
                        sparse_dict[(i, j, k)] = arr[i, j, k]
        return sparse_dict
    
    def reward_function_to_dict(self, arr):
        sparse_dict = {}

        # Iterate over the 3D array and add non-zero elements to the dictionary
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if arr[i, j] != 0:
                    sparse_dict[(i, j)] = arr[i, j]
        return sparse_dict
           
    def estimate_baseline(self):
        result = np.zeros((self.nb_states, self.nb_actions), dtype=float)

        # First, compute n(s): total visits per state
        n_s = defaultdict(int)
        for (s, a), count in self.count_state_action.items():
            n_s[s] += count

        # Now compute the baseline probabilities
        for s in range(self.nb_states):
            if n_s[s] == 0:
                # If the state has never been seen, assign uniform policy
                result[s] = 1.0 / self.nb_actions
            else:
                for a in range(self.nb_actions):
                    count = self.count_state_action.get((s, a), 0)
                    result[s, a] = count / n_s[s]

        return result            

    def _initial_calculations(self):
        """
        Starts all the calculations which can be done before the actual training.
        """
        self._build_model()
        self._compute_R_state_action()

    def fit(self):
        """
        Starts the actual training by reiterating between self._policy_evaluation() and self._policy_improvement()
        until convergence of the action-value function or the maximal number of iterations (self.max_nb_it) is reached.
        :return:
        """
        if self.checks:
            self._check_if_valid_transitions()
        old_q = np.ones([self.nb_states, self.nb_actions])
        self.nb_it = 0

        while np.linalg.norm(self.q - old_q) > 10 ** (-3) and self.nb_it < self.max_nb_it:
            # print(self.nb_it)
            self.nb_it += 1
            old_q = self.q.copy()
            self._policy_evaluation()
            # q_func = self._policy_evaluation_old()
            self._policy_improvement()
            # for i in range(len(q_func)):
            #     for j in range(len(q_func[i])):
            #         if q_func[i,j] != self.q[i,j]:
            #             print(q_func[i,j], " does not equal ", self.q[i,j])
            # print("next_it")
            if self.checks:
                self._check_if_valid_policy()            
            
            
        if self.nb_it > self.max_nb_it:
            with open("notconverging.txt", "a") as myfile:
                myfile.write(f"{self.NAME} is not converging. \n")

    def _count(self):
        """
        Counts the state-action pairs and state-action-triplets and stores them.
        """
        if self.episodic:
            batch_trajectory = [val for sublist in self.data for val in sublist]
        else:
            batch_trajectory = self.data.copy()
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
        

    def _compute_R_state_action(self):
        """
        Applies the estimated transition probabilities and the reward matrix with shape (nb_states, nb_states) to
        estimate a new reward matrix in the shape (nb_states, nb_actions) such that self.R_state_action[s, a] is the
        expected reward when choosing action a in state s in the estimated MDP.
        """
        result = defaultdict(float)

        for (i, j, k), p_val in self.transition_model.items():
            r_val = self.R_state_state.get((i, k), 0.0)
            result[(i, j)] += p_val * r_val

        # Convert result to dense NumPy array

        self.R_state_action = np.zeros((self.nb_states, self.nb_actions))

        for (i, j), val in result.items():
            self.R_state_action[i, j] = val

        return self.R_state_action
    

    def _policy_improvement(self):
        """
        Updates the current policy self.pi (Here: greedy update).
        """
        self.pi = np.zeros([self.nb_states, self.nb_actions])
        for s in range(self.nb_states):
            self.pi[s, np.argmax(self.q[s, :])] = 1
            
    def _policy_evaluation_old(self):
        """
        Computes the action-value function for the current policy self.pi.
        """
        nb_sa = self.nb_actions * self.nb_states
        M = np.eye(nb_sa) - self.gamma * np.einsum('ijk,kl->ijkl', self.transition_model_old, self.pi).reshape(nb_sa, nb_sa)
        q_func = np.linalg.solve(M, self.R_state_action.reshape(nb_sa)).reshape(self.nb_states, self.nb_actions)
        return q_func
    
    def _policy_evaluation(self):
        """
        Computes the action-value function for the current policy self.pi.
        """
        # nb_sa = self.nb_actions * self.nb_states
        # M = np.eye(nb_sa) - self.gamma * np.einsum('ijk,kl->ijkl', self.transition_model_old, self.pi).reshape(nb_sa, nb_sa)
        # self.q = np.linalg.solve(M, self.R_state_action.reshape(nb_sa)).reshape(self.nb_states, self.nb_actions)
        nb_sa = self.nb_actions * self.nb_states

        # Create identity matrix I
        M = eye(nb_sa, format="lil")  # LIL format allows efficient element-wise updates

        # Iterate over sparse transition_model
        for (i, j, k), value in self.transition_model.items():
            for l in range(self.nb_actions):  # Iterate over action indices
                M[i * self.nb_actions + j, k * self.nb_actions + l] -= self.gamma * value * self.pi[k, l]

        # Convert M to CSR for efficient solving
        M = M.tocsr()

        # Solve for q
        q_vector = spsolve(M, self.R_state_action.reshape(nb_sa))  # Solve Mq = R
        self.q = q_vector.reshape(self.nb_states, self.nb_actions)  # Reshape to desired format
        # print("YAHOO! ITSA ME MARIO ANDA ITSA MY BROTHA LUIGI! YAHOO!!")
        # print(self.q)

    def _check_if_valid_policy(self):
        checks = np.unique((np.sum(self.pi, axis=1)))
        valid = True
        for i in range(len(checks)):
            if np.abs(checks[i] - 0) > 10 ** (-6) and np.abs(checks[i] - 1) > 10 ** (-6):
                valid = False
        if not valid:
            print(f'!!! Policy not summing up to 1 !!!')

    def _check_if_valid_transitions(self):
        """
        Checks that for each (state, action) pair, the transition probabilities over next states sum to 1 or 0.
        """
        from collections import defaultdict

        sum_probs = defaultdict(float)
        for (s, a, s_prime), prob in self.transition_model.items():
            sum_probs[(s, a)] += prob

        valid = True
        for (s, a), total_prob in sum_probs.items():
            if not (np.isclose(total_prob, 1.0, atol=1e-8) or np.isclose(total_prob, 0.0, atol=1e-8)):
                print(f'Invalid transition sum for state {s}, action {a}: {total_prob}')
                valid = False

        if not valid:
            print('!!! Transitions not summing up to 0 or 1 !!!')


    def compute_safety(self):
        return {'Probability': None, 'lower_limit': None}

    @property
    def get_v(self):
        v = np.einsum('ij,ij->i', self.pi, self.q)
        return v

    # Get the advantage of the policy w.r.t. pi_b
    def get_advantage(self, state, q_pi_b_est):
        v_pi_b_est_state = q_pi_b_est[state] @ self.pi_b[state]
        advantage = (v_pi_b_est_state - q_pi_b_est[state]) @ self.pi[state]
        return advantage