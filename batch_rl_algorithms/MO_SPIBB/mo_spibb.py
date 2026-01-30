"""
Contains the implementation of MO-SPIBB (S-OPT) in the draft
"""
import cvxpy as cp
import numpy as np
from copy import deepcopy
from collections import defaultdict
from .utils import direct_policy_evaluation, bounded_successive_approximation, default_termination
from .base_agent import Agent



class ConstSPIBBAgent():
    """
    The agent based on the S-OPT equation in the draft
    """
    
    NAME = "MO_SPIBB"
    def __init__(self,
                 termination_condition,
                 coeff_list,
                 pi_b,
                 estimate_baseline,
                 R_state_state,
                 C_state_state,
                 data,
                 gamma,
                 episodic,
                 nb_states,
                 nb_actions,
                 epsilon,
                 max_nb_it=100,
                 **kwargs):
        """

        :param termintation_condition: the function that decides when to stop the iterative procedure
        :param coeff_list: the list of lambda_parameters to try search over
        :param kwargs:
        """
        self.termination_condition = default_termination
        self._name = "S_OPT"
        self.params = deepcopy(kwargs)
        self.episodic = episodic
        self.gamma = gamma
        self.data = data
        self.C_state_state = C_state_state
        self.R_state_state = R_state_state
        self.estimate_baseline = estimate_baseline
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.epsilon = epsilon
        self.max_nb_it=max_nb_it
        if self.estimate_baseline:
            pass
        else:
            self.pi_b = pi_b
            
            
        self.P = self.estimate_model()
        self.error_function = self.compute_error_function()
        
        self._count()
        self._build_model()
        self.R_state_action = self._compute_R_state_action()
        self.C_state_action = self._compute_C_state_action()

        # Prepare coefficients list
        lambda_R_vals = [1.0]  # >=0
        lambda_C_vals = [1.0]  # >=0
        self.lambda_coeffs = [(lr, lc) for lc in lambda_C_vals for lr in lambda_R_vals][0]
        
        self.operator = self.make_policy_iteration_operator()
        
    def _compute_R_state_action(self):
        """
        Applies the estimated transition probabilities and the reward matrix with shape (nb_states, nb_states) to
        estimate a new reward matrix in the shape (nb_states, nb_actions) such that self.R_state_action[s, a] is the
        expected reward when choosing action a in state s in the estimated MDP.
        """
        result = defaultdict(float)

        for (i, j, k), p_val in self.transition_model.items():
            r_val = self.R_state_state[i, k]
            result[(i, j)] += p_val * r_val

        # Convert result to dense NumPy array

        self.R_state_action = np.zeros((self.nb_states, self.nb_actions))

        for (i, j), val in result.items():
            self.R_state_action[i, j] = val

        return self.R_state_action
    
    def _compute_C_state_action(self):
        """
        Applies the estimated transition probabilities and the reward matrix with shape (nb_states, nb_states) to
        estimate a new reward matrix in the shape (nb_states, nb_actions) such that self.R_state_action[s, a] is the
        expected reward when choosing action a in state s in the estimated MDP.
        """
        result = defaultdict(float)

        for (i, j, k), p_val in self.transition_model.items():
            r_val = self.C_state_state[i, k]
            result[(i, j)] += p_val * r_val

        # Convert result to dense NumPy array

        self.C_state_action = np.zeros((self.nb_states, self.nb_actions))

        for (i, j), val in result.items():
            self.C_state_action[i, j] = val

        return self.C_state_action
    
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
    
    def compute_error_function(self, delta=1.0):
        """
        Computes the e_Q function based on the dataset

        :param batch:
        :param nstates:
        :param nactions:
        :param delta:
        :return:
        """
        if self.episodic:
            batch = [val for sublist in self.data for val in sublist]
        else:
            batch = self.data.copy()
        count_sa = np.zeros((self.nb_states, self.nb_actions))
        eQ = np.zeros((self.nb_states, self.nb_actions))

        # for each transition in batch
        for transition in batch:
            state = transition[1]
            action = transition[0]
            next_state = transition[2]

            count_sa[state, action] += 1.0

        # for each (s,a)
        for s in range(self.nb_states):
            for a in range(self.nb_actions):
                if count_sa[s, a] == 0.0:
                    eQ[s, a] = np.inf
                else:
                    eQ[s, a] = np.sqrt(2 * np.log(2 * ((self.nb_states * self.nb_actions) / delta)) / count_sa[s, a])

        return eQ
    
    def estimate_model(self, zero_unseen=True):
        """
        build the MLE transition \hat{P} here

        :param batch:
        :param nstates:
        :param nactions:
        :param zero_unseen: a flag to take handle for unseen transitions
        :return:
        """
        if self.episodic:
            batch = [val for sublist in self.data for val in sublist]
        else:
            batch = self.data.copy()
        count_P = np.zeros((self.nb_states, self.nb_actions, self.nb_states))
        for transition in batch:
            state = transition[1]
            action = transition[0]
            next_state = transition[2]

            count_P[state, action, next_state] += 1.0

        # do the normalization here
        est_P = count_P / np.sum(count_P, 2)[:, :, np.newaxis]

        if zero_unseen:
            est_P = np.nan_to_num(est_P)
        else:
            est_P[np.isnan(est_P)] = 1.0/self.nb_states

        return est_P

    def make_policy_iteration_operator(self,):
        """
        IMP:
        P: sat
        R: sa
        C: sa

        coeffs: the list of coefficients (\lambda_i) for each signal AND \lamba_i >= 0

        epsilon: if eps = np.inf, then (\pi_b,e_Q,eps) constraint is not enforced
        """
        # nstates = R.shape[0]
        # nactions = R.shape[1]

        # calculate the estimates for the baseline policy
        vR_b = direct_policy_evaluation(self.P, self.R_state_action, self.gamma, self.pi_b)
        QR_b = self.R_state_action + self.gamma * np.einsum('sat,t -> sa', self.P, vR_b)
        AR_b = QR_b - vR_b.reshape((self.nb_states, 1))

        vC_b = direct_policy_evaluation(self.P, self.C_state_action, self.gamma, self.pi_b)
        QC_b = self.C_state_action + self.gamma * np.einsum('sat,t -> sa', self.P, vC_b)
        AC_b = QC_b - vC_b.reshape((self.nb_states, 1))


        def constrained_spibb_policy_iteration_operator(policy):

            # compute Q using direct policy evaluation
            # for the reward
            vR = direct_policy_evaluation(self.P, self.R_state_action, self.gamma, policy)
            QR = self.R_state_action + self.gamma * np.einsum('sat,t -> sa', self.P, vR)

            # for the cost
            vC = direct_policy_evaluation(self.P, self.C_state_action, self.gamma, policy)
            QC = self.C_state_action + self.gamma * np.einsum('sat,t -> sa', self.P, vC)

            # create the objective

            QL = self.lambda_coeffs[0] * QR - self.lambda_coeffs[1] * QC
            # placeholder policy
            soln_pi = np.zeros((self.nb_states, self.nb_actions))

            # add state based constraints
            for s in range(self.nb_states):

                # OPT
                pi = cp.Variable(shape=(1, self.nb_actions))  # prob for each action in each state
                obj = cp.Maximize(cp.sum(cp.multiply(pi, QL[[s]])))  # <Q_L(s,.), \pi(.|s)>

                # add lower bound constraint
                constr = [pi[0] >= 0.0]

                # define the probability constraints
                constr += [cp.sum(pi[0]) == 1.0]

                # to find which err function estimates are okay to use in the optimization
                # only keep those that are < np.inf, and keep the rest to 0
                ok_err = np.zeros_like(self.error_function[s])
                correction_idx = self.error_function[s] < np.inf
                ok_err[correction_idx] = self.error_function[s][correction_idx]

                # add the constraints now based on corrected ok_err
                if self.epsilon < np.inf:
                    constr += [cp.sum(cp.multiply(cp.abs(pi[0] - self.pi_b[s]), ok_err)) <= self.epsilon]

                # Advantage based constraints
                constr += [cp.sum(cp.multiply(pi, AR_b[[s]])) >= 0.0]  # R
                constr += [cp.sum(cp.multiply(pi, AC_b[[s]])) <= 0.0]  # C

                # Add another constraint based on correction index that preserves the
                #   value of the baseline policy for that index
                for a in range(self.nb_actions):
                    if (self.error_function[s][a] >= np.inf) and (self.epsilon < np.inf):
                        constr += [pi[0][a] == self.pi_b[s][a]]

                # solve
                prob = cp.Problem(obj, constr)
                prob.solve()

                new_policy = pi.value

                # copy the solution for this state
                soln_pi[s] = new_policy[0]

            return soln_pi

        return constrained_spibb_policy_iteration_operator
    
    
    def fit(self):
        try:
            # Note: we are giving the baseline policy as the initial policy to the operator
            #        this can be the random policy also i.e. pi_random
            self.pi = bounded_successive_approximation(self.pi_b,
                                                        operator=self.operator,
                                                        termination_condition=self.termination_condition,
                                                        max_limit=self.max_nb_it )
        except cp.error.SolverError:
            # if unable to solve return the baseline
            print("Couldn't solve, returning baseline")
            self.pi = self.pi_b
