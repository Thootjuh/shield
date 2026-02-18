"""
MO_SPIBB_Sparse.py

Fully sparse MO-SPIBB (S-OPT).

All internal models are sparse dictionaries.
The returned policy is a dense numpy array (S,A).
"""

import cvxpy as cp
import numpy as np
from copy import deepcopy
from collections import defaultdict
from .utils_sparse import direct_policy_evaluation, bounded_successive_approximation, default_termination


class ConstSPIBBAgent():
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
                 max_nb_it=25,
                 **kwargs):

        self.gamma = gamma
        self.data = data
        self.episodic = episodic
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.epsilon = epsilon
        self.max_nb_it = max_nb_it
        
        self.R_state_state = self._to_sparse_dict(R_state_state)
        self.C_state_state = self._to_sparse_dict(C_state_state)


        self.pi_b = pi_b
        self.termination_condition = default_termination
        # ---------- Counts ----------
        self._count()

        # ---------- Sparse models ----------
        self.transition_model = self.estimate_model()
        self.error_function = self.compute_error_function()
        self.R_state_action = self._compute_R_state_action()
        self.C_state_action = self._compute_C_state_action()

        self.lambda_coeffs = (1.0, 1.0)

        self.operator = self.make_policy_iteration_operator()


    def _to_sparse_dict(self, arr_or_dict):
        """
        If input is a 2D numpy array of shape (state, next_state),
        convert it to a sparse dictionary {(s, s'): value} for non-zero values.
        If it's already a dictionary, return as-is.
        """
        if isinstance(arr_or_dict, dict):
            return arr_or_dict

        if isinstance(arr_or_dict, np.ndarray):
            sparse_dict = {}
            states, next_states = arr_or_dict.shape
            for s in range(states):
                for ns in range(next_states):
                    value = arr_or_dict[s, ns]
                    if value != 0:
                        sparse_dict[(s, ns)] = value
            return sparse_dict
    
    def _count(self):
        if self.episodic:
            batch = [x for traj in self.data for x in traj]
        else:
            batch = self.data

        self.count_state_action = defaultdict(int)
        self.count_state_action_state = defaultdict(int)

        for a, s, sp, _ in batch:
            self.count_state_action[(s, a)] += 1
            self.count_state_action_state[(s, a, sp)] += 1


    def estimate_model(self):
        P = {}
        for (s, a, sp), c in self.count_state_action_state.items():
            P[(s, a, sp)] = c / self.count_state_action[(s, a)]
        return P

    def compute_error_function(self, delta=1.0):
        eQ = {}
        for (s, a), c in self.count_state_action.items():
            if c <= 7:
                eQ[(s, a)] = np.inf
            else:
                eQ[(s, a)] = np.sqrt(
                    2 * np.log(2 * (self.nb_states * self.nb_actions / delta)) / c
                )
        return eQ

    def _compute_R_state_action(self):
        Rsa = defaultdict(float)
        for (s, a, sp), p in self.transition_model.items():
            Rsa[(s, a)] += p * self.R_state_state.get((s, sp), 0.0)
        return Rsa

    def _compute_C_state_action(self):
        Csa = defaultdict(float)
        for (s, a, sp), p in self.transition_model.items():
            Csa[(s, a)] += p * self.C_state_state.get((s, sp), 0.0)
        return Csa
    
    def _Q_from_V(self, V, Rsa):
        Q = np.zeros((self.nb_states, self.nb_actions))
        for (s, a), r in Rsa.items():
            Q[s, a] = r
        for (s, a, sp), p in self.transition_model.items():
            Q[s, a] += self.gamma * p * V[sp]
        return Q

    def _get_error(self, s, a):
        return self.error_function.get((s, a), np.inf)

    def make_policy_iteration_operator(self):
            vR_b = direct_policy_evaluation(self.transition_model,
                                            self.R_state_action,
                                            self.gamma,
                                            self.pi_b)
            QR_b = self._Q_from_V(vR_b, self.R_state_action)
            AR_b = QR_b - vR_b[:, None]

            vC_b = direct_policy_evaluation(self.transition_model,
                                            self.C_state_action,
                                            self.gamma,
                                            self.pi_b)
            QC_b = self._Q_from_V(vC_b, self.C_state_action)
            AC_b = QC_b - vC_b[:, None]
            def operator(policy):
                vR = direct_policy_evaluation(self.transition_model,
                                            self.R_state_action,
                                            self.gamma,
                                            policy)
                QR = self._Q_from_V(vR, self.R_state_action)

                vC = direct_policy_evaluation(self.transition_model,
                                            self.C_state_action,
                                            self.gamma,
                                            policy)
                QC = self._Q_from_V(vC, self.C_state_action)

                QL = self.lambda_coeffs[0] * QR - self.lambda_coeffs[1] * QC

                active_states = set(
                    s for (s, _) in self.R_state_action.keys()
                ) | set(
                    s for (s, _) in self.C_state_action.keys()
                ) | set(
                    s for (s, _) in self.count_state_action.keys()
                )
                new_pi = policy.copy()
                for s in active_states:
                    pi = cp.Variable(self.nb_actions)

                    constraints = [
                        pi >= 0,
                        cp.sum(pi) == 1
                    ]

                    if self.epsilon < np.inf:
                        err = np.array([
                            self._get_error(s, a) if self._get_error(s, a) < np.inf else 0
                            for a in range(self.nb_actions)
                        ])
                        constraints.append(
                            cp.sum(cp.multiply(cp.abs(pi - self.pi_b[s]), err)) <= self.epsilon
                        )

                    constraints += [
                        pi @ AR_b[s] >= 0,
                        pi @ AC_b[s] <= 0
                    ]

                    for a in range(self.nb_actions):
                        if self._get_error(s, a) == np.inf:
                            constraints.append(pi[a] == self.pi_b[s, a])

                    prob = cp.Problem(cp.Maximize(pi @ QL[s]), constraints)
                    # prob.solve(solver=cp.ECOS, warm_start=True)
                    prob.solve()
                    if pi.value is not None:
                        new_pi[s] = pi.value

                return new_pi

            return operator

    # ====================================================================== #
    # Training
    # ====================================================================== #

    def fit(self):
        self.pi = bounded_successive_approximation(
            self.pi_b,
            operator=self.operator,
            termination_condition=self.termination_condition,
            max_limit=self.max_nb_it
        )
