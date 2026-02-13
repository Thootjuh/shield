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

        print("start")
        self.gamma = gamma
        self.data = data
        self.episodic = episodic
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.epsilon = epsilon
        self.max_nb_it = max_nb_it

        self.R_state_state = R_state_state
        self.C_state_state = C_state_state


        self.pi_b = pi_b
        self.termination_condition = default_termination
        print("set arguments")
        # ---------- Counts ----------
        print("count")
        self._count()

        # ---------- Sparse models ----------
        print("estimate P")
        self.transition_model = self.estimate_model()
        print("error_function")
        self.error_function = self.compute_error_function()
        print("reward SA")
        self.R_state_action = self._compute_R_state_action()
        print("Cost SA")
        self.C_state_action = self._compute_C_state_action()

        self.lambda_coeffs = (1.0, 1.0)

        print("operator")
        self.operator = self.make_policy_iteration_operator()

    # ====================================================================== #
    # Counting
    # ====================================================================== #

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

    # ====================================================================== #
    # Sparse models
    # ====================================================================== #

    def estimate_model(self):
        P = {}
        for (s, a, sp), c in self.count_state_action_state.items():
            P[(s, a, sp)] = c / self.count_state_action[(s, a)]
        return P

    def compute_error_function(self, delta=1.0):
        eQ = {}
        for (s, a), c in self.count_state_action.items():
            if c == 0:
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

    # ====================================================================== #
    # Helpers
    # ====================================================================== #

    def _Q_from_V(self, V, Rsa):
        Q = np.zeros((self.nb_states, self.nb_actions))
        for (s, a), r in Rsa.items():
            Q[s, a] = r
        for (s, a, sp), p in self.transition_model.items():
            Q[s, a] += self.gamma * p * V[sp]
        return Q

    def _get_error(self, s, a):
        return self.error_function.get((s, a), np.inf)

    # ====================================================================== #
    # SPIBB Operator
    # ====================================================================== #

    def make_policy_iteration_operator(self):

        # ---- baseline advantages (computed once) ----
        print("vR_b")
        vR_b = direct_policy_evaluation(
            self.transition_model,
            self.R_state_action,
            self.gamma,
            self.pi_b
        )
        print("QR_b")
        QR_b = self._Q_from_V(vR_b, self.R_state_action)
        AR_b = QR_b - vR_b[:, None]

        print("vC_b")
        vC_b = direct_policy_evaluation(
            self.transition_model,
            self.C_state_action,
            self.gamma,
            self.pi_b
        )
        
        print("QC_b")
        QC_b = self._Q_from_V(vC_b, self.C_state_action)
        AC_b = QC_b - vC_b[:, None]

        # ---- active states only ----
        active_states = set(
            s for (s, _) in self.R_state_action.keys()
        ) | set(
            s for (s, _) in self.C_state_action.keys()
        ) | set(
            s for (s, _) in self.count_state_action.keys()
        )

        # ---- projection helper ----
        print("project")
        def project_spiBB_simplex(pi_greedy, pi_b, err, epsilon, fixed_mask):
            """
            Project greedy policy onto SPIBB constraint set:
            - simplex
            - L1 baseline constraint
            - fixed actions
            """
            pi = pi_greedy.copy()

            # enforce fixed actions
            pi[fixed_mask] = pi_b[fixed_mask]

            # remaining probability mass
            free = ~fixed_mask
            mass_free = 1.0 - pi[fixed_mask].sum()

            if mass_free <= 0:
                return pi_b.copy()

            # normalize free part
            pi_free = pi[free]
            pi_free = np.maximum(pi_free, 0)
            if pi_free.sum() == 0:
                pi_free = np.ones_like(pi_free) / len(pi_free)
            else:
                pi_free /= pi_free.sum()

            pi[free] = mass_free * pi_free

            # L1 trust region projection (approximate, fast)
            if epsilon < np.inf:
                diff = pi - pi_b
                weighted_l1 = np.sum(np.abs(diff) * err)
                if weighted_l1 > epsilon:
                    scale = epsilon / (weighted_l1 + 1e-12)
                    pi = pi_b + scale * diff

            # final simplex safety
            pi = np.maximum(pi, 0)
            pi /= pi.sum()

            return pi

        # ---- main operator ----
        print("QR_b")
        def operator(policy):

            vR = direct_policy_evaluation(
                self.transition_model,
                self.R_state_action,
                self.gamma,
                policy
            )
            QR = self._Q_from_V(vR, self.R_state_action)

            vC = direct_policy_evaluation(
                self.transition_model,
                self.C_state_action,
                self.gamma,
                policy
            )
            QC = self._Q_from_V(vC, self.C_state_action)

            QL = self.lambda_coeffs[0] * QR - self.lambda_coeffs[1] * QC

            new_pi = policy.copy()

            # iterate ONLY over active states
            for s in active_states:

                q = QL[s]

                # greedy distribution (ties handled)
                best = np.flatnonzero(q == q.max())
                pi_greedy = np.zeros(self.nb_actions)
                pi_greedy[best] = 1.0 / len(best)

                # error vector
                err = np.array([
                    self._get_error(s, a) if self._get_error(s, a) < np.inf else 0.0
                    for a in range(self.nb_actions)
                ])

                # fixed actions
                fixed_mask = np.array([
                    self._get_error(s, a) == np.inf
                    for a in range(self.nb_actions)
                ])

                # enforce advantage constraints
                # reward advantage >= 0
                if np.dot(pi_greedy, AR_b[s]) < 0:
                    pi_greedy = self.pi_b[s].copy()

                # cost advantage <= 0
                if np.dot(pi_greedy, AC_b[s]) > 0:
                    pi_greedy = self.pi_b[s].copy()

                # SPIBB projection
                new_pi[s] = project_spiBB_simplex(
                    pi_greedy,
                    self.pi_b[s],
                    err,
                    self.epsilon,
                    fixed_mask
                )

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
