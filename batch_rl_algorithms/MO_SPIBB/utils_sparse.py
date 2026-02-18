import numpy as np
import cvxpy as cp
from typing import Callable, Generator, Tuple

def direct_policy_evaluation(
    P: dict,
    R: dict,
    discount: float,
    policy: np.ndarray,
    tol: float = 1e-4,
    max_iter: int = 10_000
) -> np.ndarray:


    nb_states, nb_actions = policy.shape

    transitions_sa = {}
    for (s, a, sp), prob in P.items():
        transitions_sa.setdefault(s, {}).setdefault(a, []).append((sp, prob))

    V = np.zeros(nb_states)

    for _ in range(max_iter):
        delta = 0.0

        for s in range(nb_states):
            v_old = V[s]
            v_new = 0.0

            for a in range(nb_actions):
                pi_sa = policy[s, a]
                if pi_sa == 0.0:
                    continue
                r_sa = R.get((s, a), 0.0)
                exp_next = 0.0
                for sp, prob in transitions_sa.get(s, {}).get(a, []):
                    exp_next += prob * V[sp]

                v_new += pi_sa * (r_sa + discount * exp_next)

            V[s] = v_new
            delta = max(delta, abs(v_new - v_old))

        if delta < tol:
            break

    return V



def generate_iterates(xinit: np.ndarray,
                      operator: Callable[[np.ndarray], np.ndarray],
                      termination_condition: Callable[[np.ndarray, np.ndarray], bool]
                      ) -> Generator[np.ndarray, None, None]:
    """

    :param xinit: initial value
    :param operator: operator to apply on the function
    :param termination_condition: checks when to terminate
    :return: a generator that gives the next iterates
    """
    x, xprev = operator(xinit), xinit
    
    yield x
    while not termination_condition(xprev, x):
        x, xprev = operator(x), x
        yield x


def successive_approximation(xinit: np.ndarray,
                             operator=lambda x: x,
                             termination_condition=lambda xprev, x: False):
    """
    Iteratively applies the operator until satisfied by the terminatation condition

    :param xinit: initial value
    :param operator: operator to apply on the function
    :param termination_condition: checks when to terminate
    :return:
    """
    for iterate in generate_iterates(xinit, operator, termination_condition):
        pass
    return iterate


def bounded_successive_approximation(xinit,
                                     operator=lambda x: x,
                                     termination_condition=lambda xprev, x: False,
                                     max_limit=100):
    """
    Iterations are bounded bt the max_limit variable

    :param xinit:
    :param operator:
    :param termination_condition:
    :param max_limit:
    :return:
    """
    count = 0
    for iterate in generate_iterates(xinit, operator, termination_condition):
        print(count)
        count += 1
        if count >= max_limit:
            break
            

    return iterate

def default_termination(xprev, x, epsilon=1e-4):
    """
    A standard termination condition
    :param xprev:
    :param x:
    :param epsilon:
    :return:
    """
    return np.linalg.norm(xprev - x) < epsilon



def estimate_model(batch, nstates, nactions, zero_unseen=True):
    """
    build the MLE transition \hat{P} here

    :param batch:
    :param nstates:
    :param nactions:
    :param zero_unseen: a flag to take handle for unseen transitions
    :return:
    """
    count_P = np.zeros((nstates, nactions, nstates))
    for transition in batch:
        state = transition[0]
        action = transition[1]
        next_state = transition[-1]

        count_P[state, action, next_state] += 1.0

    # do the normalization here
    est_P = count_P / np.sum(count_P, 2)[:, :, np.newaxis]

    if zero_unseen:
        est_P = np.nan_to_num(est_P)
    else:
        est_P[np.isnan(est_P)] = 1.0/nstates

    return est_P


def compute_error_function(batch, nstates: int, nactions: int, delta=1.0):
    """
    Computes the e_Q function based on the dataset

    :param batch:
    :param nstates:
    :param nactions:
    :param delta:
    :return:
    """
    count_sa = np.zeros((nstates, nactions))
    eQ = np.zeros((nstates, nactions))

    # for each transition in batch
    for transition in batch:
        state = transition[0]
        action = transition[1]

        count_sa[state, action] += 1.0

    # for each (s,a)
    for s in range(nstates):
        for a in range(nactions):
            if count_sa[s, a] == 0.0:
                eQ[s, a] = np.inf
            else:
                eQ[s, a] = np.sqrt(2 * np.log(2 * ((nstates * nactions) / delta)) / count_sa[s, a])

    return eQ