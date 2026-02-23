"""
base agent class, and the unconstrained Policy Improvement for tabular CMDPs
"""

from copy import deepcopy
import numpy as np
import cvxpy as cp

class Agent:
    """
    base class
    """
    def __init__(self, **kwargs):
        self.params = deepcopy(kwargs)

    def set_logger(self, logger):
        self.logger = logger

    def make_policy_iteration_operator(self, **args):
        raise NotImplementedError

    def log(self):
        pass