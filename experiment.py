import os
import sys
import ast
import time
from distutils import util
import configparser
from collections import defaultdict
from collections.abc import Mapping
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.stats import norm


from environments.gym_cartpole_env import cartPole, cartPolePolicy
from environments.gym_maze.gym_maze_env import maze, mazePolicy
from environments.gym_crashing_mountain_car import crashingMountainCar, crashingMountainCarPolicy
from environments.gym_lunar_lander import LunarLander, LunarLanderPolicy
from environments.gym_frozen_lake import gymIce
from environments.MovingObstacles import MovingObstacles, MovingObstaclesPolicy

from batch_rl_algorithms.basic_rl import Basic_rl
from batch_rl_algorithms.pi_star import PiStar
from batch_rl_algorithms.spibb import SPIBB, Lower_SPIBB
from batch_rl_algorithms.r_min import RMin
from batch_rl_algorithms.soft_spibb import ApproxSoftSPIBB, ExactSoftSPIBB, LowerApproxSoftSPIBB, AdvApproxSoftSPIBB
from batch_rl_algorithms.duipi import DUIPI
from batch_rl_algorithms.ramdp import RaMDP
from batch_rl_algorithms.mbie import MBIE
from batch_rl_algorithms.pi_star import PiStar
from batch_rl_algorithms.rmdp import WorstCaseRMDP
from batch_rl_algorithms.shielded.shielded_rmdp import Shield_WorstCaseRMDP
from batch_rl_algorithms.shielded.shielded_baseline import shieldedBaseline
from batch_rl_algorithms.shielded.shielded_spibb import Shield_SPIBB, Shield_Lower_SPIBB
from batch_rl_algorithms.shielded.shielded_duipi import shield_DUIPI
from batch_rl_algorithms.shielded.shielded_raMDP import Shield_RaMDP
from batch_rl_algorithms.shielded.shielded_mbie import shield_MBIE
from batch_rl_algorithms.shielded.shielded_r_min import Shield_RMin
from batch_rl_algorithms.spibb_dqn.spibb_dqn import spibb_dqn
from batch_rl_algorithms.CQL_DQN.train_CQL_DQN import train_cql_dqn, train_cql_dqn_hybrid

from shield import ShieldRandomMDP, ShieldCartpole, ShieldCrashingMountainCar, ShieldMaze, ShieldLunarLander, ShieldFrozenLake, ShieldMovingObstacles
from PACIntervalEstimator import PACIntervalEstimator
from evaluate_policy import evaluate_policy
from discretization.grid.define_imdp import imdp_builder

from discretization.MRL.helper_functions import trajToDF, state2region, state2region_fast
from discretization.MRL.model import MDP_model
from discretization.MRL.mrl_model import MRL_model
from discretization.MRL.testing import predict_cluster
from discretization.MRL_scratch.mrl_scratch import MRL_scratch
from discretization.greedyCut.greedyCut import GreedyCut

directory = os.path.dirname(os.path.expanduser(__file__))

def write_policy_to_file(policy, filename):
    """
    Writes a policy to a file.

    Parameters
    ----------
    policy : np.ndarray
        A 2D array of shape (num_states, num_actions) where
        policy[s, a] is the probability of taking action a in state s.
    filename : str
        Path to the output file.
    """
    if policy.ndim != 2:
        raise ValueError("Policy must be a 2D numpy array (states x actions).")

    with open(filename, "w") as f:
        for state_probs in policy:
            line = ",".join(map(str, state_probs))
            f.write(line + "\n")

class Experiment:
    # Class to implement general batch RL experiments
    results = []
    nb_iterations = None
    fixed_params_exp_list = None
    fixed_params_exp_columns = None
    variable_params_exp_columns = None
    algorithms_columns = ['method', 'hyperparam', 'method_perf', 'discounted_method_perf', 'run_time', 'nb_states', 'success_rate', 'failure_rate', 'avoid_rate', 'predicted_perf']

    def __init__(self, experiment_config, seed, nb_iterations, machine_specific_experiment_directory):
        """
        :param experiment_config: config file which describes the experiment, see, for example,
        experiments/wet_chicken_full.ini
        :param seed: seed for this experiment
        :param nb_iterations: number of iterations of this experiment
        :param machine_specific_experiment_directory: the directory in which the results will be stored
        """
        self.seed = seed
        np.random.seed(seed)
        self.experiment_config = experiment_config
        self.machine_specific_experiment_directory = machine_specific_experiment_directory

        self.filename_header = f'results_{seed}'
        self.nb_iterations = nb_iterations
        self.safety_deltas = ast.literal_eval(self.experiment_config['META']['safety_deltas'])
        if self.safety_deltas:
            self.safety_columns = ['delta', 'bound']
        else:
            self.safety_columns = []
        print(f'Initialising experiment with seed {seed} and {nb_iterations} iterations.')
        print(f'The machine_specific_experiment_directory is {self.machine_specific_experiment_directory}.')
        self.theoretical_safety = bool(util.strtobool(self.experiment_config['META']['theoretical_safety']))
        self.g_max = float(self.experiment_config['ENV_PARAMETERS']['G_MAX'])
        self.algorithms_dict = ast.literal_eval(self.experiment_config['ALGORITHMS']['algorithms_dict'])
        self.speed_up = bool(util.strtobool(self.experiment_config['META']['speed_up']))
        self.set_up_speed_up_dict()
        self._set_env_params()

    def run(self):
        """
        Runs the experiment.
        """
        for iteration in range(self.nb_iterations):
            self.to_append_run = self.fixed_params_exp_list + [iteration]
            self._run_one_iteration()
            self._save(iteration)

    def _save(self, iteration):
        """
        Saves the result after each iteration.
        :param iteration: iteration + 1 iterations are done, usage only for naming
        """
        print(self.fixed_params_exp_columns + self.variable_params_exp_columns + self.algorithms_columns + self.safety_columns)
        results_df = pd.DataFrame(self.results,
                                  columns=self.fixed_params_exp_columns + self.variable_params_exp_columns + self.algorithms_columns + self.safety_columns)
        filename = self.filename_header + f"_up_to_iteration_{iteration + 1}.csv"
        results_df.to_csv(os.path.join(self.machine_specific_experiment_directory, filename))
        print(str(len(self.results)) + ' lines saved to ' + os.path.join(self.machine_specific_experiment_directory,
                                                                         filename))
        if iteration > 0:
            os.remove(os.path.join(self.machine_specific_experiment_directory,
                                   self.filename_header + f"_up_to_iteration_{iteration}.csv"))

    def set_up_speed_up_dict(self):
        """
        Makes use of self.speed_up_dict, according to the rules in the algorithms in batch_rl_algorithms/, if
        self.speed_up=True.
        """
        if self.speed_up:
            self.speed_up_dict = {
                'count_state_action': None,
                'count_state_action_state': None,
            }
            if self.theoretical_safety:
                self.speed_up_dict['augmented_count_state_action'] = None
            if ApproxSoftSPIBB.NAME in self.algorithms_dict.keys():
                if 'mpeb' in self.algorithms_dict[ApproxSoftSPIBB.NAME]['error_kinds']:
                    self.speed_up_dict['q_pi_b_est'] = None
                    self.speed_up_dict['var_q'] = None
            if ExactSoftSPIBB.NAME in self.algorithms_dict.keys():
                if 'mpeb' in self.algorithms_dict[ApproxSoftSPIBB.NAME]['error_kinds']:
                    self.speed_up_dict['q_pi_b_est'] = None
                    self.speed_up_dict['var_q'] = None
            if LowerApproxSoftSPIBB.NAME in self.algorithms_dict.keys():
                if 'mpeb' in self.algorithms_dict[LowerApproxSoftSPIBB.NAME]['error_kinds']:
                    self.speed_up_dict['q_pi_b_est'] = None
                    self.speed_up_dict['var_q'] = None
            if AdvApproxSoftSPIBB.NAME in self.algorithms_dict.keys():
                if 'mpeb' in self.algorithms_dict[ApproxSoftSPIBB.NAME]['error_kinds']:
                    self.speed_up_dict['q_pi_b_est'] = None
                    self.speed_up_dict['var_q'] = None
                else:
                    self.speed_up_dict['q_pi_b_est'] = None
        else:
            self.speed_up_dict = None

    def _set_env_params(self):
        pass

    def _run_one_iteration(self, params_env):
        pass

    def _compute_speed_up_dict(self):
        """
        Sets the speed_up_dict up, when a new data set was generated.
        :return:
        """
        if 'var_q' in self.speed_up_dict.keys():
            preparer = ApproxSoftSPIBB(pi_b=self.pi_b, gamma=self.gamma, nb_states=self.nb_states,
                                       nb_actions=self.nb_actions, data=self.data, R=self.R_state_state, epsilon=0,
                                       error_kind='mpeb', episodic=self.episodic, delta=1, max_nb_it=0,
                                       g_max=self.g_max,
                                       ensure_independence=self.theoretical_safety)
            self.speed_up_dict['var_q'] = preparer.var_q
            self.speed_up_dict['q_pi_b_est'] = preparer.q_pi_b_est
        elif 'q_pi_b_est' in self.speed_up_dict.keys():
            preparer = AdvApproxSoftSPIBB(pi_b=self.pi_b, gamma=self.gamma, nb_states=self.nb_states,
                                          nb_actions=self.nb_actions, data=self.data, R=self.R_state_state,
                                          epsilon=0, error_kind='hoeffding', episodic=self.episodic, delta=1,
                                          max_nb_it=0, ensure_independence=self.theoretical_safety)
            self.speed_up_dict['q_pi_b_est'] = preparer.q_pi_b_est
        else:
            preparer = Basic_rl(pi_b=self.pi_b, gamma=self.gamma, nb_states=self.nb_states, nb_actions=self.nb_actions,
                                data=self.data, R=self.R_state_state, episodic=self.episodic)
        if self.theoretical_safety:
            self.speed_up_dict['augmented_count_state_action'] = preparer.augmented_count_state_action
        self.speed_up_dict['count_state_action'] = preparer.count_state_action
        self.speed_up_dict['count_state_action_state'] = preparer.count_state_action_state

        
    def _run_algorithms(self):
        """
        Runs all algorithms for one data set.
        """
        if self.speed_up:
            self._compute_speed_up_dict()
        self._run_shielded_baseline()
        self._run_baseline()
        for key in self.algorithms_dict.keys():
            print(key)
            if key in {SPIBB.NAME, Lower_SPIBB.NAME}:
                self._run_spibb(key)
            elif key in {ExactSoftSPIBB.NAME, ApproxSoftSPIBB.NAME, LowerApproxSoftSPIBB.NAME, AdvApproxSoftSPIBB.NAME}:
                self._run_soft_spibb(key)
            # elif key in {'SPIBB-DQN'}:
                # self._run_spibb_dqn(key)
            elif key in {Shield_SPIBB.NAME, Shield_Lower_SPIBB.NAME}:
                self._run_spibb_shielded(key)
            elif key in {Basic_rl.NAME}:
                self._run_basic_rl(key)
            else:
                print("KEY NOT FOUND")
            
    def _run_baseline(self):
        if self.discretization_method=='grid':
            method_perf, discounted_method_perf, pi_b_succ_rate, pi_b_fail_rate, pi_b_avoid_rate = evaluate_policy(self.env, self.pi_b, 100, 250, self.discretization_method, env_name=self.env_name, generate_gif=self.generate_gif, gif_name="baseline_grid.gif",render_env=self.render_env, gamma=self.gamma)
        elif self.discretization_method=='mrl':
            method_perf, discounted_method_perf,pi_b_succ_rate, pi_b_fail_rate, pi_b_avoid_rate = evaluate_policy(self.env, self.pi_b, 100, 250, self.discretization_method, predictor=self.predictor, dimensions=self.dimensions, env_name=self.env_name, generate_gif=self.generate_gif, gif_name="baseline_mrl.gif",render_env=self.render_env, gamma=self.gamma)
        elif self.discretization_method=='GreedyCut':
            method_perf, discounted_method_perf,pi_b_succ_rate, pi_b_fail_rate, pi_b_avoid_rate = evaluate_policy(self.env, self.pi_b, 100, 250, self.discretization_method, predictor=self.predictor, dimensions=self.dimensions, ai=self.discretizer, env_name=self.env_name, generate_gif=self.generate_gif, gif_name="baseline_GC.gif",render_env=self.render_env, gamma=self.gamma)
        method = "baseline_" + self.discretization_method
        print("perf = ", method_perf)
        print("succ_rate = ", pi_b_succ_rate)
        print("avoid_rate = ", pi_b_avoid_rate)
        with open("perf.txt", "a") as f:
            f.write("BASELINE GRID\n")
            f.write(f"perf = {method_perf}\n")
            f.write(f"succ rate = {pi_b_succ_rate}\n")
            f.write(f"avoid rate = {pi_b_avoid_rate}\n")
            f.write("\n")
        # method_perf_pred = policy_evaluation_exact(self.pi_b,self.R_s_a, self.transition_model, self.gamma)[0][self.initial_state]
        method_perf_pred = 0
        hyperparam = None
        run_time = 0.0
        self.results.append(self.to_append + [method, hyperparam, method_perf, discounted_method_perf, run_time, self.nb_states, pi_b_succ_rate, pi_b_fail_rate, pi_b_avoid_rate, method_perf_pred])
        
    def _run_shielded_baseline(self):
        pi_b_s = shieldedBaseline(pi_b=self.pi_b, gamma=self.gamma, nb_states=self.nb_states,
                                            nb_actions=self.nb_actions, data=self.data, R=self.R_state_state,
                                            episodic=self.episodic, shield=self.shielder, speed_up_dict=self.speed_up_dict, estimate_baseline=self.estimate_baseline)
        t_0 = time.time()
        pi_b_s.fit()
        t_1 = time.time()
        if self.discretization_method=='mrl':
            basic_rl_perf,discounted_method_perf, pi_b_succ_rate, pi_b_fail_rate, pi_b_avoid_rate = evaluate_policy(self.env, pi_b_s.pi, 100, 250, self.discretization_method, predictor=self.predictor, dimensions=self.dimensions, env_name=self.env_name, generate_gif=self.generate_gif, gif_name="shielded_baseline_mrl.gif",render_env=self.render_env, gamma=self.gamma)
        elif self.discretization_method=='grid':
            basic_rl_perf,discounted_method_perf, pi_b_succ_rate, pi_b_fail_rate, pi_b_avoid_rate = evaluate_policy(self.env, pi_b_s.pi, 100, 250, self.discretization_method, env_name=self.env_name, generate_gif=self.generate_gif, gif_name="shielded_baseline_grid.gif",render_env=self.render_env, gamma=self.gamma)
        elif self.discretization_method=='GreedyCut':
            basic_rl_perf,discounted_method_perf, pi_b_succ_rate, pi_b_fail_rate, pi_b_avoid_rate = evaluate_policy(self.env, pi_b_s.pi, 100, 250, self.discretization_method, predictor=self.predictor, dimensions=self.dimensions, ai=self.discretizer, env_name=self.env_name, generate_gif=self.generate_gif, gif_name="shielded_baseline_GC.gif",render_env=self.render_env, gamma=self.gamma)
        method = "baseline_" + self.discretization_method
        # method_perf_pred = policy_evaluation_exact(pi_b_s.pi,self.R_s_a, self.transition_model, self.gamma)[0][self.initial_state]
        method_perf_pred = 0
        method = pi_b_s.NAME + "_" + self.discretization_method
        method_perf = basic_rl_perf
        hyperparam = None
        run_time = t_1 - t_0
        self.results.append(self.to_append + [method, hyperparam, method_perf, discounted_method_perf, run_time, self.nb_states, pi_b_succ_rate, pi_b_fail_rate,  pi_b_avoid_rate, method_perf_pred])
    
    def _run_spibb_shielded(self, key):
        """
        Runs SPIBB or Lower-SPIBB for one data set, with all hyper-parameters.
        :param key: shield_SPIBB.NAME or shield_Lower_SPIBB.NAME, depending on which algorithm is supposed to be run
        """
        # 1. Modified data
        for N_wedge in self.algorithms_dict[key]['hyperparam']:
            spibb = algorithm_name_dict[key](pi_b=self.pi_b, gamma=self.gamma, nb_states=self.nb_states,
                                             nb_actions=self.nb_actions, data=self.data, R=self.R_state_state,
                                             N_wedge=N_wedge, episodic=self.episodic, shield=self.shielder, 
                                             speed_up_dict=self.speed_up_dict,estimate_baseline=self.estimate_baseline)
            t_0 = time.time()
            spibb.fit()
            t_1 = time.time()
            if self.discretization_method=='mrl':
                # spibb_perf = evaluate_policy(self.env, spibb.pi, 1, 100)
                spibb_perf, discounted_method_perf,succ_rate, failure_rate, avoid_rate = evaluate_policy(self.env, spibb.pi, 1000, 250, self.discretization_method, predictor=self.predictor, dimensions=self.dimensions, env_name=self.env_name, generate_gif=self.generate_gif, gif_name="shielded_spibb_mrl.gif",render_env=self.render_env, gamma=self.gamma)
            elif self.discretization_method=='grid':
                spibb_perf, discounted_method_perf,succ_rate, failure_rate, avoid_rate = evaluate_policy(self.env, spibb.pi, 1000, 250,  self.discretization_method, env_name=self.env_name, generate_gif=self.generate_gif, gif_name="shielded_spibb_grid.gif",render_env=self.render_env, gamma=self.gamma)
            elif self.discretization_method=='GreedyCut':
                spibb_perf, discounted_method_perf,succ_rate, failure_rate, avoid_rate = evaluate_policy(self.env, spibb.pi, 1000, 250, self.discretization_method, predictor=self.predictor, dimensions=self.dimensions, ai=self.discretizer, env_name=self.env_name, generate_gif=self.generate_gif, gif_name="shielded_spibb_GC.gif",render_env=self.render_env, gamma=self.gamma)
            method = "baseline_" + self.discretization_method
            # method_perf_pred = policy_evaluation_exact(spibb.pi,self.R_s_a, self.transition_model, self.gamma)[0][self.initial_state]
            method_perf_pred = 0
            method = spibb.NAME + "_" + self.discretization_method
            method_perf = spibb_perf
            hyperparam = N_wedge
            run_time = t_1 - t_0
            write_policy_to_file(spibb.pi, "policy_shielded.txt")
            self.results.append(self.to_append + [method, hyperparam, method_perf, discounted_method_perf, run_time, self.nb_states, succ_rate, failure_rate, avoid_rate, method_perf_pred])
                  
    def _run_spibb(self, key):
        """
        Runs SPIBB or Lower-SPIBB for one data set, with all hyper-parameters.
        :param key: SPIBB.NAME or Lower_SPIBB.NAME, depending on which algorithm is supposed to be run
        """
        for N_wedge in self.algorithms_dict[key]['hyperparam']:
            spibb = algorithm_name_dict[key](pi_b=self.pi_b, gamma=self.gamma, nb_states=self.nb_states,
                                             nb_actions=self.nb_actions, data=self.data, R=self.R_state_state,
                                             N_wedge=N_wedge, episodic=self.episodic, speed_up_dict=self.speed_up_dict, estimate_baseline=self.estimate_baseline)
            t_0 = time.time()
            spibb.fit()
            t_1 = time.time()
            print("trained policy")
            if self.discretization_method=='mrl':
                # spibb_perf = evaluate_policy(self.env, spibb.pi, 1, 100)
                spibb_perf, discounted_method_perf,succ_rate, failure_rate, avoid_rate = evaluate_policy(self.env, spibb.pi, 1000, 250, self.discretization_method, predictor=self.predictor, dimensions=self.dimensions, env_name=self.env_name, generate_gif=self.generate_gif, gif_name="spibb_mrl.gif",render_env=self.render_env, gamma=self.gamma)
            elif self.discretization_method=='grid':
                spibb_perf, discounted_method_perf,succ_rate, failure_rate, avoid_rate = evaluate_policy(self.env, spibb.pi, 1000, 250, self.discretization_method, env_name=self.env_name, generate_gif=self.generate_gif, gif_name="spibb_grid.gif",render_env=self.render_env, gamma=self.gamma)
            elif self.discretization_method=='GreedyCut':
                spibb_perf, discounted_method_perf,succ_rate, failure_rate, avoid_rate = evaluate_policy(self.env, spibb.pi, 1000, 250, self.discretization_method, predictor=self.predictor, dimensions=self.dimensions, ai=self.discretizer, env_name=self.env_name, generate_gif=self.generate_gif, gif_name="spibb_GC.gif",render_env=self.render_env, gamma=self.gamma)
            print("evaluated policy")
            # spibb_perf = self._policy_evaluation_exact(spibb.pi)
            # method_perf_pred = policy_evaluation_exact(spibb.pi,self.R_s_a, self.transition_model, self.gamma)[0][self.initial_state]
            method_perf_pred = 0
            method = spibb.NAME + "_" + self.discretization_method
            method_perf = spibb_perf
            hyperparam = N_wedge
            run_time = t_1 - t_0
            write_policy_to_file(spibb.pi, "policy.txt")
            self.results.append(self.to_append + [method, hyperparam, method_perf, discounted_method_perf, run_time, self.nb_states, succ_rate, failure_rate,avoid_rate, method_perf_pred])
    
    def _run_spibb_dqn(self, key):
        for N_wedge in self.algorithms_dict[key]['hyperparam']:
            spibb = spibb_dqn(baseline=self.pi_b, gamma=self.gamma, state_shape=[self.dimensions], nb_actions=self.nb_actions, dataset_raw=self.data_cont, env=self.env, minimum_count=N_wedge)
            t_0 = time.time()
            spibb.learn(passes_on_dataset = 25)
            t_1 = time.time()
            spibb_perf, discounted_method_perf,succ_rate, failure_rate, avoid_rate = evaluate_policy(self.env, None, 100, 250, self.discretization_method, ai=spibb.ai, env_name=self.env_name, generate_gif=self.generate_gif, gif_name="spibb_dqn.gif",render_env=self.render_env, gamma=self.gamma)
            method = 'spibb_dqn'
            method_perf = spibb_perf
            hyperparam = N_wedge
            run_time = t_1 - t_0
            self.results.append(self.to_append + [method, hyperparam, method_perf, discounted_method_perf, run_time, self.nb_states, succ_rate, avoid_rate, failure_rate, 0])
            
    def _run_cql_dqn(self):
        t_0 = time.time()
        agent = train_cql_dqn(self.env, self.env_name, self.data_cont)
        # agent = train_cql_dqn_hybrid(self.env, self.data_cont, 100000, 100)
        t_1 = time.time()
        spibb_perf, discounted_method_perf, succ_rate, failure_rate, avoid_rate = evaluate_policy(self.env, None, 100, 250, self.discretization_method, ai=agent, env_name=self.env_name, generate_gif=self.generate_gif, gif_name="cql_dqn.gif",render_env=self.render_env, gamma=self.gamma)
        print("perf = ", spibb_perf)
        method = 'cql_dqn'
        method_perf = spibb_perf
        hyperparam = None
        run_time = t_1 - t_0
        self.results.append(self.to_append + [method, hyperparam, method_perf, discounted_method_perf, run_time, self.nb_states, succ_rate, failure_rate, avoid_rate, 0])
            
    def _run_soft_spibb(self, key):
        """
        Runs Approx-Soft-SPIBB, Exact-Soft-SPIBB, Adv-Approx-Soft-SPIBB or Lower-Approx-Soft-SPIBB for one data set,
        with all hyper-parameters.
        :param key: ApproxSoftSPIBB.NAME, ExactSoftSPIBB.NAME, LowerApproxSoftSPIBB.NAME or AdvApproxSoftSPIBB.NAME,
        depending on which algorithm is supposed to be run
        """
        error_kinds = self.algorithms_dict[key]['error_kinds']
        one_steps = self.algorithms_dict[key]['1-step']
        if self.safety_deltas:
            deltas = self.safety_deltas
        else:
            deltas = self.algorithms_dict[key]['deltas']
        for error_kind in error_kinds:
            for delta in deltas:
                for one_step in one_steps:
                    for epsilon in self.algorithms_dict[key]['hyperparam']:
                        if one_step:
                            max_nb_it = 1
                            prefix = '1-Step-'
                        else:
                            max_nb_it = 5000
                            prefix = ''
                        soft_spibb = algorithm_name_dict[key](pi_b=self.pi_b, gamma=self.gamma,
                                                              nb_states=self.nb_states,
                                                              nb_actions=self.nb_actions, data=self.data,
                                                              R=self.R_state_state,
                                                              epsilon=epsilon, error_kind=error_kind,
                                                              episodic=self.episodic,
                                                              delta=delta, max_nb_it=max_nb_it,
                                                              speed_up_dict=self.speed_up_dict, g_max=self.g_max,
                                                              ensure_independence=self.theoretical_safety,
                                                              estimate_baseline=self.estimate_baseline)
                        t_0 = time.time()
                        soft_spibb.fit()
                        t_1 = time.time()
                        spibb_perf = self._policy_evaluation_exact(soft_spibb.pi)
                        method = prefix + soft_spibb.NAME + '_' + error_kind
                        method_perf = spibb_perf
                        hyperparam = epsilon
                        run_time = t_1 - t_0
                        if self.safety_deltas:
                            bound = soft_spibb.get_advantage(self.initial_state) - 1 / (1 - self.gamma) * epsilon
                            self.results.append(
                                self.to_append + [method, hyperparam, method_perf, run_time, self.nb_states, delta, bound])
                        else:
                            self.results.append(self.to_append + [method, hyperparam, method_perf, run_time, self.nb_states])

    def _run_basic_rl(self, key):
        """
        Runs Basic RL for one data set.
        :param key: BasicRL.NAME
        """
        basic_rl = algorithm_name_dict[key](pi_b=self.pi_b, gamma=self.gamma, nb_states=self.nb_states,
                                            nb_actions=self.nb_actions, data=self.data, R=self.R_state_state,
                                            episodic=self.episodic, speed_up_dict=self.speed_up_dict, estimate_baseline=self.estimate_baseline)
        t_0 = time.time()
        basic_rl.fit()
        t_1 = time.time()
        basic_rl_perf = self._policy_evaluation_exact(basic_rl.pi)
        method = basic_rl.NAME
        method_perf = basic_rl_perf
        hyperparam = None
        run_time = t_1 - t_0
        self.results.append(self.to_append + [method, hyperparam, method_perf, run_time, self.nb_states])
        
    

    def _policy_evaluation_exact(self, pi):
        """
        Evaluates policy pi exactly.
        :param pi: policy as numpy array with shape (nb_states, nb_actions)
        """
        return policy_evaluation_exact(pi, self.R_state_action, self.P, self.gamma)[0][self.initial_state]
    
    def array_to_dict(self, arr):
        sparse_dict = {}

        # Iterate over the 2D array and add non-zero elements to the dictionary
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                    if arr[i, j] != 0:
                        sparse_dict[(i, j)] = arr[i, j]
        return sparse_dict
    
    def compute_r_state_action(self, P, R):
        if isinstance(P, dict):
            print(type(R))
            if isinstance(R, np.ndarray):
                R = self.array_to_dict(R)
            return self.compute_r_state_action_sparse(P, R)
        return self.compute_r_state_action_dense(P, R)
    
    def compute_r_state_action_sparse(self, P, R):
            result = defaultdict(float)

            for (i, j, k), p_val in P.items():
                r_val = R.get((i, k), 0.0)
                result[(i, j)] += p_val * r_val

            # Convert result to dense NumPy array

            dense_result = np.zeros((self.nb_states, self.nb_actions))

            for (i, j), val in result.items():
                dense_result[i, j] = val

            return dense_result
        
    def compute_r_state_action_dense(self, P, R):
        return np.einsum('ijk,ik->ij', P, R)

    

class MovingObstaclesExperiment(Experiment):
    fixed_params_exp_columns = ['seed', 'gamma']
    def _set_env_params(self):
        """
        Reads in all parameters necessary from self.experiment_config to set up the Wet Chicken experiment.
        """
        self.generate_gif = False
        self.episodic = True
        self.gamma = float(self.experiment_config['ENV_PARAMETERS']['GAMMA'])
        self.env_name = str(self.experiment_config['META']['env_name'])
        
        print("start env")
        self.env = MovingObstacles()
        self.render_env = None
        print("get values")
        self.nb_states = self.env.get_nb_states()
        self.nb_actions = self.env.get_nb_actions()
        

        self.traps = self.env.get_traps()
        self.goal = self.env.get_goal_state()

        self.initial_state_cont, self.initial_state = self.env.get_init_state()
        
        self.R_state_state = self.env.get_reward_function()

        
        self.fixed_params_exp_list = [self.seed, self.gamma]
        self.dimensions = 4



        self.epsilons_baseline = ast.literal_eval(self.experiment_config['BASELINE']['epsilons_baseline'])
        
        self.pi_b = MovingObstaclesPolicy(self.env, epsilon=self.epsilons_baseline[0]).pi
        self.nb_trajectories_list = ast.literal_eval(self.experiment_config['BASELINE']['nb_trajectories_list'])
        self.variable_params_exp_columns = ['i', 'epsilon_baseline', 'pi_b_perf', 'length_trajectory']

        self.estimate_baseline=bool((util.strtobool(self.experiment_config['ENV_PARAMETERS']['estimate_baseline'])))
        
    def _run_one_iteration(self):
        print("Starting run")
        for epsilon_baseline in self.epsilons_baseline:
            print(f'Process with seed {self.seed} starting with epsilon_baseline {epsilon_baseline} out of'
                  f' {self.epsilons_baseline}')
            
            print("creating Baseline Policy")
            self.pi_b = MovingObstaclesPolicy(self.env, epsilon=epsilon_baseline).pi
            # self.to_append_run_one_iteration = self.to_append_run + [epsilon_baseline,
            #                                                             self._policy_evaluation_exact(self.pi_b)]
            self.to_append_run_one_iteration = self.to_append_run + [epsilon_baseline,
                                                                        0]
            for nb_trajectories in self.nb_trajectories_list:
                print(
                    f'Process with seed {self.seed} starting with nb_trajectories {nb_trajectories} out of '
                    f'{self.nb_trajectories_list}')
                # Generate trajectories, both stored as trajectories and (s,a,s',r) transition samples
                print("Generating Trajectories")
                # generate data on the real cartpole environment. Translate this data to the partitioning in generate_batch
                self.pi_b = MovingObstaclesPolicy(self.env, epsilon=epsilon_baseline).pi
                data_grid, batch_traj, self.data_cont = self.generate_batch(nb_trajectories, self.env, self.pi_b)
                self.to_append = self.to_append_run_one_iteration + [nb_trajectories]
                
                self.data_df = trajToDF(self.data_cont, self.dimensions, 1)
                # self.data_df.to_csv("data_set.csv", index=False)
                # print(len(self.data_df.index))
                # self.data_df = pd.read_csv("data_set.csv")
                # print(self.data_cont)
                # with pd.option_context('display.max_rows', None,
                #        'display.max_columns', None):
                #     print(self.data_df)
                print("getting abstraction")
                
                # ------------------------ MRL --------------------------
                self.discretization_method = 'mrl'
                
                # Get discretization
                m = MDP_model()
                m.fit(
                    self.data_df,
                    pfeatures=self.dimensions,
                    h = -1,
                    gamma = 1,
                    max_k = 100,
                    distance_threshold=0.5,
                    th = 10,
                    eta = 25,
                    precision_thresh = -1e-14,
                    classification = 'DecisionTreeClassifier',
                    split_classifier_params = {'random_state':0, 'max_depth':5},
                    clustering = 'KMeans',
                    n_clusters = 3,
                    random_state = 0,
                    plot=False,
                    verbose=False,
                    stochastic=True
                )
                print("Trained the model!!")
                
                # discretize data
                self.predictor = predict_cluster(m.df_trained, self.dimensions)
                # d_data = self.discretize_data(self.data_cont, self.predictor)
                d_data = self.discretize_data_from_df(self.data_cont, m.df_trained)
                self.data = d_data
                nb_states = m.df_trained["CLUSTER"].max()+1
                self.nb_states = nb_states
                print("nb states = ", nb_states)                
                # get discrete reward function
                self.R_state_state = np.zeros((nb_states, nb_states))
                traps = []
                goal = []
                for state in range(len(self.R_state_state)):
                    try:
                        r = m.R_df[state]
                    except:
                        r = 0.0
                    self.R_state_state[:, state] = r
                    if r < 0.0:
                        traps.append(state)
                    if r > 0.0:
                        goal.append(state)
                

                # get structure transition function
                self.structure = self.get_empty_structure(nb_states)
                self.structure = self.add_trans_from_data(self.structure, d_data)
                self._count(d_data)
                self._build_model()           
                self.R_s_a = self.compute_r_state_action(self.transition_model, self.R_state_state)
                self.initial_state = d_data[0][0][1]
                
                # Calculate Shield                
                self.estimator = PACIntervalEstimator(self.structure, 0.1, d_data, self.nb_actions, alpha=5)
                self.estimator.calculate_intervals()
                self.intervals = self.estimator.get_intervals()                
                
                print("Calculating Shield") 
                # print(m.R_df) 
                self.shielder = ShieldMovingObstacles(self.structure, traps, goal, self.intervals, self.initial_state)
                self.shielder.calculateShield()
                # self.shielder.printShield()
                
                # # Run the algoirhtm
                self.pi_b = MovingObstaclesPolicy(self.env, epsilon=epsilon_baseline).compute_baseline_size(nb_states)

                print("Running Algorithms")
                self._run_algorithms()
                
                
                # ----------------------------- GRID ---------------------------------
                self.discretization_method = 'grid'
                self.pi_b = MovingObstaclesPolicy(self.env, epsilon=epsilon_baseline).pi
                self.initial_state_cont, self.initial_state = self.env.get_init_state()
                print("The baseline has length:", len(self.pi_b))
                self.nb_states = self.env.get_nb_states()
                self.data = data_grid
                self.R_state_state = self.env.get_reward_function()
                
                print("Estimating Intervals")            
                self._count(self.data)
                self._build_model()
                self.R_s_a = self.compute_r_state_action(self.transition_model, self.R_state_state)
                self.structure = self._tm_to_next_states()
                self.estimator = PACIntervalEstimator(self.structure, 0.1, self.data, self.nb_actions, alpha=5)
                self.estimator.calculate_intervals()
                self.intervals = self.estimator.get_intervals()   
                # self.estimator = imdp_builder(self.data, self.count_state_action_state, self.count_state_action, self.episodic, beta=1e-4, kstep=1)
                # self.intervals = self.estimator.get_intervals()
                
                
                print("Calculating Shield")  
                # self.structure = self.build_transition_matrix()
                self.shielder = ShieldMovingObstacles(self.structure, [self.traps], [self.goal], self.intervals, self.initial_state)
                self.shielder.calculateShield()
                # self.shielder.printShield()
                print("Running Algorithms")
                self._run_algorithms()
                # ----------------------------- GreedyCut ----------------------------------
                self.data = self.data_cont
                self.discretization_method = 'GreedyCut'
                bounds = [(0, 1),(0, 1),(-0.2, 0.2),(-0.2, 0.2)]
                self.discretizer = GreedyCut(
                    state_dim = self.dimensions,
                    trajectories=self.data,
                    B=10,
                    bounds=bounds,
                    initial_splits=[4,4,4,4],
                    binary_dims=[]
                )
                self.predictor = self.discretizer.Greedy()
                print(self.predictor)
                d_data = self.discretizer.get_discretized_dataset()
                self.data = d_data
                with open("trajectories.txt", "w") as f:
                    for i, traj in enumerate(d_data):
                        f.write(f"Trajectory {i}:\n")
                        for transition in traj:
                            action, state, next_state, reward = transition
                            f.write(f"{action}, {state}, {next_state}, {reward}\n")
                        f.write("\n")
                        
                with open("trajectories_grid.txt", "w") as f:
                    for i, traj in enumerate(data_grid):
                        f.write(f"Trajectory {i}:\n")
                        for transition in traj:
                            action, state, next_state, reward = transition
                            f.write(f"{action}, {state}, {next_state}, {reward}\n")
                        f.write("\n")
                nb_states = self.discretizer.get_num_regions(self.predictor)
                self.nb_states = nb_states
                print("nb_states = ", nb_states)
                # get structure transition function
                self.structure = self.get_empty_structure(nb_states)
                self.structure = self.add_trans_from_data(self.structure, d_data)
                self._count(d_data)
                
                self._build_model()           
                self.R_s_a = self.compute_r_state_action(self.transition_model, self.R_state_state)
                self.initial_state = d_data[0][0][1]
                
                # Calculate Shield                
                self.estimator = PACIntervalEstimator(self.structure, 0.1, d_data, self.nb_actions, alpha=5)
                self.estimator.calculate_intervals()
                self.intervals = self.estimator.get_intervals()                
                
                print("Calculating Shield") 
                # print(m.R_df) 
                goal = []
                traps = []
                self.R_state_state = np.zeros((nb_states, nb_states))
                for traj in d_data:
                    final_transition = traj[-1]
                    s, a, s_next, r = final_transition
                    if  r<0 and s_next not in traps:
                        traps.append(s_next)
                        self.R_state_state[s,s_next]=-10
                    elif r>0 and s_next not in goal:
                        goal.append(s_next)
                        self.R_state_state[s,s_next]=10
                print(traps)
                print(goal)
                # option 2: Use the known parameters to set the traps based on the centres. Any state for which the centre is outside the
                self.shielder = ShieldMovingObstacles(self.structure, traps, goal, self.intervals, self.initial_state)
                self.shielder.calculateShield()
                # self.shielder.printShield()
                
                # # Run the algoirhtm
                self.pi_b = MovingObstaclesPolicy(self.env, epsilon=epsilon_baseline).compute_baseline_size(nb_states)
                print("Running Algorithms")
                self._run_algorithms()
                
                
                # ----------------------------- SPIBB-DQN ----------------------------------
                self.discretization_method = 'SPIBB-DQN'
                self.pi_b = MovingObstaclesPolicy(self.env, epsilon=epsilon_baseline).pi
                self.data = self.data_cont
                self._run_spibb_dqn('SPIBB-DQN')
                
                # ----------------------------- CQL-DQN ----------------------------------
                self.discretization_method = 'CQL-DQN'
                self.data = self.data_cont
                self._run_cql_dqn()
         
        
    def generate_batch(self, nb_trajectories, env, pi, max_steps=250):
        """
        Generates a data batch for an episodic MDP.
        :param nb_steps: number of steps in the data batch
        :param env: environment to be used to generate the batch on
        :param pi: policy to be used to generate the data as numpy array with shape (nb_states, nb_actions)
        :return: data batch as a list of sublists of the form [state, action, next_state, reward]
        """
        trajectories = []
        trajectories_cont = []
        action_counts = [0,0,0,0]
        for i in np.arange(nb_trajectories):
            # if i % 100 == 0:
            # print(i)
            nb_steps = 0
            trajectorY = []
            trajectorY_cont = []
            env.reset()
            state, region = env.get_init_state()
            is_done = False
            while nb_steps < max_steps and not is_done:
                # print("AAAAA")
                nb_steps += 1
                action_choice = np.random.choice(pi.shape[1], p=pi[region])
                action_counts[action_choice]+=1
                state, next_state, reward = env.step(action_choice)
                # print(state)
                # print(type(state))
                region = env.state2region(state)
                next_region = env.state2region(next_state)
                is_done = env.is_done()                    
                trajectorY.append([action_choice, region, next_region, reward])
                terminated = is_done
                truncated = nb_steps>=max_steps
                trajectorY_cont.append([state, action_choice, next_state, reward, terminated, truncated])
                region = next_region
                
            trajectories_cont.append(trajectorY_cont)
            trajectories.append(trajectorY)
            # print(nb_steps)
        batch_traj = [val for sublist in trajectories for val in sublist]
        print("counts = ",  action_counts)
        return trajectories, batch_traj, trajectories_cont
            
    
    def _count(self, data):
        """
        Counts the state-action pairs and state-action-triplets and stores them.
        """
        if self.episodic:
            batch_trajectory = [val for sublist in data for val in sublist]
        else:
            batch_trajectory = data.copy()
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
            
    def _tm_to_next_states(self):
        structure = np.empty((self.nb_states, self.nb_actions), dtype=object)
        for s in range(self.nb_states):
            for a in range(self.nb_actions):
                structure[s, a] = []

        # Populate from sparse transition model
        for (s, a, s_prime), prob in self.transition_model.items():
            if prob > 0:
                # print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa")
                structure[s, a].append(s_prime)
        
        # If a state has no successors in the data, just map to itself
        for s in range(self.nb_states):
            for a in range(self.nb_actions):
                if len(structure[s, a]) == 0:
                    structure[s, a].append(s)
            
        return structure
    
    def estimate_transitions(self):
        count = 0
        # Prepare the reduced matrix with empty lists
        transition_matrix = np.empty((self.nb_states, self.nb_actions), dtype=object)
        for s in range(self.nb_states):
            for a in range(self.nb_actions):
                if s == self.traps:
                    transition_matrix[s, a] = [self.traps]
                else:
                    print(s, " = ", self.env.get_successor_states(s,a))
                    transition_matrix[s, a] = list(self.env.get_successor_states(s,a))
        self.transition_matrix = transition_matrix
        
    def discretize_data(self, data, predictor):
        data_disc = []
        for trajectory in data:
            traj = []
            for transition in trajectory:
                s = transition[0]
                a = transition[1]
                ns = transition[2]
                r = transition[3]
                terminated = transition[4]
                died = transition[5]
                s_d = state2region(predictor, s, self.dimensions)
                ns_d = state2region(predictor, ns, self.dimensions)
                traj.append([a, s_d, ns_d, r])
            data_disc.append(traj)
        return data_disc
                             
    def discretize_data_from_df(self, data, df):
        """
        Discretizes trajectories using cluster assignments directly from a dataframe.

        Parameters
        ----------
        data : list
            trajectories_cont produced by generate_batch
            [state, action, next_state, reward, terminated, truncated]

        df : pandas.DataFrame
            dataframe containing FEATURE_0..FEATURE_7, CLUSTER, NEXT_CLUSTER

        Returns
        -------
        data_disc : list
            discretized trajectories in the form
            [action, state_region, next_state_region, reward]
        """

        feature_cols = [f"FEATURE_{i}" for i in range(self.dimensions)]

        # --- build lookup dictionary much faster ---
        features = df[feature_cols].to_numpy()
        clusters = df["CLUSTER"].to_numpy()

        state_to_cluster = {tuple(features[i]): clusters[i] for i in range(len(features))}

        data_disc = []

        for trajectory in data:
            traj = []

            for s, a, ns, r, terminated, truncated in trajectory:
                s_region = state_to_cluster.get(tuple(s))
                ns_region = state_to_cluster.get(tuple(ns))

                if s_region is None or ns_region is None:
                    raise ValueError(
                        f"State {tuple(s)} and next_state {tuple(ns)} not found in lookup."
                    )

                traj.append([a, s_region, ns_region, r])

            data_disc.append(traj)

        return data_disc
    
    def get_empty_structure(self, nb_states):
        empty_structure = np.empty((nb_states, self.nb_actions), dtype=object)                      
        
        for state in range(nb_states):
            for action in range(self.nb_actions):
                empty_structure[state, action] = np.array([], dtype=int)
                
        return empty_structure            
    def add_trans_from_data(self,structure, data):
        for trajectory in data:
            for transition in trajectory:
                s=transition[1]
                a=transition[0]
                ns=transition[2]
                poss_next = structure[s,a]
                if not ns in poss_next:
                    poss_next = np.append(poss_next, [ns])
                    structure[s][a] = poss_next
        print("first element of structure = ", structure[0,0])
        return structure  
    
       
    def build_transition_matrix(self):
        """
        Builds a reduced transition matrix that lists possible next states
        for each (state, action) pair, based on the observed trajectories.

        Returns
        -------
        np.ndarray
            A 2D array of shape (num_states, num_actions), where each entry
            is a list of possible next states for that (state, action).
        """
        count = 0
        # Prepare the reduced matrix with empty lists
        transition_matrix = self.transition_matrix.copy()


        # Fill matrix with next states from counts
        for (state, action, next_state) in self.count_state_action_state.keys():
            if next_state not in transition_matrix[state, action]:
                transition_matrix[state, action].append(next_state)

        # for s in range(self.nb_states):
        #     for a in range(self.nb_actions):
        #         if len(transition_matrix[s, a]) == 0:
        #             transition_matrix[s, a] = [self.traps]

        # for i in range(len(transition_matrix)):
        #     for j in range(len(transition_matrix[i])):
        #         if len(transition_matrix[i][j]) > 1:
        #             count+=1
        # print(transition_matrix)
        # print(count)
        return transition_matrix   


class GymCartPoleExperiment(Experiment):
    fixed_params_exp_columns = ['seed', 'gamma']
    def _set_env_params(self):
        """
        Reads in all parameters necessary from self.experiment_config to set up the Wet Chicken experiment.
        """
        self.generate_gif = True
        self.episodic = True
        self.gamma = float(self.experiment_config['ENV_PARAMETERS']['GAMMA'])
        self.env_name = str(self.experiment_config['META']['env_name'])
        
        print("start env")
        self.env = cartPole()
        self.render_env = cartPole(True)
        print("get values")
        self.nb_states = self.env.get_nb_states()
        self.nb_actions = self.env.get_nb_actions()
        

        self.traps = self.env.get_traps()
        self.goal = self.env.get_goal_state()

        self.initial_state_cont, self.initial_state = self.env.get_init_state()
        
        # self.P = self.env.get_transition_function()
        self.R_state_state = self.env.get_reward_function()

        # print("calcing r_sa")
        # self.R_state_action = self.compute_r_state_action(self.P, self.R_state_state)
        
        self.fixed_params_exp_list = [self.seed, self.gamma]
        self.dimensions = 4
        # pi_rand = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        # print("calcing rand perf")
        # pi_rand_perf = self._policy_evaluation_exact(pi_rand)
        # print(f"pi_rand_perf = {pi_rand_perf}")

        # self.fixed_params_exp_list.append(pi_rand_perf)

        # pi_star = PiStar(pi_b=None, gamma=self.gamma, nb_states=self.nb_states, nb_actions=self.nb_actions,
        #                  data=[[]], R=self.R_state_state, episodic=self.episodic, P=self.P)
        # pi_star.fit()
        # pi_star_perf = self._policy_evaluation_exact(pi_star.pi)
        # print(f"pi_star_perf = {pi_star_perf}")
        # self.fixed_params_exp_list.append(pi_star_perf)


        self.epsilons_baseline = ast.literal_eval(self.experiment_config['BASELINE']['epsilons_baseline'])
        
        # pi_base_perf = self._policy_evaluation_exact(self.env.get_baseline_policy(self.epsilons_baseline[0]))
        # print(self.env.get_baseline_policy(self.epsilons_baseline[0]))
        # print(f"pi_baseline_perf = {pi_base_perf}")
        self.pi_b = cartPolePolicy(self.env, epsilon=self.epsilons_baseline[0]).pi
        # pi_base_perf = evaluate_policy(self.env, self.pi_b, 1, 100)
        # print(pi_base_perf)
        self.nb_trajectories_list = ast.literal_eval(self.experiment_config['BASELINE']['nb_trajectories_list'])
        self.variable_params_exp_columns = ['i', 'epsilon_baseline', 'pi_b_perf', 'length_trajectory']

        self.estimate_baseline=bool((util.strtobool(self.experiment_config['ENV_PARAMETERS']['estimate_baseline'])))
        print("estimating transitions")
        # self.estimate_transitions()
        
    def _run_one_iteration(self):
        for epsilon_baseline in self.epsilons_baseline:
            print(f'Process with seed {self.seed} starting with epsilon_baseline {epsilon_baseline} out of'
                  f' {self.epsilons_baseline}')
            
            print("creating Baseline Policy")
            self.pi_b = cartPolePolicy(self.env, epsilon=epsilon_baseline).pi
            # self.to_append_run_one_iteration = self.to_append_run + [epsilon_baseline,
            #                                                             self._policy_evaluation_exact(self.pi_b)]
            self.to_append_run_one_iteration = self.to_append_run + [epsilon_baseline,
                                                                        0]
            for nb_trajectories in self.nb_trajectories_list:
                print(
                    f'Process with seed {self.seed} starting with nb_trajectories {nb_trajectories} out of '
                    f'{self.nb_trajectories_list}')
                # Generate trajectories, both stored as trajectories and (s,a,s',r) transition samples
                print("Generating Trajectories")
                # generate data on the real cartpole environment. Translate this data to the partitioning in generate_batch
                self.pi_b = cartPolePolicy(self.env, epsilon=epsilon_baseline).pi
                data_grid, batch_traj, self.data_cont = self.generate_batch(nb_trajectories, self.env, self.pi_b)
                self.to_append = self.to_append_run_one_iteration + [nb_trajectories]
                
                self.data_df = trajToDF(self.data_cont, self.dimensions, 1)
                self.data_df.to_csv("data_set.csv", index=False)
                print(len(self.data_df.index))
                # self.data_df = pd.read_csv("data_set.csv")
                # print(self.data_cont)
                # with pd.option_context('display.max_rows', None,
                #        'display.max_columns', None):
                #     print(self.data_df)
                print("getting abstraction")
                # Get the intervals from the abstraction using the method from badings
                # self._count()
                # # print("done counts")
                # self.estimator = imdp_builder(self.data, self.count_state_action_state, self.count_state_action, self.episodic, beta=1e-4, kstep=1)
                # self.intervals = self.estimator.get_intervals()
                
                # ------------------------ MRL --------------------------
                self.discretization_method = 'mrl'
                
                # Get discretization
                m = MDP_model()
                m.fit(
                    self.data_df,
                    pfeatures=self.dimensions,
                    h = -1,
                    gamma = 1,
                    max_k = 100,
                    distance_threshold=0.5,
                    th = 10,
                    eta = 25,
                    precision_thresh = -1, #1e-14
                    classification = 'DecisionTreeClassifier',
                    split_classifier_params = {'random_state':0, 'max_depth':10},
                    clustering = 'Agglomerative',
                    n_clusters = None,
                    random_state = 0,
                    plot=False,
                    verbose=False
                )
                print("Trained the model!!")
                
                # discretize data
                self.predictor = predict_cluster(m.df_trained, self.dimensions)
                d_data = self.discretize_data(self.data_cont, self.predictor)
                self.data = d_data
                nb_states = m.df_trained["CLUSTER"].nunique()
                self.nb_states = nb_states
                print("nb states = ", nb_states)                
                # get discrete reward function
                self.R_state_state = np.zeros((nb_states, nb_states))
                traps = []
                goal = []
                for state in range(len(self.R_state_state)):
                    r = m.R_df[state]
                    self.R_state_state[:, state] = r
                    if r == 0.0:
                        traps.append(state)

                # get structure transition function
                self.structure = self.get_empty_structure(nb_states)
                self.structure = self.add_trans_from_data(self.structure, d_data)
                self._count(d_data)
                self._build_model()           
                self.R_s_a = self.compute_r_state_action(self.transition_model, self.R_state_state)
                self.initial_state = d_data[0][0][1]
                
                # Calculate Shield                
                self.estimator = PACIntervalEstimator(self.structure, 0.1, d_data, self.nb_actions, alpha=5)
                self.estimator.calculate_intervals()
                self.intervals = self.estimator.get_intervals()                
                
                print("Calculating Shield") 
                # print(m.R_df) 
                self.shielder = ShieldCartpole(self.structure, traps, goal, self.intervals, self.initial_state)
                self.shielder.calculateShield()
                # self.shielder.printShield()
                
                # # Run the algoirhtm
                self.pi_b = cartPolePolicy(self.env, epsilon=epsilon_baseline).compute_baseline_size(nb_states)

                print("Running Algorithms")
                self._run_algorithms()
                
                
                # ----------------------------- GRID ---------------------------------
                self.discretization_method = 'grid'
                self.pi_b = cartPolePolicy(self.env, epsilon=epsilon_baseline).pi
                self.initial_state_cont, self.initial_state = self.env.get_init_state()
                print("The baseline has length:", len(self.pi_b))
                self.nb_states = self.env.get_nb_states()
                self.data = data_grid
                self.R_state_state = self.env.get_reward_function()
                
                print("Estimating Intervals")            
                self._count(self.data)
                self._build_model()
                self.R_s_a = self.compute_r_state_action(self.transition_model, self.R_state_state)
                self.structure = self._tm_to_next_states()
                self.estimator = PACIntervalEstimator(self.structure, 0.1, self.data, self.nb_actions, alpha=5)
                self.estimator.calculate_intervals()
                self.intervals = self.estimator.get_intervals()   
                # self.estimator = imdp_builder(self.data, self.count_state_action_state, self.count_state_action, self.episodic, beta=1e-4, kstep=1)
                # self.intervals = self.estimator.get_intervals()
                
                
                print("Calculating Shield")  
                # self.structure = self.build_transition_matrix()
                self.shielder = ShieldCartpole(self.structure, [self.traps], self.goal, self.intervals, self.initial_state)
                self.shielder.calculateShield()
                # self.shielder.printShield()
                print("Running Algorithms")
                self._run_algorithms()
                # ----------------------------- GreedyCut ----------------------------------
                self.data = self.data_cont
                self.discretization_method = 'GreedyCut'
                bounds = [(-4.8, 4.8),(-3.0, 3.0),(-0.41887903, 0.41887903),(-3.5, 3.5)]
                self.discretizer = GreedyCut(
                    state_dim = self.dimensions,
                    trajectories=self.data,
                    B=10,
                    bounds=bounds
                )
                self.predictor = self.discretizer.Greedy()
                print(self.predictor)
                d_data = self.discretizer.get_discretized_dataset()
                self.data = d_data
                with open("trajectories.txt", "w") as f:
                    for i, traj in enumerate(d_data):
                        f.write(f"Trajectory {i}:\n")
                        for transition in traj:
                            action, state, next_state, reward = transition
                            f.write(f"{action}, {state}, {next_state}, {reward}\n")
                        f.write("\n")
                        
                with open("trajectories_grid.txt", "w") as f:
                    for i, traj in enumerate(data_grid):
                        f.write(f"Trajectory {i}:\n")
                        for transition in traj:
                            action, state, next_state, reward = transition
                            f.write(f"{action}, {state}, {next_state}, {reward}\n")
                        f.write("\n")
                nb_states = self.discretizer.get_num_regions(self.predictor)
                self.nb_states = nb_states
                print("nb_states = ", nb_states)
                # get structure transition function
                self.structure = self.get_empty_structure(nb_states)
                self.structure = self.add_trans_from_data(self.structure, d_data)
                self._count(d_data)
                
                self._build_model()           
                self.R_s_a = self.compute_r_state_action(self.transition_model, self.R_state_state)
                self.initial_state = d_data[0][0][1]
                
                # Calculate Shield                
                self.estimator = PACIntervalEstimator(self.structure, 0.1, d_data, self.nb_actions, alpha=5)
                self.estimator.calculate_intervals()
                self.intervals = self.estimator.get_intervals()                
                
                print("Calculating Shield") 
                # print(m.R_df) 
                goal = []
                # option 1: Any region that has at least one trap is considered to be a trap state
                traps = []
                self.R_state_state = np.ones((nb_states, nb_states))
                for traj in d_data:
                    final_transition = traj[-1]
                    s, a, s_next, r = final_transition
                    if  r==0 and s_next not in traps:
                        traps.append(s_next)
                        self.R_state_state[s,s_next]=0
                print(traps)
                # option 2: Use the known parameters to set the traps based on the centres. Any state for which the centre is outside the
                self.shielder = ShieldCartpole(self.structure, traps, goal, self.intervals, self.initial_state)
                self.shielder.calculateShield()
                # self.shielder.printShield()
                
                # # Run the algoirhtm
                self.pi_b = cartPolePolicy(self.env, epsilon=epsilon_baseline).compute_baseline_size(nb_states)
                print("Running Algorithms")
                self._run_algorithms()
                

                
                # ----------------------------- SPIBB-DQN ----------------------------------
                self.discretization_method = 'SPIBB-DQN'
                self.pi_b = cartPolePolicy(self.env, epsilon=epsilon_baseline).pi
                self.data = self.data_cont
                self._run_spibb_dqn('SPIBB-DQN')
                
                # ----------------------------- CQL-DQN ----------------------------------
                self.discretization_method = 'CQL-DQN'
                self.data = self.data_cont
                self._run_cql_dqn()
         
        
    def generate_batch(self, nb_trajectories, env, pi, max_steps=1000):
        """
        Generates a data batch for an episodic MDP.
        :param nb_steps: number of steps in the data batch
        :param env: environment to be used to generate the batch on
        :param pi: policy to be used to generate the data as numpy array with shape (nb_states, nb_actions)
        :return: data batch as a list of sublists of the form [state, action, next_state, reward]
        """
        trajectories = []
        trajectories_cont = []
        for _ in np.arange(nb_trajectories):
            nb_steps = 0
            trajectorY = []
            trajectorY_cont = []
            env.reset()
            state, region = env.get_init_state()
            is_done = False
            while nb_steps < max_steps and not is_done:
                # print("AAAAA")
                action_choice = np.random.choice(pi.shape[1], p=pi[region])
                state, next_state, reward = env.step(action_choice)
                # print(state)
                # print(type(state))
                region = env.state2region(state)
                next_region = env.state2region(next_state)
                is_done = env.is_done()                    
                trajectorY.append([action_choice, region, next_region, reward])
                terminated = env.is_terminated()
                truncated = env.is_truncated()
                trajectorY_cont.append([state, action_choice, next_state, reward, terminated, truncated])
                region = next_region
                nb_steps += 1
            trajectories_cont.append(trajectorY_cont)
            trajectories.append(trajectorY)
        batch_traj = [val for sublist in trajectories for val in sublist]
        return trajectories, batch_traj, trajectories_cont
            
    
    def _count(self, data):
        """
        Counts the state-action pairs and state-action-triplets and stores them.
        """
        if self.episodic:
            batch_trajectory = [val for sublist in data for val in sublist]
        else:
            batch_trajectory = data.copy()
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
            
    def _tm_to_next_states(self):
        structure = np.empty((self.nb_states, self.nb_actions), dtype=object)
        for s in range(self.nb_states):
            for a in range(self.nb_actions):
                structure[s, a] = []

        # Populate from sparse transition model
        for (s, a, s_prime), prob in self.transition_model.items():
            if prob > 0:
                # print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa")
                structure[s, a].append(s_prime)
        
        # If a state has no successors in the data, just map to itself
        for s in range(self.nb_states):
            for a in range(self.nb_actions):
                if len(structure[s, a]) == 0:
                    structure[s, a].append(s)
            
        return structure
    
    def estimate_transitions(self):
        count = 0
        # Prepare the reduced matrix with empty lists
        transition_matrix = np.empty((self.nb_states, self.nb_actions), dtype=object)
        for s in range(self.nb_states):
            for a in range(self.nb_actions):
                if s == self.traps:
                    transition_matrix[s, a] = [self.traps]
                else:
                    print(s, " = ", self.env.get_successor_states(s,a))
                    transition_matrix[s, a] = list(self.env.get_successor_states(s,a))
        self.transition_matrix = transition_matrix
        
    def discretize_data(self, data, predictor):
        data_disc = []
        for trajectory in data:
            traj = []
            for transition in trajectory:
                s = transition[0]
                a = transition[1]
                ns = transition[2]
                r = transition[3]
                terminated = transition[4]
                died = transition[5]
                s_d = state2region(predictor, s, self.dimensions)
                ns_d = state2region(predictor, ns, self.dimensions)
                traj.append([a, s_d, ns_d, r])
            data_disc.append(traj)
        return data_disc
                             
    def get_empty_structure(self, nb_states):
        empty_structure = np.empty((nb_states, self.nb_actions), dtype=object)                      
        
        for state in range(nb_states):
            for action in range(self.nb_actions):
                empty_structure[state, action] = np.array([], dtype=int)
                
        return empty_structure            
    def add_trans_from_data(self,structure, data):
        for trajectory in data:
            for transition in trajectory:
                s=transition[1]
                a=transition[0]
                ns=transition[2]
                poss_next = structure[s,a]
                if not ns in poss_next:
                    poss_next = np.append(poss_next, [ns])
                    structure[s][a] = poss_next
        print("first element of structure = ", structure[0,0])
        return structure  
    
    def reduce_transition_matrix(self, transition_matrix):
        """
        Reduces a transition matrix to only include possible end states for each state-action pair.

        Args:
        - transition_matrix (numpy.ndarray): A 3D numpy array of shape (num_states, num_actions, num_states) 
        where each element represents the probability of transitioning from one state to another
        given a certain action.

        Returns:
        - numpy.ndarray: A 3D numpy array of shape (num_states, num_actions, num_possible_transitions) 
        where each element contains the indices of possible end states.
        """
        num_states = len(transition_matrix)
        num_actions = len(transition_matrix[0])
        # Prepare the reduced matrix to hold the indices of possible states
        reduced_matrix = np.empty((num_states, num_actions), dtype=object)
        
        # Loop through each state and action to populate the reduced matrix
        for state in range(num_states):
            for action in range(num_actions):
                # Get indices of nonzero probabilities (possible end states)
                possible_states = np.nonzero(transition_matrix[state, action])[0]
                reduced_matrix[state, action] = np.array(possible_states)
        
        return reduced_matrix
       
    def build_transition_matrix(self):
        """
        Builds a reduced transition matrix that lists possible next states
        for each (state, action) pair, based on the observed trajectories.

        Returns
        -------
        np.ndarray
            A 2D array of shape (num_states, num_actions), where each entry
            is a list of possible next states for that (state, action).
        """
        count = 0
        # Prepare the reduced matrix with empty lists
        transition_matrix = self.transition_matrix.copy()


        # Fill matrix with next states from counts
        for (state, action, next_state) in self.count_state_action_state.keys():
            if next_state not in transition_matrix[state, action]:
                transition_matrix[state, action].append(next_state)

        # for s in range(self.nb_states):
        #     for a in range(self.nb_actions):
        #         if len(transition_matrix[s, a]) == 0:
        #             transition_matrix[s, a] = [self.traps]

        # for i in range(len(transition_matrix)):
        #     for j in range(len(transition_matrix[i])):
        #         if len(transition_matrix[i][j]) > 1:
        #             count+=1
        # print(transition_matrix)
        # print(count)
        return transition_matrix
class RewardDict(Mapping):
    def __init__(self, nb_states, terminal_states, goal_states, reward_per_next):
        self.nb_states = nb_states
        self.terminal_idx = terminal_states
        self.goal_idx = goal_states
        self.reward_per_next = reward_per_next
            
    def __getitem__(self, key):
        state, next_state = key

        if next_state in self.goal_idx:
            return 100
        if next_state in self.terminal_idx:
            return -100

        return self.reward_per_next.get(next_state, 0.0)

    def __iter__(self):
        raise RuntimeError("Full iteration over N^2 reward table is infeasible.")

    def __len__(self):
        return self.nb_states * self.nb_states
     
class GymLunarLanderExperiment(Experiment):
    fixed_params_exp_columns = ['seed', 'gamma']
    def _set_env_params(self):
        """
        Reads in all parameters necessary from self.experiment_config to set up the Wet Chicken experiment.
        """
        self.generate_gif = True

        self.env_name = str(self.experiment_config['META']['env_name'])
        self.episodic = True
        self.gamma = float(self.experiment_config['ENV_PARAMETERS']['GAMMA'])
        
        
        
        self.fixed_params_exp_list = [self.seed, self.gamma]
        self.dimensions = 8


        self.epsilons_baseline = ast.literal_eval(self.experiment_config['BASELINE']['epsilons_baseline'])
        
        self.nb_trajectories_list = ast.literal_eval(self.experiment_config['BASELINE']['nb_trajectories_list'])
        self.variable_params_exp_columns = ['i', 'epsilon_baseline', 'pi_b_perf', 'length_trajectory']

        self.estimate_baseline=bool((util.strtobool(self.experiment_config['ENV_PARAMETERS']['estimate_baseline'])))
        print("estimating transitions")
        # self.estimate_transitions()
        
    def _run_one_iteration(self):
        print("start env")
        seed = np.random.randint(0, 1_000_000)
        self.env = LunarLander(seed=seed)
        self.render_env = LunarLander(seed=seed, render_mode=True)
        print("get values")
        self.R_state_state_grid = self.env.get_reward_function() # make sure this runs before get_traps and get_goal_states
        self.nb_states = self.env.get_nb_states()
        self.nb_actions = self.env.get_nb_actions()
        self.traps = [self.env.get_traps()]
        self.goal = [self.env.get_goal_state()]

        self.initial_state_cont, self.initial_state = self.env.get_init_state()
        
        
        
        for epsilon_baseline in self.epsilons_baseline:
            print(f'Process with seed {self.seed} starting with epsilon_baseline {epsilon_baseline} out of'
                  f' {self.epsilons_baseline}')
            
            print("creating Baseline Policy")
            self.pi_b_obj = LunarLanderPolicy(self.env, epsilon=epsilon_baseline)
            self.pi_b = self.pi_b_obj.pi
            self.epsilon = epsilon_baseline
            # self.to_append_run_one_iteration = self.to_append_run + [epsilon_baseline,
            #                                                             self._policy_evaluation_exact(self.pi_b)]
            self.to_append_run_one_iteration = self.to_append_run + [epsilon_baseline,
                                                                        0]
            
            for nb_trajectories in self.nb_trajectories_list:
                print(
                    f'Process with seed {self.seed} starting with nb_trajectories {nb_trajectories} out of '
                    f'{self.nb_trajectories_list}')
                # Generate trajectories, both stored as trajectories and (s,a,s',r) transition samples
                print("Generating Trajectories")
                # generate data on the real cartpole environment. Translate this data to the partitioning in generate_batch
                print("The traps are: ", self.traps)
                print("The goals are: ", self.goal)
                data_grid, batch_traj, self.data_cont = self.generate_batch(nb_trajectories, self.env, self.pi_b, self.pi_b_obj)
                self.to_append = self.to_append_run_one_iteration + [nb_trajectories]
                
                # self.write_trajectories_to_txt(self.data_cont, "data_traj.txt")
                
                print("gathered data, transforming it into a dataframe")
                self.data_df = trajToDF(self.data_cont, self.dimensions, 1)
                # self.data_df.to_csv("data_set.csv", index=False)
                print(len(self.data_df.index))
                

                # # ------------------------ MRL --------------------------
                # # print("getting abstraction")
                # # self.discretization_method = 'mrl'
                
                # # # Get discretization
                # # m = MDP_model()
                # # m.fit(
                # #     self.data_df,
                # #     pfeatures=self.dimensions,
                # #     h = -1,
                # #     gamma = 1,
                # #     max_k = 75,
                # #     distance_threshold=None,
                # #     th = 10,
                # #     eta = 25,
                # #     precision_thresh = 1e-14,
                # #     classification = 'DecisionTreeClassifier',
                # #     split_classifier_params = {'random_state':0, 'max_depth':5},
                # #     clustering = 'KMeans',
                # #     n_clusters = 10,
                # #     random_state = 0,
                # #     plot=False,
                # #     verbose=False,
                # #     stochastic=False
                # # )
                # # print("Trained the model!!")
                
                # # # discretize data
                # # self.predictor = predict_cluster(m.df_trained, self.dimensions)
                # # # d_data = self.discretize_data(self.data_cont, self.predictor)
                # # print("start discretization")
                # # tb = time.time()
                # # d_data = self.discretize_data_from_df(self.data_cont, m.df_trained)
                # # self.data = d_data
                # # ta = time.time()
                # # print("time for discretization = ", ta-tb)
                # # # tfile = open('data_df_trained.txt', 'a')
                # # # tfile.write(m.df_trained.to_string())
                # # # tfile.close()
                
                # # # self.write_discrete_data_to_txt(d_data, "data_d.txt")
                # # nb_states = m.df_trained["CLUSTER"].nunique()
                # # self.nb_states = nb_states
                # # print("nb states = ", nb_states)                
                # # # get discrete reward function
                # # self.R_state_state = np.zeros((nb_states, nb_states))
                # # #TODO fix this for this environment, also maybe write some stuff so that this does not have te be done manually
                # # traps = []
                # # goal = []
                # # print(m.R_df)
                # # for state in range(len(self.R_state_state)):
                # #     r = m.R_df[state]
                # #     self.R_state_state[:, state] = r
                # #     if r <= -95:
                # #         traps.append(state)
                # #     if r >= 95:
                # #         goal.append(state)
                # # print("traps = ", traps)
                # # print("goal = ", goal)
                # # self.initial_state = d_data[0][0][1]

                # # # get structure transition function
                # # self.structure = self.get_empty_structure(nb_states)
                # # self.structure = self.add_trans_from_data(self.structure, d_data)
                # # self._count(d_data)
                # # self._build_model()
                # # self.R_s_a = self.compute_r_state_action(self.transition_model, self.R_state_state)
                # # print("found structure")
                
                # # # Calculate Shield                
                # # self.estimator = PACIntervalEstimator(self.structure, 0.1, d_data, self.nb_actions, alpha=5)
                # # self.estimator.calculate_intervals()
                # # self.intervals = self.estimator.get_intervals()                
                
                # # print("Calculating Shield") 
                # # # print(m.R_df) 
                # # self.shielder = ShieldLunarLander(self.structure, traps, goal, self.intervals, self.initial_state)
                # # self.shielder.calculateShield()
                # # # self.shielder.printShield()
                
                # # # # Run the algoirhtm
                # # self.pi_b = LunarLanderPolicy(self.env, epsilon=epsilon_baseline).compute_baseline_size(nb_states, d_data)

                # # print("Running Algorithms")
                # # self._run_algorithms()
                
                # ----------------------------- GreedyCut ----------------------------------
                self.data = self.data_cont
                self.discretization_method = 'GreedyCut'
                bounds = [(-1.0,1.0),(-2.5,2.5),(-10,10),(-10,10),(-7.5,7.5),(-10,10),(0,1),(0,1)]
                self.discretizer = GreedyCut(
                    state_dim = self.dimensions,
                    trajectories=self.data,
                    B=20,
                    bounds=bounds,
                    initial_splits=[4,4,3,3,4,4,2,2],
                    binary_dims=[6,7]
                    # success_reward=100
                    
                )
                self.predictor = self.discretizer.Greedy()
                print(self.predictor)
                d_data = self.discretizer.get_discretized_dataset()
                self.data = d_data
                # with open("trajectories.txt", "w") as f:
                #     for i, traj in enumerate(d_data):
                #         f.write(f"Trajectory {i}:\n")
                #         for transition in traj:
                #             action, state, next_state, reward = transition
                #             f.write(f"{action}, {state}, {next_state}, {reward}\n")
                #         f.write("\n")
                        
                # with open("trajectories_grid.txt", "w") as f:
                #     for i, traj in enumerate(data_grid):
                #         f.write(f"Trajectory {i}:\n")
                #         for transition in traj:
                #             action, state, next_state, reward = transition
                #             f.write(f"{action}, {state}, {next_state}, {reward}\n")
                #         f.write("\n")
                nb_states = self.discretizer.get_num_regions(self.predictor)
                self.nb_states = nb_states
                print("nb_states = ", nb_states)
                # get structure transition function
                print("Build Model")
                # print(m.R_df) 
                goal = []
                traps = []
                for traj in d_data:
                    final_transition = traj[-1]
                    s, a, s_next, r = final_transition
                    if  r==-100 and s_next not in traps:
                        traps.append(s_next)
                    if  r==100 and s_next not in goal:
                        goal.append(s_next)    
                print(traps)
                print(goal)
                
                # R_state_state (as a RewardDict)
                # self.R_state_state = np.zeros((nb_states, nb_states))
                reward_per_next = {}
                for next_state in range(self.nb_states):
                    if next_state in traps:
                        reward_per_next[next_state] = -100
                    elif next_state in goal:
                        reward_per_next[next_state] = 100
                    else:
                        center = self.discretizer.region2centre(next_state)
                        r = self.env.get_reward_from_centre(center)
                        reward_per_next[next_state] = r
            
                
                self.R_state_state = RewardDict(
                    nb_states=self.nb_states,
                    terminal_states=traps,
                    goal_states=goal,
                    reward_per_next=reward_per_next,
                )
                self.structure = self.get_empty_structure(nb_states)
                self.structure = self.add_trans_from_data(self.structure, d_data)
                self._count(d_data)
                self._build_model()           
                self.R_s_a = self.compute_r_state_action(self.transition_model, self.R_state_state)
                self.initial_state = d_data[0][0][1]
                
                # Calculate Shield                
                self.estimator = PACIntervalEstimator(self.structure, 0.1, d_data, self.nb_actions, alpha=5)
                self.estimator.calculate_intervals()
                self.intervals = self.estimator.get_intervals()                
                
                print("Calculating Shield") 
                # 1: Loop through all state-next_state pairs in the abstraction
                # 2
                # option 2: Use the known parameters to set the traps based on the centres. Any state for which the centre is outside the
                self.shielder = ShieldLunarLander(self.structure, traps, goal, self.intervals, self.initial_state)
                self.shielder.calculateShield()
                # self.shielder.printShield()
                
                # # Run the algoirhtm
                self.pi_b = LunarLanderPolicy(self.env, epsilon=epsilon_baseline).compute_baseline_greedy(nb_states, self.discretizer)
                print("Running Algorithms")
                self._run_algorithms()
                # ----------------------------- GRID ---------------------------------
                self.discretization_method = 'grid'
                self.pi_b = LunarLanderPolicy(self.env, epsilon=epsilon_baseline).pi
                self.initial_state_cont, self.initial_state = self.env.get_init_state()
                print("The baseline has length:", len(self.pi_b))
                self.nb_states = self.env.get_nb_states()
                self.data = data_grid
                self.R_state_state = self.R_state_state_grid
                
                print("Estimating Intervals")            
                self._count(self.data)
                self._build_model()
                self.R_s_a = self.compute_r_state_action(self.transition_model, self.R_state_state)
                self.structure = self._tm_to_next_states()
                self.estimator = PACIntervalEstimator(self.structure, 0.1, self.data, self.nb_actions, alpha=5)
                self.estimator.calculate_intervals()
                self.intervals = self.estimator.get_intervals()   
                # self.estimator = imdp_builder(self.data, self.count_state_action_state, self.count_state_action, self.episodic, beta=1e-4, kstep=1)
                # self.intervals = self.estimator.get_intervals()
                
                
                print("Calculating Shield")  
                # self.structure = self.build_transition_matrix()
                self.shielder = ShieldLunarLander(self.structure, self.traps, self.goal, self.intervals, self.initial_state)
                self.shielder.calculateShield()
                # self.shielder.printShield()
                print("Running Algorithms")
                self._run_algorithms()
                # # ----------------------------- SPIBB-DQN ----------------------------------
                # # self.pi_b = LunarLanderPolicy(self.env, epsilon=epsilon_baseline).pi
                # # self.discretization_method = 'SPIBB-DQN'
                # # self.data = self.data_cont
                # # self._run_spibb_dqn('SPIBB-DQN')
                # ----------------------------- CQL-DQN ----------------------------------
                # with open("perf.txt", "a") as f:
                #     f.write(f"---------- NEW ITTERATION with {nb_trajectories} traj ----------\n")
                # self.discretization_method = 'CQL-DQN'
                # self.data = self.data_cont
                # # self._run_cql_dqn()
                # agent = train_cql_dqn(self.env, self.env_name, self.data_cont)
                # print("Evaluating CQL")
                # spibb_perf, discounted_method_perf, succ_rate, failure_rate, avoid_rate = evaluate_policy(self.env, None, 100, 250, self.discretization_method, ai=agent, env_name=self.env_name, generate_gif=self.generate_gif, gif_name="cql_dqn.gif",render_env=self.render_env, gamma=self.gamma)
                # print("perf = ", spibb_perf)
                # print("succ rate = ", succ_rate)
                # print("avoid rate = ", avoid_rate)
                # with open("perf.txt", "a") as f:
                #     f.write("CQL-DQN\n")
                #     f.write(f"perf = {spibb_perf}\n")
                #     f.write(f"succ rate = {succ_rate}\n")
                #     f.write(f"avoid rate = {avoid_rate}\n")
                #     f.write("\n")
                    
                # self.discretization_method = 'test'
                # print("Evaluating CQL-DQN policy on the grid")
                # spibb_perf, discounted_method_perf, succ_rate, failure_rate, avoid_rate = evaluate_policy(self.env, None, 100, 250, self.discretization_method, ai=agent, env_name=self.env_name, generate_gif=self.generate_gif, gif_name="cql_dqn_grid.gif",render_env=self.render_env, gamma=self.gamma)
                # print("perf = ", spibb_perf)
                # print("succ rate = ", succ_rate)
                # print("avoid rate = ", avoid_rate)
                # with open("perf.txt", "a") as f:
                #     f.write("CQL-DQN policy on the grid\n")
                #     f.write(f"perf = {spibb_perf}\n")
                #     f.write(f"succ rate = {succ_rate}\n")
                #     f.write(f"avoid rate = {avoid_rate}\n")
                #     f.write("\n")
                    
                # self.discretization_method = 'heuristic'
                # spibb_perf, discounted_method_perf, succ_rate, failure_rate, avoid_rate = evaluate_policy(self.env, None, 100, 250, self.discretization_method, ai=agent, env_name=self.env_name, generate_gif=self.generate_gif, gif_name="heuristic.gif",render_env=self.render_env, gamma=self.gamma)
                # with open("perf.txt", "a") as f:
                #     f.write("Heuristic\n")
                #     f.write(f"perf = {spibb_perf}\n")
                #     f.write(f"succ rate = {succ_rate}\n")
                #     f.write(f"avoid rate = {avoid_rate}\n")
                #     f.write("\n")
                    
                # self.discretization_method = 'grid'
                # self.pi_b = LunarLanderPolicy(self.env, epsilon=0).pi
                # self.initial_state_cont, self.initial_state = self.env.get_init_state()
                # print("The baseline has length:", len(self.pi_b))
                # self.nb_states = self.env.get_nb_states()
                # self.data = data_grid
                # self.R_state_state = self.R_state_state_grid
                
                # print("run_baseline")
                # self._run_baseline()
                

                
                # print("Calculating Shield")  
                # # self.structure = self.build_transition_matrix()
                # self.shielder = ShieldLunarLander(self.structure, self.traps, self.goal, self.intervals, self.initial_state)
                # self.shielder.calculateShield()
                # # self.shielder.printShield()
                # print("Running Algorithms")
                # print("Evaluating SPIBB-GRID")
                # self._run_algorithms()
                

                
                
                
    def write_discrete_data_to_txt(self, data_disc, filename="discrete_trajectories.txt"):
        """
        Writes discretized trajectories to a text file.

        Format per step:
        State_region: <s>, Action: <a>, Next_state_region: <ns>, Reward: <r>
        """

        with open(filename, "w") as f:
            for traj_idx, trajectory in enumerate(data_disc):
                f.write(f"Trajectory {traj_idx}\n")
                f.write("-" * 50 + "\n")

                for step_idx, transition in enumerate(trajectory):
                    action, state_region, next_state_region, reward = transition

                    line = (
                        f"Step {step_idx} | "
                        f"State_region: {state_region}, "
                        f"Action: {action}, "
                        f"Next_state_region: {next_state_region}, "
                        f"Reward: {reward}\n"
                    )

                    f.write(line)

                f.write("\n")
    def write_trajectories_to_txt(self, trajectories_cont, filename="trajectories.txt"):
        """
        Writes continuous trajectories to a text file.

        Each step will be written as:
        State: <state>, Action: <action>, Next_state: <next_state>, Reward: <reward>, Terminated: <terminated>, Truncated: <truncated>
        """

        with open(filename, "w") as f:
            for traj_idx, trajectory in enumerate(trajectories_cont):
                f.write(f"Trajectory {traj_idx}\n")
                f.write("-" * 60 + "\n")

                for step_idx, step in enumerate(trajectory):
                    state, action, next_state, reward, terminated, truncated = step

                    f.write(
                        f"Step {step_idx} | "
                        f"State: {state}, "
                        f"Action: {action}, "
                        f"Next_state: {next_state}, "
                        f"Reward: {reward}, "
                        f"Terminated: {terminated}, "
                        f"Truncated: {truncated}\n"
                    )

                f.write("\n")     
        
    def generate_batch(self, nb_trajectories, env, pi, policy_object, max_steps=500):
        """
        Generates a data batch for an episodic MDP.
        :param nb_steps: number of steps in the data batch
        :param env: environment to be used to generate the batch on
        :param pi: policy to be used to generate the data as numpy array with shape (nb_states, nb_actions)
        :return: data batch as a list of sublists of the form [state, action, next_state, reward]
        """
        trajectories = []
        trajectories_cont = []
        succ_count = 0

        for traj_count in np.arange(nb_trajectories):
            if traj_count % 100 == 0:
                print("traj count: ", traj_count)
            nb_steps = 0
            trajectorY = []
            trajectorY_cont = []
            env.reset()
            env.set_random_state()
            state, region = env.get_init_state()
            is_done = False
            reached_max_it = False
            while not reached_max_it and not is_done:
                # action_choice = policy_object.heuristic(state)
                if np.random.rand() < self.epsilon:
                    action_choice = np.random.choice(pi.shape[1], p=pi[region])  # random action
                else:
                    action_choice = policy_object.heuristic(state)  # heuristic action
                state, next_state, reward = env.step(action_choice)
                region = env.state2region(state)
                next_region = env.state2region(next_state)
                if reward == -100:
                    next_region = env.get_traps()
                elif reward == 100:
                    next_region  = env.get_goal_state()
                    succ_count+=1
                is_done = env.is_done()                    
                trajectorY.append([action_choice, region, next_region, reward])
                terminated = env.is_terminated()
                truncated = env.is_truncated()
                # if terminated:
                #     print(f"terminated on region = {region}, action = {action_choice}, next region = {next_region}, reward = {reward}")
                nb_steps += 1
                if not nb_steps < max_steps:
                    reached_max_it = True
                    truncated = True
                trajectorY_cont.append([state, action_choice, next_state, reward, terminated, truncated])
                region = next_region
            # print("traj has length", len(trajectorY))
            trajectories_cont.append(trajectorY_cont)
            trajectories.append(trajectorY)
        batch_traj = [val for sublist in trajectories for val in sublist]
        print(f"we had {succ_count}/{nb_trajectories} succesfull episodes")
        return trajectories, batch_traj, trajectories_cont
            
    
    def _count(self, data):
        """
        Counts the state-action pairs and state-action-triplets and stores them.
        """
        if self.episodic:
            batch_trajectory = [val for sublist in data for val in sublist]
        else:
            batch_trajectory = data.copy()
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
            
    def _tm_to_next_states(self):
        structure = np.empty((self.nb_states, self.nb_actions), dtype=object)
        for s in range(self.nb_states):
            for a in range(self.nb_actions):
                structure[s, a] = []

        # Populate from sparse transition model
        for (s, a, s_prime), prob in self.transition_model.items():
            if prob > 0:
                structure[s, a].append(s_prime)
        
        # If a state has no successors in the data, just map to itself and the crash state
        for s in range(self.nb_states):
            has_outgoing_transition = False
            for a in range(self.nb_actions):
                if len(structure[s, a]) == 0:
                    structure[s, a].append(s)
                else:
                    has_outgoing_transition = True
            # if not has_outgoing_transition:
                # for a in range(self.nb_actions):
                    # structure[s, a].append(self.traps[0])
                    
                    
                    
        return structure
    
    def estimate_transitions(self):
        count = 0
        # Prepare the reduced matrix with empty lists
        transition_matrix = np.empty((self.nb_states, self.nb_actions), dtype=object)
        for s in range(self.nb_states):
            for a in range(self.nb_actions):
                if s == self.traps:
                    transition_matrix[s, a] = [self.traps]
                else:
                    print(s, " = ", self.env.get_successor_states(s,a))
                    transition_matrix[s, a] = list(self.env.get_successor_states(s,a))
        self.transition_matrix = transition_matrix

    def discretize_data_from_df(self, data, df):
        """
        Discretizes trajectories using cluster assignments directly from a dataframe.

        Parameters
        ----------
        data : list
            trajectories_cont produced by generate_batch
            [state, action, next_state, reward, terminated, truncated]

        df : pandas.DataFrame
            dataframe containing FEATURE_0..FEATURE_7, CLUSTER, NEXT_CLUSTER

        Returns
        -------
        data_disc : list
            discretized trajectories in the form
            [action, state_region, next_state_region, reward]
        """

        feature_cols = [f"FEATURE_{i}" for i in range(8)]

        # --- build lookup dictionary much faster ---
        features = df[feature_cols].to_numpy()
        clusters = df["CLUSTER"].to_numpy()

        state_to_cluster = {tuple(features[i]): clusters[i] for i in range(len(features))}

        data_disc = []

        for trajectory in data:
            traj = []

            for s, a, ns, r, terminated, truncated in trajectory:
                s_region = state_to_cluster.get(tuple(s))
                ns_region = state_to_cluster.get(tuple(ns))

                if s_region is None or ns_region is None:
                    raise ValueError(
                        f"State {tuple(s)} and next_state {tuple(ns)} not found in lookup."
                    )

                traj.append([a, s_region, ns_region, r])

            data_disc.append(traj)

        return data_disc

    def discretize_data(self, data, predictor):
        all_states = []
        all_next_states = []
        structure = []  # to rebuild trajectories later

        # Step 1: Collect everything
        for trajectory in data:
            traj_info = []
            for transition in trajectory:
                s, a, ns, r, terminated, died = transition
                all_states.append(s)
                all_next_states.append(ns)
                traj_info.append((a, r))
            structure.append(traj_info)

        # Step 2: Predict ALL at once
        all_states = np.asarray(all_states)
        all_next_states = np.asarray(all_next_states)

        s_regions = predictor.predict(all_states)
        ns_regions = predictor.predict(all_next_states)

        # Step 3: Rebuild dataset
        data_disc = []
        idx = 0

        for traj_info in structure:
            traj = []
            for (a, r) in traj_info:
                traj.append([a, s_regions[idx], ns_regions[idx], r])
                idx += 1
            data_disc.append(traj)

        return data_disc
                             
    def get_empty_structure(self, nb_states):
        empty_structure = np.empty((nb_states, self.nb_actions), dtype=object)                      
        
        for state in range(nb_states):
            for action in range(self.nb_actions):
                empty_structure[state, action] = np.array([], dtype=int)
                
        return empty_structure            
    def add_trans_from_data(self,structure, data):
        # num_states = len(structure)
        # num_actions = len(structure[0])
        for trajectory in data:
            for transition in trajectory:
                s=transition[1]
                a=transition[0]
                ns=transition[2]
                poss_next = structure[s,a]
                if not ns in poss_next:
                    poss_next = np.append(poss_next, [int(ns)])
                    structure[s][a] = poss_next
        return structure  
    
    def reduce_transition_matrix(self, transition_matrix):
        """
        Reduces a transition matrix to only include possible end states for each state-action pair.

        Args:
        - transition_matrix (numpy.ndarray): A 3D numpy array of shape (num_states, num_actions, num_states) 
        where each element represents the probability of transitioning from one state to another
        given a certain action.

        Returns:
        - numpy.ndarray: A 3D numpy array of shape (num_states, num_actions, num_possible_transitions) 
        where each element contains the indices of possible end states.
        """
        num_states = len(transition_matrix)
        num_actions = len(transition_matrix[0])
        # Prepare the reduced matrix to hold the indices of possible states
        reduced_matrix = np.empty((num_states, num_actions), dtype=object)
        
        # Loop through each state and action to populate the reduced matrix
        for state in range(num_states):
            for action in range(num_actions):
                # Get indices of nonzero probabilities (possible end states)
                possible_states = np.nonzero(transition_matrix[state, action])[0]
                reduced_matrix[state, action] = np.array(possible_states)
        
        return reduced_matrix
       
    def build_transition_matrix(self):
        """
        Builds a reduced transition matrix that lists possible next states
        for each (state, action) pair, based on the observed trajectories.

        Returns
        -------
        np.ndarray
            A 2D array of shape (num_states, num_actions), where each entry
            is a list of possible next states for that (state, action).
        """
        count = 0
        # Prepare the reduced matrix with empty lists
        transition_matrix = self.transition_matrix.copy()


        # Fill matrix with next states from counts
        for (state, action, next_state) in self.count_state_action_state.keys():
            if next_state not in transition_matrix[state, action]:
                transition_matrix[state, action].append(next_state)

        # for s in range(self.nb_states):
        #     for a in range(self.nb_actions):
        #         if len(transition_matrix[s, a]) == 0:
        #             transition_matrix[s, a] = [self.traps]

        # for i in range(len(transition_matrix)):
        #     for j in range(len(transition_matrix[i])):
        #         if len(transition_matrix[i][j]) > 1:
        #             count+=1
        # print(transition_matrix)
        # print(count)
        return transition_matrix
    
class GymFrozenLakeExperiment(Experiment):
    # Inherits from the base class Experiment to implement the Wet Chicken experiment specifically.
    fixed_params_exp_columns = ['seed', 'gamma']
    
    def _set_env_params(self):
        """
        Reads in all parameters necessary from self.experiment_config to set up the Wet Chicken experiment.
        """
        self.generate_gif = True
        self.episodic = True
        self.env_name = self.experiment_config['META']['env_name']
        self.gamma = float(self.experiment_config['ENV_PARAMETERS']['GAMMA'])
        self.fixed_params_exp_list = [self.seed, self.gamma]
        self.dimensions = 2
        
        print("start env")
        self.env = gymIce()
        self.render_env = gymIce(True)
        print("end env")
        
        self.nb_states = self.env.get_nb_states()
        self.nb_actions = self.env.get_nb_actions()
        self.traps = self.env.get_traps()
        self.goal = self.env.get_goal_state()
        self.initial_state_cont, self.initial_state = self.env.get_init_state()
        
        
        self.R_state_state = self.env.get_reward_function()

        self.baseline_method = self.experiment_config['BASELINE']['method']
        self.fixed_params_exp_list = [self.seed, self.gamma]

        self.epsilons_baseline = ast.literal_eval(self.experiment_config['BASELINE']['epsilons_baseline'])
        

        self.pi_b = self.env.get_baseline_policy(epsilon=self.epsilons_baseline[0])
        self.nb_trajectories_list = ast.literal_eval(self.experiment_config['BASELINE']['nb_trajectories_list'])
        self.variable_params_exp_columns = ['i', 'epsilon_baseline', 'pi_b_perf', 'length_trajectory']

        self.estimate_baseline=bool((util.strtobool(self.experiment_config['META']['estimate_baseline'])))
        print("estimating transitions")

        
    def _run_one_iteration(self):
        for epsilon_baseline in self.epsilons_baseline:
            print(f'Process with seed {self.seed} starting with epsilon_baseline {epsilon_baseline} out of'
                  f' {self.epsilons_baseline}')
            
            self.pi_b = self.env.get_baseline_policy(epsilon=epsilon_baseline)
            self.to_append_run_one_iteration = self.to_append_run + [epsilon_baseline,
                                                                        0]

            for nb_trajectories in self.nb_trajectories_list:
                print(
                    f'Process with seed {self.seed} starting with nb_trajectories {nb_trajectories} out of '
                    f'{self.nb_trajectories_list}')
                # Generate trajectories, both stored as trajectories and (s,a,s',r) transition samples
                print("Generating Trajectories")
                # generate data on the real cartpole environment. Translate this data to the partitioning in generate_batch
                self.pi_b = self.env.get_baseline_policy(epsilon=epsilon_baseline)
                data_grid, batch_traj, self.data_cont = self.generate_batch(nb_trajectories, self.env, self.pi_b)
                self.to_append = self.to_append_run_one_iteration + [nb_trajectories]
                # with open("data_grid.txt", "w") as f:
                #     for traj_idx, trajectory in enumerate(data_grid, start=1):
                #         f.write(f"----trajectory {traj_idx}----\n")
                #         for step_idx, transition in enumerate(trajectory, start=1):
                #             action, region, next_region, reward = transition
                #             f.write(
                #                 f"step {step_idx}: "
                #                 f"action={action}, region={region}, "
                #                 f"next_region={next_region}, reward={reward}\n"
                #             )
                #         f.write("\n")
                self.data_df = trajToDF(self.data_cont, self.dimensions, self.data_cont[0][0][3])
                
                # ------------------------ MRL --------------------------
                self.discretization_method = 'mrl'
                # Get discretization
                m = MRL_model()
                m.fit(
                    self.data_df,
                    pfeatures=self.dimensions,
                    h = -1,
                    gamma = 0.95,
                    max_k = 75,
                    distance_threshold=None,
                    th = 5,
                    eta = 25,
                    precision_thresh = -0.5,
                    classification = 'DecisionTreeClassifier',
                    split_classifier_params = {'random_state':0, 'max_depth':4},
                    clustering = 'Agglomerative',
                    n_clusters = 2,
                    random_state = 0,
                    plot=True,
                    verbose=False,
                    stochastic=True
                )
                print("Trained the model!!")
                
                # discretize data
                self.predictor = predict_cluster(m.df_trained, self.dimensions)
                d_data = self.discretize_data_from_df(self.data_cont, m.df_trained)
                nb_states = m.df_trained["CLUSTER"].max()+1 
                self.nb_states = nb_states
                self.data = d_data
                print("nb states = ", nb_states)  
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    print(m.R_df)    
                      
                # get discrete reward function
                self.R_state_state = np.zeros((nb_states, nb_states))
                traps = []
                goal = []
                for state in range(len(self.R_state_state)):
                    r = m.R_df.get(state, 0.0)
                    self.R_state_state[:, state] = r
                    if r < -1.0:
                        traps.append(state)
                    if r > 1.0:
                        goal.append(state)

                # get structure transition function
                self.structure = self.get_empty_structure(nb_states)
                self.structure = self.add_trans_from_data(self.structure, d_data)
                self._count(d_data)
                self._build_model()           
                self.R_s_a = self.compute_r_state_action(self.transition_model, self.R_state_state)
                self.initial_state = d_data[0][0][1]
                # # with open("data_d.txt", "w") as f:
                # #     for item in d_data:
                # #         f.write(item)
                # #         f.write("\n")
                # # print(d_data)
                
                # Calculate Shield                
                self.estimator = PACIntervalEstimator(self.structure, 0.1, d_data, self.nb_actions, alpha=5)
                self.estimator.calculate_intervals()
                self.intervals = self.estimator.get_intervals()   
                             
                
                print("Calculating Shield") 
                # print(m.R_df) 
                self.shielder = ShieldFrozenLake(self.structure, traps, goal, self.intervals)
                self.shielder.calculateShield()
                self.shielder.printShield()
                
                # # Run the algoirhtm
                self.pi_b = self.env.compute_baseline_policy_size(size=nb_states, epsilon=epsilon_baseline)

                print("Running Algorithms")
                self._run_algorithms()
                
                # # ----------------------------- GRID ---------------------------------
                # self.discretization_method = 'grid'
                # self.pi_b = self.env.get_baseline_policy(epsilon=epsilon_baseline)
                # self.nb_states = self.env.get_nb_states()
                # self.data = data_grid
                # self.R_state_state = self.env.get_reward_function()
                # self.initial_state_cont, self.initial_state = self.env.get_init_state()
                
                # print("Estimating Intervals")            
                # self._count(self.data)
                # self._build_model()
                # self.R_s_a = self.compute_r_state_action(self.transition_model, self.R_state_state)
                # self.structure = self._tm_to_next_states()
                # self.estimator = PACIntervalEstimator(self.structure, 0.1, self.data, self.nb_actions, alpha=5)
                # self.estimator.calculate_intervals()
                # self.intervals = self.estimator.get_intervals()   
                
                
                # print("Calculating Shield")  
                # # self.structure = self.build_transition_matrix()
                # self.shielder = ShieldFrozenLake(self.structure, self.traps, self.goal, self.intervals)
                # self.shielder.calculateShield()
                # # self.shielder.printShield()
                # print("Running Algorithms")
                # self._run_algorithms()
                # # ----------------------------- SPIBB-DQN ----------------------------------
                # self.discretization_method = 'SPIBB-DQN'
                # self.data = self.data_cont
                # self._run_spibb_dqn('SPIBB-DQN')
                # # ----------------------------- CQL-DQN ----------------------------------                
                # self.discretization_method = 'CQL-DQN'
                # self.data = self.data_cont
                # self._run_cql_dqn()
                
    def plot_trajectories_by_region(self, trajectories, predictor, grid_size=(8, 8)):
        """
        trajectories: list of trajectories
            each trajectory = list of [state, action, next_state, reward, terminated]
        state2region: function mapping continuous state -> discrete region
        grid_size: (width, height) of frozen lake
        """

        # 1. Extract all states
        states = []

        for traj in trajectories:
            for i, transition in enumerate(traj):
                state, action, next_state, reward, terminated, truncated = transition
                states.append(state)

                # Include final next_state
                if terminated or truncated:
                    states.append(next_state)

        states = np.array(states)

        # 2. Map states to regions
        region_to_states = defaultdict(list)

        for s in states:
            region = state2region(predictor, s, self.dimensions)
            region_to_states[region].append(s)

        # 3. Assign colors to regions
        unique_regions = list(region_to_states.keys())
        cmap = plt.cm.get_cmap('tab20', len(unique_regions))

        region_colors = {
            region: cmap(i) for i, region in enumerate(unique_regions)
        }

        # 4. Plot
        plt.figure(figsize=(6, 6))

        # Draw Frozen Lake grid background
        for x in range(grid_size[0] + 1):
            plt.axvline(x, color='lightgray', linewidth=1)
        for y in range(grid_size[1] + 1):
            plt.axhline(y, color='lightgray', linewidth=1)

        # Plot states
        for region, region_states in region_to_states.items():
            region_states = np.array(region_states)
            plt.scatter(
                region_states[:, 0],
                region_states[:, 1],
                s=10,
                color=region_colors[region],
                label=str(region),
                alpha=0.7
            )

        plt.xlim(0, grid_size[0])
        plt.ylim(0, grid_size[1])

        plt.gca().set_aspect('equal')
        plt.gca().invert_yaxis()  # match grid-style orientation (optional)

        plt.title("Continuous States Colored by Discrete Region")
        plt.xlabel("X")
        plt.ylabel("Y")

        # Optional legend (can be large if many regions)
        if len(unique_regions) <= 20:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig("grid_mrl.png")
        plt.show()
    
    def generate_batch(self, nb_trajectories, env, pi, max_steps=1000):
        """
        Generates a data batch for an episodic MDP.
        :param nb_steps: number of steps in the data batch
        :param env: environment to be used to generate the batch on
        :param pi: policy to be used to generate the data as numpy array with shape (nb_states, nb_actions)
        :return: data batch as a list of sublists of the form [state, action, next_state, reward]
        """
        trajectories = []
        trajectories_cont = []
        env.reset()
        for _ in np.arange(nb_trajectories):
            nb_steps = 0
            trajectorY = []
            trajectorY_cont = []            
            state = self.env.get_state()
            region = self.env.state2region(state)
            # print(f"init state = {state}, region = {region}")
            is_done = False
            while nb_steps < max_steps and not is_done:
                # step through the environment
                action_choice = np.random.choice(pi.shape[1], p=pi[region])
                state, next_state, reward = env.step(action_choice)

                # add discrete transition
                region = env.state2region(state)
                next_region = env.state2region(next_state)
                trajectorY.append([action_choice, region, next_region, reward])
                
                # add continuous transition
                terminated = env.is_terminated()
                truncated = env.is_truncated()
                trajectorY_cont.append([state, action_choice, next_state, reward, terminated, truncated])
                
                # update values
                is_done = env.is_done()     
                region = next_region
                state = next_state
                nb_steps += 1
            trajectories_cont.append(trajectorY_cont)
            trajectories.append(trajectorY)
            env.reset()
            env.set_random_state()
        batch_traj = [val for sublist in trajectories for val in sublist]
        return trajectories, batch_traj, trajectories_cont
            
    def discretize_data_from_df(self, data, df):
        """
        Discretizes trajectories using cluster assignments directly from a dataframe.

        Parameters
        ----------
        data : list
            trajectories_cont produced by generate_batch
            [state, action, next_state, reward, terminated, truncated]

        df : pandas.DataFrame
            dataframe containing FEATURE_0..FEATURE_1, CLUSTER, NEXT_CLUSTER

        Returns
        -------
        data_disc : list
            discretized trajectories in the form
            [action, state_region, next_state_region, reward]
        """

        feature_cols = [f"FEATURE_{i}" for i in range(2)]

        # build lookup dictionary much faster
        features = df[feature_cols].to_numpy()
        clusters = df["CLUSTER"].to_numpy()

        state_to_cluster = {tuple(features[i]): clusters[i] for i in range(len(features))}

        data_disc = []

        for trajectory in data:
            traj = []

            for s, a, ns, r, terminated, truncated in trajectory:
                s_region = state_to_cluster.get(tuple(s))
                ns_region = state_to_cluster.get(tuple(ns))

                if s_region is None or ns_region is None:
                    raise ValueError(
                        f"State {tuple(s)} and next_state {tuple(ns)} not found in lookup."
                    )

                traj.append([a, s_region, ns_region, r])

            data_disc.append(traj)

        return data_disc
    
    def discretize_data(self, data, predictor):
        data_disc = []
        for trajectory in data:
            traj = []
            for transition in trajectory:
                s = transition[0]
                a = transition[1]
                ns = transition[2]
                r = transition[3]
                terminated = transition[4]
                died = transition[5]
                s_d = state2region(predictor, s, self.dimensions)
                ns_d = state2region(predictor, ns, self.dimensions)
                traj.append([a, s_d, ns_d, r])
            data_disc.append(traj)
        return data_disc
    
    def get_empty_structure(self, nb_states):
        empty_structure = np.empty((nb_states, self.nb_actions), dtype=object)                      
        
        for state in range(nb_states):
            for action in range(self.nb_actions):
                empty_structure[state, action] = np.array([], dtype=int)
                
        return empty_structure            
    def add_trans_from_data(self,structure, data):
        for trajectory in data:
            for transition in trajectory:
                s=transition[1]
                a=transition[0]
                ns=transition[2]
                poss_next = structure[s,a]
                if not ns in poss_next:
                    poss_next = np.append(poss_next, [ns])
                    structure[s][a] = poss_next
        print("first element of structure = ", structure[0,0])
        return structure  
    
    def _count(self, data):
        """
        Counts the state-action pairs and state-action-triplets and stores them.
        """
        if self.episodic:
            batch_trajectory = [val for sublist in data for val in sublist]
        else:
            batch_trajectory = data.copy()
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
            
    def _tm_to_next_states(self):
        structure = np.empty((self.nb_states, self.nb_actions), dtype=object)
        for s in range(self.nb_states):
            for a in range(self.nb_actions):
                structure[s, a] = []

        # Populate from sparse transition model
        for (s, a, s_prime), prob in self.transition_model.items():
            if prob > 0:
                structure[s, a].append(s_prime)
        
        # If a state has no successors in the data, just map to itself and the crash state
        for s in range(self.nb_states):
            has_outgoing_transition = False
            for a in range(self.nb_actions):
                if len(structure[s, a]) == 0:
                    structure[s, a].append(s)
                else:
                    has_outgoing_transition = True
            # if not has_outgoing_transition:
                # for a in range(self.nb_actions):
                    # structure[s, a].append(self.traps[0])
                    
                    
                    
        return structure

class GymMazeExperiment(Experiment):
    fixed_params_exp_columns = ['seed', 'gamma']
    def _set_env_params(self):
        """
        Reads in all parameters necessary from self.experiment_config to set up the Wet Chicken experiment.
        """
        self.episodic = True
        self.generate_gif = True

        self.gamma = float(self.experiment_config['ENV_PARAMETERS']['GAMMA'])
        self.env_name = str(self.experiment_config['META']['env_name'])
        print("start env")
        self.env = maze()
        self.render_env = maze(True)
        print("get values")
        self.nb_states = self.env.get_nb_states()
        self.nb_actions = self.env.get_nb_actions()
        

        self.traps = self.env.get_traps()
        self.goal = self.env.get_goal_state()

        self.initial_state = self.env.get_init_state()[1]
        
        # self.P = self.env.get_transition_function()
        self.R_state_state = self.env.get_reward_function()

        # print("calcing r_sa")
        # self.R_state_action = self.compute_r_state_action(self.P, self.R_state_state)
        
        self.fixed_params_exp_list = [self.seed, self.gamma]

        # pi_rand = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        # print("calcing rand perf")
        # pi_rand_perf = self._policy_evaluation_exact(pi_rand)
        # print(f"pi_rand_perf = {pi_rand_perf}")

        # self.fixed_params_exp_list.append(pi_rand_perf)

        # pi_star = PiStar(pi_b=None, gamma=self.gamma, nb_states=self.nb_states, nb_actions=self.nb_actions,
        #                  data=[[]], R=self.R_state_state, episodic=self.episodic, P=self.P)
        # pi_star.fit()
        # pi_star_perf = self._policy_evaluation_exact(pi_star.pi)
        # print(f"pi_star_perf = {pi_star_perf}")
        # self.fixed_params_exp_list.append(pi_star_perf)


        self.epsilons_baseline = ast.literal_eval(self.experiment_config['BASELINE']['epsilons_baseline'])
        
        # pi_base_perf = self._policy_evaluation_exact(self.env.get_baseline_policy(self.epsilons_baseline[0]))
        # print(self.env.get_baseline_policy(self.epsilons_baseline[0]))
        # print(f"pi_baseline_perf = {pi_base_perf}")
        self.pi_b = mazePolicy(self.env, epsilon=self.epsilons_baseline[0]).pi
        # pi_base_perf = evaluate_policy(self.env, self.pi_b, 1, 100)
        # print(pi_base_perf)
        self.nb_trajectories_list = ast.literal_eval(self.experiment_config['BASELINE']['nb_trajectories_list'])
        self.variable_params_exp_columns = ['i', 'epsilon_baseline', 'pi_b_perf', 'length_trajectory']

        self.estimate_baseline=bool((util.strtobool(self.experiment_config['ENV_PARAMETERS']['estimate_baseline'])))
        print("estimating transitions")
        # self.estimate_transitions()
        self.dimensions = 2
        
    def _run_one_iteration(self):
        for epsilon_baseline in self.epsilons_baseline:
            print(f'Process with seed {self.seed} starting with epsilon_baseline {epsilon_baseline} out of'
                  f' {self.epsilons_baseline}')
            
            print("creating Baseline Policy")
            self.pi_b = mazePolicy(self.env, epsilon=epsilon_baseline).pi
            # self.to_append_run_one_iteration = self.to_append_run + [epsilon_baseline,
            #                                                             self._policy_evaluation_exact(self.pi_b)]
            self.to_append_run_one_iteration = self.to_append_run + [epsilon_baseline,
                                                                        0]
            for nb_trajectories in self.nb_trajectories_list:
                print(
                    f'Process with seed {self.seed} starting with nb_trajectories {nb_trajectories} out of '
                    f'{self.nb_trajectories_list}')
                # Generate trajectories, both stored as trajectories and (s,a,s',r) transition samples
                print("Generating Trajectories")
                # generate data on the real cartpole environment. Translate this data to the partitioning in generate_batch
                self.pi_b = mazePolicy(self.env, epsilon=epsilon_baseline).pi
                data_grid, batch_traj, self.data_cont = self.generate_batch(nb_trajectories, self.env, self.pi_b)
                self.to_append = self.to_append_run_one_iteration + [nb_trajectories]
                
                self.data_df = trajToDF(self.data_cont, self.dimensions, self.data_cont[0][0][3])

                # print(self.data_cont)
                # with pd.option_context('display.max_rows', None,
                #        'display.max_columns', None):
                #     print(self.data_df)
                # print("getting abstraction")
                # Get the intervals from the abstraction using the method from badings
                # self._count()
                # # print("done counts")
                # self.estimator = imdp_builder(self.data, self.count_state_action_state, self.count_state_action, self.episodic, beta=1e-4, kstep=1)
                # self.intervals = self.estimator.get_intervals()
                
                # ------------------------ MRL --------------------------
                self.discretization_method = 'mrl'
                # Get discretization
                m = MRL_model()
                m.fit(
                    self.data_df,
                    pfeatures=self.dimensions,
                    h = -1,
                    gamma = 1,
                    max_k = 50,
                    distance_threshold=0.5,
                    th = 1,
                    eta = 25,
                    precision_thresh = 1e-14, #1e-14
                    classification = 'DecisionTreeClassifier',
                    split_classifier_params = {'random_state':0, 'max_depth':2},
                    clustering = 'Agglomerative',
                    n_clusters = None,
                    random_state = 0,
                    plot=True,
                    verbose=False,
                    stochastic=False
                )
                print("Trained the model!!")
                
                # discretize data
                self.predictor = predict_cluster(m.df_trained, self.dimensions)
                d_data = self.discretize_data(self.data_cont, self.predictor)
                
                nb_states = m.df_trained["CLUSTER"].nunique()
                self.nb_states = nb_states
                self.data = d_data
                print("nb states = ", nb_states)                
                # get discrete reward function
                self.R_state_state = np.zeros((nb_states, nb_states))
                traps = []
                goal = []
                for state in range(len(self.R_state_state)):
                    r = m.R_df[state]
                    self.R_state_state[:, state] = r
                    if r <= -1.0:
                        traps.append(state)
                    if r >= 0.0:
                        goal.append(state)

                # get structure transition function
                self.structure = self.get_empty_structure(nb_states)
                self.structure = self.add_trans_from_data(self.structure, d_data)
                self._count(d_data)
                self._build_model()           
                self.R_s_a = self.compute_r_state_action(self.transition_model, self.R_state_state)
                self.initial_state = d_data[0][0][1]
                # # with open("data_d.txt", "w") as f:
                # #     for item in d_data:
                # #         f.write(item)
                # #         f.write("\n")
                # # print(d_data)
                
                # Calculate Shield                
                self.estimator = PACIntervalEstimator(self.structure, 0.1, d_data, self.nb_actions, alpha=5)
                self.estimator.calculate_intervals()
                self.intervals = self.estimator.get_intervals()   
                             
                
                print("Calculating Shield") 
                # print(m.R_df) 
                self.shielder = ShieldMaze(self.structure, traps, goal, self.intervals, self.initial_state)
                self.shielder.calculateShield()
                # self.shielder.printShield()
                
                # # Run the algoirhtm
                self.pi_b = mazePolicy(self.env, epsilon=epsilon_baseline).compute_baseline_size(nb_states)

                print("Running Algorithms")
                self._run_algorithms()
                
                # ----------------------------- GRID ---------------------------------
                self.discretization_method = 'grid'
                self.pi_b = mazePolicy(self.env, epsilon=epsilon_baseline).pi
                self.nb_states = self.env.get_nb_states()
                self.data = data_grid
                self.R_state_state = self.env.get_reward_function()
                self.initial_state = self.env.get_init_state()[1]
                
                print("Estimating Intervals")            
                self._count(self.data)
                self._build_model()
                self.R_s_a = self.compute_r_state_action(self.transition_model, self.R_state_state)
                self.structure = self._tm_to_next_states()
                self.estimator = PACIntervalEstimator(self.structure, 0.1, self.data, self.nb_actions, alpha=5)
                self.estimator.calculate_intervals()
                self.intervals = self.estimator.get_intervals()   
                
                
                print("Calculating Shield")  
                # self.structure = self.build_transition_matrix()
                self.shielder = ShieldMaze(self.structure, self.traps, self.goal, self.intervals, self.initial_state)
                self.shielder.calculateShield()
                self.shielder.printShield()
                print("Running Algorithms")
                self._run_algorithms()
                # ----------------------------- SPIBB-DQN ----------------------------------
                self.discretization_method = 'SPIBB-DQN'
                self.data = self.data_cont
                self._run_spibb_dqn('SPIBB-DQN')
                
    def generate_batch(self, nb_trajectories, env, pi, max_steps=100):
        """
        Generates a data batch for an episodic MDP.
        :param nb_steps: number of steps in the data batch
        :param env: environment to be used to generate the batch on
        :param pi: policy to be used to generate the data as numpy array with shape (nb_states, nb_actions)
        :return: data batch as a list of sublists of the form [state, action, next_state, reward]
        """
        succ_count = 0
        trajectories = []
        trajectories_cont = []
        env.reset()
        for _ in np.arange(nb_trajectories):
            nb_steps = 0
            trajectorY = []
            trajectorY_cont = []
            state = env.get_state()
            region = env.state2region(state)
            reached_max_it = False
            is_done = False
            # print(state)
            while not reached_max_it and not is_done:
                # print("AAAAA")
                action_choice = np.random.choice(pi.shape[1], p=pi[region])
                state, next_state, reward = env.step(action_choice)
                if reward == 1:
                    succ_count+=1
                # print(state)
                # print(type(state))
                region = env.state2region(state)
                next_region = env.state2region(next_state)
                is_done = env.is_done()                    
                trajectorY.append([action_choice, region, next_region, reward])
                terminated = env.is_terminated()
                truncated = env.is_truncated()
                nb_steps += 1
                if nb_steps >= max_steps:
                    reached_max_it = True
                    truncated = True
                trajectorY_cont.append([state, action_choice, next_state, reward, terminated, truncated])
                region = next_region
                

            trajectories_cont.append(trajectorY_cont)
            trajectories.append(trajectorY)
            env.reset()
            env.set_random_state()
        batch_traj = [val for sublist in trajectories for val in sublist]
        print("number of successes = ", succ_count)
        return trajectories, batch_traj, trajectories_cont
            
    
    def _count(self, data):
        """
        Counts the state-action pairs and state-action-triplets and stores them.
        """
        if self.episodic:
            batch_trajectory = [val for sublist in data for val in sublist]
        else:
            batch_trajectory = data.copy()
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
            
    def _tm_to_next_states(self):
        structure = np.empty((self.nb_states, self.nb_actions), dtype=object)
        for s in range(self.nb_states):
            for a in range(self.nb_actions):
                structure[s, a] = []

        # Populate from sparse transition model
        for (s, a, s_prime), prob in self.transition_model.items():
            if prob > 0:
                structure[s, a].append(s_prime)
        
        # If a state has no successors in the data, just map to itself
        for s in range(self.nb_states):
            for a in range(self.nb_actions):
                if len(structure[s, a]) == 0:
                    structure[s, a].append(s)
                    
        return structure
    

    def estimate_transitions(self):
        count = 0
        # Prepare the reduced matrix with empty lists
        transition_matrix = np.empty((self.nb_states, self.nb_actions), dtype=object)
        for s in range(self.nb_states):
            for a in range(self.nb_actions):
                if s == self.traps:
                    transition_matrix[s, a] = [self.traps]
                else:
                    print(s, " = ", self.env.get_successor_states(s,a))
                    transition_matrix[s, a] = list(self.env.get_successor_states(s,a))
        self.transition_matrix = transition_matrix
        
    def discretize_data(self, data, predictor):
        data_disc = []
        for trajectory in data:
            traj = []
            for transition in trajectory:
                s = transition[0]
                a = transition[1]
                ns = transition[2]
                r = transition[3]
                terminated = transition[4]
                died = transition[5]
                s_d = state2region(predictor, s, self.dimensions)
                ns_d = state2region(predictor, ns, self.dimensions)
                traj.append([a, s_d, ns_d, r])
            data_disc.append(traj)
        return data_disc
      
    def get_empty_structure(self, nb_states):
        empty_structure = np.empty((nb_states, self.nb_actions), dtype=object)                      
        
        for state in range(nb_states):
            for action in range(self.nb_actions):
                empty_structure[state, action] = np.array([], dtype=int)
                
        return empty_structure
                
    def add_trans_from_data(self,structure, data):
        # num_states = len(structure)
        # num_actions = len(structure[0])
        for trajectory in data:
            for transition in trajectory:
                s=transition[1]
                a=transition[0]
                ns=transition[2]
                poss_next = structure[s,a]
                if not ns in poss_next:
                    poss_next = np.append(poss_next, [int(ns)])
                    structure[s][a] = poss_next
        return structure  
    
    def reduce_transition_matrix(self, transition_matrix):
        """
        Reduces a transition matrix to only include possible end states for each state-action pair.

        Args:
        - transition_matrix (numpy.ndarray): A 3D numpy array of shape (num_states, num_actions, num_states) 
        where each element represents the probability of transitioning from one state to another
        given a certain action.

        Returns:
        - numpy.ndarray: A 3D numpy array of shape (num_states, num_actions, num_possible_transitions) 
        where each element contains the indices of possible end states.
        """
        num_states = len(transition_matrix)
        num_actions = len(transition_matrix[0])
        # Prepare the reduced matrix to hold the indices of possible states
        reduced_matrix = np.empty((num_states, num_actions), dtype=object)
        
        # Loop through each state and action to populate the reduced matrix
        for state in range(num_states):
            for action in range(num_actions):
                # Get indices of nonzero probabilities (possible end states)
                possible_states = np.nonzero(transition_matrix[state, action])[0]
                reduced_matrix[state, action] = np.array(possible_states)
        
        return reduced_matrix
       
    def build_transition_matrix(self):
        """
        Builds a reduced transition matrix that lists possible next states
        for each (state, action) pair, based on the observed trajectories.

        Returns
        -------
        np.ndarray
            A 2D array of shape (num_states, num_actions), where each entry
            is a list of possible next states for that (state, action).
        """
        count = 0
        # Prepare the reduced matrix with empty lists
        transition_matrix = self.transition_matrix.copy()


        # Fill matrix with next states from counts
        for (state, action, next_state) in self.count_state_action_state.keys():
            if next_state not in transition_matrix[state, action]:
                transition_matrix[state, action].append(next_state)

        # for s in range(self.nb_states):
        #     for a in range(self.nb_actions):
        #         if len(transition_matrix[s, a]) == 0:
        #             transition_matrix[s, a] = [self.traps]

        # for i in range(len(transition_matrix)):
        #     for j in range(len(transition_matrix[i])):
        #         if len(transition_matrix[i][j]) > 1:
        #             count+=1
        # print(transition_matrix)
        # print(count)
        return transition_matrix

    
class GymCrashingMountainCar(Experiment):
    fixed_params_exp_columns = ['seed', 'gamma']
    def _set_env_params(self):
        """
        Reads in all parameters necessary from self.experiment_config to set up the Wet Chicken experiment.
        """
        self.episodic = True
        self.generate_gif = True

        self.gamma = float(self.experiment_config['ENV_PARAMETERS']['GAMMA'])
        
        print("start env")
        self.env = crashingMountainCar()
        print("get values")
        self.nb_states = self.env.get_nb_states()
        self.state_shape = [2]
        self.nb_actions = self.env.get_nb_actions()

        self.traps = self.env.get_traps()
        self.goal = self.env.get_goal_state()
        print("goal: ", self.goal)
        print("trap: ", self.traps)
        self.initial_state = self.env.get_init_state()
        print("close_to_the_edge", self.env.state2region([0.4, 0.2]))
        # self.P = self.env.get_transition_function()
        self.R_state_state = self.env.get_reward_function()

        # print("calcing r_sa")
        # self.R_state_action = self.compute_r_state_action(self.P, self.R_state_state)
        
        self.fixed_params_exp_list = [self.seed, self.gamma]

        # pi_rand = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        # print("calcing rand perf")
        # pi_rand_perf = self._policy_evaluation_exact(pi_rand)
        # print(f"pi_rand_perf = {pi_rand_perf}")

        # self.fixed_params_exp_list.append(pi_rand_perf)

        # pi_star = PiStar(pi_b=None, gamma=self.gamma, nb_states=self.nb_states, nb_actions=self.nb_actions,
        #                  data=[[]], R=self.R_state_state, episodic=self.episodic, P=self.P)
        # pi_star.fit()
        # pi_star_perf = self._policy_evaluation_exact(pi_star.pi)
        # print(f"pi_star_perf = {pi_star_perf}")
        # self.fixed_params_exp_list.append(pi_star_perf)


        self.epsilons_baseline = ast.literal_eval(self.experiment_config['BASELINE']['epsilons_baseline'])
        
        # pi_base_perf = self._policy_evaluation_exact(self.env.get_baseline_policy(self.epsilons_baseline[0]))
        # print(self.env.get_baseline_policy(self.epsilons_baseline[0]))
        # print(f"pi_baseline_perf = {pi_base_perf}")
        self.pi_b = crashingMountainCarPolicy(self.env, epsilon=self.epsilons_baseline[0]).pi
        # pi_base_perf = evaluate_policy(self.env, self.pi_b, 1, 100)
        # print(pi_base_perf)
        self.nb_trajectories_list = ast.literal_eval(self.experiment_config['BASELINE']['nb_trajectories_list'])
        self.variable_params_exp_columns = ['i', 'epsilon_baseline', 'pi_b_perf', 'length_trajectory']

        self.estimate_baseline=bool((util.strtobool(self.experiment_config['ENV_PARAMETERS']['estimate_baseline'])))
        print("estimating transitions")
        # self.estimate_transitions()
        
    def _run_one_iteration(self):
        for epsilon_baseline in self.epsilons_baseline:
            print(f'Process with seed {self.seed} starting with epsilon_baseline {epsilon_baseline} out of'
                  f' {self.epsilons_baseline}')
            
            print("creating Baseline Policy")
            self.pi_b = crashingMountainCarPolicy(self.env, epsilon=epsilon_baseline).pi
            # self.to_append_run_one_iteration = self.to_append_run + [epsilon_baseline,
            #                                                             self._policy_evaluation_exact(self.pi_b)]
            self.to_append_run_one_iteration = self.to_append_run + [epsilon_baseline,
                                                                        0]
            for nb_trajectories in self.nb_trajectories_list:
                print(
                    f'Process with seed {self.seed} starting with nb_trajectories {nb_trajectories} out of '
                    f'{self.nb_trajectories_list}')
                # Generate trajectories, both stored as trajectories and (s,a,s',r) transition samples
                print("Generating Trajectories")
                # generate data on the real mountain car environment. Translate this data to the partitioning in generate_batch
                self.data, batch_traj, self.data_cont = self.generate_batch(nb_trajectories, self.env, self.pi_b)
                self.to_append = self.to_append_run_one_iteration + [nb_trajectories]
                # print(self.data_cont)
                
                # ------------------------ MRL --------------------------
                self.discretization_method = 'mrl'
                self.dimensions = 2
                self.data_df = trajToDF(self.data_cont, self.dimensions)
                # print("data ", self.data_df)
                # Get discretization
                m = MDP_model()
                m.fit(
                    self.data_df,
                    pfeatures=self.dimensions,
                    h = -1,
                    gamma = self.gamma,
                    max_k = 100,
                    distance_threshold=0.2,
                    th = 0,
                    eta = 50,
                    precision_thresh = 1e-14,
                    classification = 'DecisionTreeClassifier',
                    split_classifier_params = {'random_state':0, 'max_depth':3},
                    clustering = 'Agglomerative',
                    n_clusters = None,
                    random_state = 0,
                    plot=False,
                    verbose=False
                )
                print("Trained the model!!")
                self.predictor = predict_cluster(m.df_trained, self.dimensions)

                # Get transition function:
                m.create_PR(0.2, 0.8, -1, 0.2, "max")
                P_temp = m.P.copy()
                P_temp = P_temp.transpose(1, 0, 2)
                self.structure = self.reduce_transition_matrix(P_temp)
                d_data = self.discretize_data(self.data_cont, self.predictor)
                # print(d_data)
                self.structure = self.add_trans_from_data(self.structure, d_data)
                # print(self.structure)
                print("NB_states = ", len(self.structure))
                # Calculate Shield                
                self.estimator = PACIntervalEstimator(self.structure, 0.1, d_data, self.nb_actions, alpha=5)
                self.estimator.calculate_intervals()
                self.intervals = self.estimator.get_intervals()                
                
                print("Calculating Shield") 
                print(m.R_df) 
                traps = m.R_df[m.R_df == -250.0].index.tolist()
                goal_mrl = m.R_df[m.R_df == 100.0].index.tolist()
                self.shielder = ShieldCrashingMountainCar(self.structure, traps, goal_mrl, self.intervals, self.initial_state)
                self.shielder.calculateShield()
                self.shielder.printShield()
                
                # Run the algoirhtm
                self.pi_b = crashingMountainCarPolicy(self.env, epsilon=epsilon_baseline).compute_baseline_size(len(self.structure))
                self.nb_states = len(self.structure)
                self.data = d_data
                # In this environment, the reward is always 1 for every step, so we create a matrix of shape (nb_states, nb_states) filled with ones
                self.R_state_state = np.ones((self.nb_states, self.nb_states))
                print(self.nb_states)
                print("Running Algorithms")
                self._run_algorithms()
                 
                # ----- GRID ----
                # print("Estimating Intervals")            
                # # Get the intervals from the abstraction using the method from bahdings
                # self._count()
                # # print("done counts")
                # self.estimator = imdp_builder(self.data, self.count_state_action_state, self.count_state_action, self.episodic, beta=1e-4, kstep=1)
                # self.intervals = self.estimator.get_intervals()
                
                
                # # self.estimator = PACIntervalEstimator(self.structure, 0.1, self.data, self.nb_actions, alpha=5)
                # # self.estimator.calculate_intervals()
                # # self.intervals = self.estimator.get_intervals()
                # print("Calculating Shield")  
                # self.structure = self.build_transition_matrix()
                
                # # MOUNTAINCAR
                # self.shielder = ShieldCrashingMountainCar(self.structure, [self.traps], self.goal, self.intervals, self.initial_state)
                # self.shielder.calculateShield()
                # # self.shielder.printShield()
                # print("Running Algorithms")
                # self._run_algorithms()
                

    # def generate_batch(self, nb_trajectories, env, pi, max_steps=1000):
    #     """
    #     Generates a data batch for an episodic MDP.
    #     :param nb_steps: number of steps in the data batch
    #     :param env: environment to be used to generate the batch on
    #     :param pi: policy to be used to generate the data as numpy array with shape (nb_states, nb_actions)
    #     :return: data batch as a list of sublists of the form [state, action, next_state, reward]
    #     """
    #     trajectories = []
    #     trajectories_cont = []
    #     for _ in np.arange(nb_trajectories):
    #         nb_steps = 0
    #         trajectorY = []
    #         env.reset()
    #         env.set_random_state()
    #         state, region = env.get_init_state()
    #         is_done = False
    #         while nb_steps < max_steps and not is_done:
    #             # print("AAAAA")
    #             action_choice = np.random.choice(pi.shape[1], p=pi[region])
    #             state, next_state, reward = env.step(action_choice)
    #             region = env.state2region(state)
    #             next_region= env.state2region(next_state)
    #             is_done = env.is_done()                    
    #             trajectorY.append([action_choice, region, next_region, reward])
    #             crashed = env.check_crashed()
    #             trajectories_cont.append([state, action_choice, next_state, reward, is_done])
    #             region = next_region
    #             nb_steps += 1
    #             # print(f"from state {state}, we reached new state {next_state} using action {action_choice} and got a reward of {reward}")
    #         trajectories.append(trajectorY)
    #     batch_traj = [val for sublist in trajectories for val in sublist]
    #     return trajectories, batch_traj, trajectories_cont
            
    def generate_batch(self, nb_trajectories, env, pi, max_steps=1000):
        """
        Generates a data batch for an episodic MDP.
        :param nb_steps: number of steps in the data batch
        :param env: environment to be used to generate the batch on
        :param pi: policy to be used to generate the data as numpy array with shape (nb_states, nb_actions)
        :return: data batch as a list of sublists of the form [state, action, next_state, reward]
        """
        trajectories = []
        trajectories_cont = []
        for _ in np.arange(nb_trajectories):
            nb_steps = 0
            trajectorY = []
            trajectorY_cont = []
            env.reset()
            env.set_random_state()
            state = env.get_state()
            region = env.state2region(state)
            is_done = False
            while nb_steps < max_steps and not is_done:
                # print("AAAAA")
                action_choice = np.random.choice(pi.shape[1], p=pi[region])
                # if state[1] <0.59:
                    # action_choice = 2
                state, next_state, reward = env.step(action_choice)
                # print(state)
                # print(type(state))
                region = env.state2region(state)
                next_region = env.state2region(next_state)
                is_done = env.is_done()                    
                trajectorY.append([action_choice, region, next_region, reward])
                crashed = env.check_crashed()
                trajectorY_cont.append([state, action_choice, next_state, reward, is_done, crashed])
                region = next_region
                nb_steps += 1
            trajectories_cont.append(trajectorY_cont)
            trajectories.append(trajectorY)
        batch_traj = [val for sublist in trajectories for val in sublist]
        return trajectories, batch_traj, trajectories_cont
    
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
    
    def reduce_transition_matrix(self, transition_matrix):
        """
        Reduces a transition matrix to only include possible end states for each state-action pair.

        Args:
        - transition_matrix (numpy.ndarray): A 3D numpy array of shape (num_states, num_actions, num_states) 
        where each element represents the probability of transitioning from one state to another
        given a certain action.

        Returns:
        - numpy.ndarray: A 3D numpy array of shape (num_states, num_actions, num_possible_transitions) 
        where each element contains the indices of possible end states.
        """
        num_states = len(transition_matrix)
        num_actions = len(transition_matrix[0])
        # Prepare the reduced matrix to hold the indices of possible states
        reduced_matrix = np.empty((num_states, num_actions), dtype=object)
        
        # Loop through each state and action to populate the reduced matrix
        for state in range(num_states):
            for action in range(num_actions):
                # Get indices of nonzero probabilities (possible end states)
                possible_states = np.nonzero(transition_matrix[state, action])[0]
                reduced_matrix[state, action] = np.array(possible_states)
        
        return reduced_matrix
    
    def discretize_data(self, data, predictor):
        data_disc = []
        for trajectory in data:
            traj = []
            for transition in trajectory:
                s = transition[0]
                a = transition[1]
                ns = transition[2]
                r = transition[3]
                terminated = transition[4]
                died = transition[5]
                s_d = state2region(predictor, s, self.dimensions)
                ns_d = state2region(predictor, ns, self.dimensions)
                traj.append([a, s_d, ns_d, r])
            data_disc.append(traj)
        return data_disc
    
    def add_trans_from_data(self,structure, data):
        # num_states = len(structure)
        # num_actions = len(structure[0])
        for trajectory in data:
            for transition in trajectory:
                s=transition[1]
                a=transition[0]
                ns=transition[2]
                poss_next = structure[s,a]
                if not ns in poss_next:
                    poss_next = np.append(poss_next, [int(ns)])
                    structure[s][a] = poss_next
        return structure  
    
    def estimate_transitions(self):
        count = 0
        # Prepare the reduced matrix with empty lists
        transition_matrix = np.empty((self.nb_states, self.nb_actions), dtype=object)
        for s in range(self.nb_states):
            for a in range(self.nb_actions):
                if s == self.traps:
                    transition_matrix[s, a] = [self.traps]
                elif s == self.goal:
                    transition_matrix[s, a] = [self.goal]
                else:
                    transition_matrix[s, a] = list(self.env.get_successor_states(s,a))
                print(s, " = ", transition_matrix[s, a])
        self.transition_matrix = transition_matrix
        
    def build_transition_matrix(self):
        """
        Builds a reduced transition matrix that lists possible next states
        for each (state, action) pair, based on the observed trajectories.

        Returns
        -------
        np.ndarray
            A 2D array of shape (num_states, num_actions), where each entry
            is a list of possible next states for that (state, action).
        """
        count = 0
        # Prepare the reduced matrix with empty lists
        transition_matrix = self.transition_matrix.copy()


        # Fill matrix with next states from counts
        for (state, action, next_state) in self.count_state_action_state.keys():
            if next_state not in transition_matrix[state, action]:
                # print("fuck")
                transition_matrix[state, action].append(next_state)

        # for s in range(self.nb_states):
        #     for a in range(self.nb_actions):
        #         if len(transition_matrix[s, a]) == 0:
        #             transition_matrix[s, a] = [self.traps]

        # for i in range(len(transition_matrix)):
        #     for j in range(len(transition_matrix[i])):
        #         if len(transition_matrix[i][j]) > 1:
        #             count+=1
        # print(transition_matrix)
        # print(count)
        return transition_matrix


import numpy as np
from collections import defaultdict
from scipy.sparse import dok_matrix, identity
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import bicgstab

def policy_evaluation_exact(pi, r, p, gamma):
    if isinstance(p, dict):
        return policy_evaluation_exact_sparse(pi, r, p, gamma)
    return policy_evaluation_exact_dense(pi, r, p, gamma)

def policy_evaluation_exact_sparse(pi, r, p_sparse, gamma):
    """
    Evaluate policy using a sparse transition dictionary.

    Args:
      pi: policy, array of shape [num_states x num_actions]
      r: rewards, array of shape [num_states x num_actions]
      p_sparse: dictionary {(s, a, s'): p(s'|s,a)} with only non-zero transitions
      gamma: discount factor

    Returns:
      v: 1D array with updated state values
      q: 2D array with updated action values
    """
    num_states, num_actions = pi.shape
    # Compute expected rewards under policy
    r_pi = np.einsum('ij,ij->i', pi, r)
    # Build sparse p_pi as a dictionary {(s, s'): prob}
    p_pi_sparse = defaultdict(float)

    for (s, a, s_prime), prob in p_sparse.items():
        p_pi_sparse[(s, s_prime)] += pi[s, a] * prob
    # Convert sparse p_pi to a scipy sparse matrix for solving
    P_pi = dok_matrix((num_states, num_states), dtype=np.float64)
    for (s, s_prime), prob in p_pi_sparse.items():
        P_pi[s, s_prime] = prob
    # Solve (I - gamma * P_pi) * v = r_pi
    A = identity(num_states, format='csr') - gamma * P_pi.tocsr()
    v, info = bicgstab(A, r_pi, atol=1e-6)
    if info != 0: # bicgstab sometimes fails with deterministic policies. If it fails to converge, use spsolve instead, which is slower but does not have this issue
        v = spsolve(A, r_pi)

    # Compute Q-values
    q = np.zeros((num_states, num_actions))
    for (s, a, s_prime), prob in p_sparse.items():
        q[s, a] += prob * v[s_prime]
    q = r + gamma * q
    
    return v, q

def policy_evaluation_exact_dense(pi, r, p, gamma):
    """
    Evaluate policy (from https://github.com/RomainLaroche/SPIBB, but changed to use
    np.linalg.solve instead of the inverse for a higher stability)
    Args:
      pi: policy, array of shape |S| x |A|
      r: the true rewards, array of shape |S| x |A|
      p: the true state transition probabilities, array of shape |S| x |A| x |S|
    Return:
      v: 1D array with updated state values
    """
    # Rewards according to policy: Hadamard product and row-wise sum
    r_pi = np.einsum('ij,ij->i', pi, r)

    # Policy-weighted transitions:
    # multiply p by pi by broadcasting pi, then sum second axis
    # result is an array of shape |S| x |S|
    p_pi = np.einsum('ijk, ij->ik', p, pi)
    # v = np.dot(np.linalg.inv((np.eye(p_pi.shape[0]) - gamma * p_pi)), r_pi)
    # New calculation to make it more stable
    v = np.linalg.solve((np.eye(p_pi.shape[0]) - gamma * p_pi), r_pi)
    return v, r + gamma * np.einsum('i, jki->jk', v, p)

# Translate the names from the algorithms to the class.
algorithm_name_dict = {SPIBB.NAME: SPIBB, Lower_SPIBB.NAME: Lower_SPIBB, 'SPIBB-DQN': spibb_dqn,
                       ApproxSoftSPIBB.NAME: ApproxSoftSPIBB, ExactSoftSPIBB.NAME: ExactSoftSPIBB,
                       AdvApproxSoftSPIBB.NAME: AdvApproxSoftSPIBB, LowerApproxSoftSPIBB.NAME: LowerApproxSoftSPIBB,
                       DUIPI.NAME: DUIPI, shield_DUIPI.NAME: shield_DUIPI, Basic_rl.NAME: Basic_rl, 
                       RMin.NAME: RMin, Shield_RMin.NAME: Shield_RMin, MBIE.NAME: MBIE, shield_MBIE.NAME : shield_MBIE, 
                       RaMDP.NAME: RaMDP, Shield_RaMDP.NAME : Shield_RaMDP, Shield_SPIBB.NAME: Shield_SPIBB, 
                       Shield_Lower_SPIBB.NAME: Shield_Lower_SPIBB,
                       WorstCaseRMDP.NAME : WorstCaseRMDP, Shield_WorstCaseRMDP.NAME : Shield_WorstCaseRMDP, PiStar.NAME: PiStar}




