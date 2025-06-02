import os
import sys
import ast
import time
from distutils import util
import configparser
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import norm

from environments.wet_chicken_discrete.baseline_policy import WetChickenBaselinePolicy
from environments.wet_chicken_discrete.dynamics import WetChicken

from environments.Airplane_discrete.airplane import Airplane
from environments.Airplane_discrete.airplane_baseline_policy import AirplaneBaselinePolicy

from environments.Slippery_gridworld.gridworld import gridWorld
from environments.Slippery_gridworld.gridworld_heuristic_policy import GridworldBaselinePolicy

from environments.pacman.pacman_dynamics_two_ghosts import pacmanSimplified
from environments.pacman.pacman_heuristic_policy import PacmanBaselinePolicy

from environments.read_env_from_prism import prism_env

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
from batch_rl_algorithms.shielded.shielded_spibb import Shield_SPIBB, Shield_Lower_SPIBB
from batch_rl_algorithms.shielded.shielded_duipi import shield_DUIPI
from batch_rl_algorithms.shielded.shielded_raMDP import Shield_RaMDP
from batch_rl_algorithms.shielded.shielded_mbie import shield_MBIE
from batch_rl_algorithms.shielded.shielded_r_min import Shield_RMin


from shield import ShieldRandomMDP, ShieldWetChicken, ShieldAirplane, ShieldSlipperyGridworld, ShieldSimplifiedPacman, ShieldPrism, ShieldTaxi
from PACIntervalEstimator import PACIntervalEstimator
directory = os.path.dirname(os.path.expanduser(__file__))


class Experiment:
    # Class to implement general batch RL experiments
    results = []
    nb_iterations = None
    fixed_params_exp_list = None
    fixed_params_exp_columns = None
    variable_params_exp_columns = None
    algorithms_columns = ['method', 'hyperparam', 'method_perf', 'run_time']

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
        for key in self.algorithms_dict.keys():
            # print("key = ", key)
            if key in {SPIBB.NAME, Lower_SPIBB.NAME}:
                self._run_spibb(key)
            elif key in {ExactSoftSPIBB.NAME, ApproxSoftSPIBB.NAME, LowerApproxSoftSPIBB.NAME,
                         AdvApproxSoftSPIBB.NAME}:
                self._run_soft_spibb(key)
            elif key in {DUIPI.NAME}:
                self._run_duipi(key)
            elif key in {shield_DUIPI.NAME}:
                self._run_duipi_shielded(key)
            elif key in {Basic_rl.NAME}:
                self._run_basic_rl(key)
            elif key in {RMin.NAME}:
                self._run_r_min(key)
            elif key in {Shield_RMin.NAME}:
                self._run_r_min_shielded(key)
            elif key in {RaMDP.NAME}:
                self._run_ramdp(key)
            elif key in {Shield_RaMDP.NAME}:
                self._run_ramdp_shielded(key)
            elif key in {MBIE.NAME}:
                self._run_mbie(key)
            elif key in {shield_MBIE.NAME}:
                self._run_mbie_shielded(key)
            elif key in {Shield_SPIBB.NAME, Shield_Lower_SPIBB.NAME}:
                self._run_spibb_shielded(key)
                # self._run_spibb_experiment(key)
            elif key in {WorstCaseRMDP.NAME}:
                self._run_rmdp(key)
            elif key in {Shield_WorstCaseRMDP.NAME}:
                self._run_rmdp_shielded(key)
            else:
                print("KEY NOT FOUND")

    def _run_rmdp_shielded(self, key):
        """
        Runs rmdp for one data set.
        """
        rmdp = algorithm_name_dict[key](pi_b=self.pi_b, gamma=self.gamma, nb_states=self.nb_states,
                                            nb_actions=self.nb_actions, data=self.data, R=self.R_state_state,
                                            episodic=self.episodic, shield=self.shielder, intervals=self.intervals, speed_up_dict=self.speed_up_dict, estimate_baseline=self.estimate_baseline)
        t_0 = time.time()
        rmdp.fit()
        t_1 = time.time()
        basic_rl_perf = self._policy_evaluation_exact(rmdp.pi)
        method = rmdp.NAME
        method_perf = basic_rl_perf
        hyperparam = None
        run_time = t_1 - t_0
        self.results.append(self.to_append + [method, hyperparam, method_perf, run_time])
    
    def _run_rmdp(self, key):
        """
        Runs rmdp for one data set.
        """
        rmdp = algorithm_name_dict[key](pi_b=self.pi_b, gamma=self.gamma, nb_states=self.nb_states,
                                            nb_actions=self.nb_actions, data=self.data, R=self.R_state_state,
                                            episodic=self.episodic, intervals=self.intervals, speed_up_dict=self.speed_up_dict, estimate_baseline=self.estimate_baseline)
        t_0 = time.time()
        rmdp.fit()
        t_1 = time.time()
        basic_rl_perf = self._policy_evaluation_exact(rmdp.pi)
        method = rmdp.NAME
        method_perf = basic_rl_perf
        hyperparam = None
        run_time = t_1 - t_0
        self.results.append(self.to_append + [method, hyperparam, method_perf, run_time])
        
    def _run_duipi(self, key):
        """
        Runs DUIPI for one data set, with all hyper-parameters and in bayesian and frequentist mode.
        :param key: DUIPI.NAME
        """
        for bayesian_notifier in self.algorithms_dict[key].keys():
            bayesian = bayesian_notifier == 'bayesian'
            if self.safety_deltas:
                xis = [norm.ppf(1 - delta) for delta in self.safety_deltas]
            else:
                xis = self.algorithms_dict[key][bayesian_notifier]
            for i, xi in enumerate(xis):
                duipi = algorithm_name_dict[key](pi_b=self.pi_b, gamma=self.gamma, nb_states=self.nb_states,
                                                 nb_actions=self.nb_actions, data=self.data, R=self.R_state_state,
                                                 xi=xi, episodic=self.episodic, bayesian=bayesian,
                                                 speed_up_dict=self.speed_up_dict, estimate_baseline=self.estimate_baseline)
                t_0 = time.time()
                duipi.fit()
                t_1 = time.time()
                duipi_perf = self._policy_evaluation_exact(duipi.pi)
                if bayesian:
                    name_addition = '_bayesian'
                else:
                    name_addition = '_frequentist'
                method = duipi.NAME + name_addition
                method_perf = duipi_perf
                hyperparam = xi
                run_time = t_1 - t_0
                if self.safety_deltas:
                    self.results.append(
                        self.to_append + [method, hyperparam, method_perf, run_time, self.safety_deltas[i],
                                          duipi.v[self.initial_state] - xi * np.sqrt(
                                              duipi.variance_v[self.initial_state])])
                else:
                    self.results.append(self.to_append + [method, hyperparam, method_perf, run_time])
                    
    def _run_duipi_shielded(self, key):
        """
        Runs DUIPI for one data set, with all hyper-parameters and in bayesian and frequentist mode.
        :param key: shield_DUIPI.NAME
        """
        for bayesian_notifier in self.algorithms_dict[key].keys():
            bayesian = bayesian_notifier == 'bayesian'
            if self.safety_deltas:
                xis = [norm.ppf(1 - delta) for delta in self.safety_deltas]
            else:
                xis = self.algorithms_dict[key][bayesian_notifier]
            for i, xi in enumerate(xis):
                duipi = algorithm_name_dict[key](pi_b=self.pi_b, gamma=self.gamma, nb_states=self.nb_states,
                                                 nb_actions=self.nb_actions, data=self.data, R=self.R_state_state,
                                                 xi=xi, episodic=self.episodic, shield=self.shielder, bayesian=bayesian,
                                                 speed_up_dict=self.speed_up_dict, estimate_baseline=self.estimate_baseline)
                t_0 = time.time()
                duipi.fit()
                t_1 = time.time()
                duipi_perf = self._policy_evaluation_exact(duipi.pi)
                if bayesian:
                    name_addition = '_bayesian'
                else:
                    name_addition = '_frequentist'
                method = duipi.NAME + name_addition
                method_perf = duipi_perf
                hyperparam = xi
                run_time = t_1 - t_0
                if self.safety_deltas:
                    self.results.append(
                        self.to_append + [method, hyperparam, method_perf, run_time, self.safety_deltas[i],
                                          duipi.v[self.initial_state] - xi * np.sqrt(
                                              duipi.variance_v[self.initial_state])])
                else:
                    self.results.append(self.to_append + [method, hyperparam, method_perf, run_time])
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
            spibb_perf = self._policy_evaluation_exact(spibb.pi)
            method = spibb.NAME
            method_perf = spibb_perf
            hyperparam = N_wedge
            run_time = t_1 - t_0
            
            self.results.append(self.to_append + [method, hyperparam, method_perf, run_time])
            
    def _run_spibb_experiment(self, key):
        """
        Runs SPIBB or Lower-SPIBB for one data set, with all hyper-parameters.
        :param key: shield_SPIBB.NAME or shield_Lower_SPIBB.NAME, depending on which algorithm is supposed to be run
        """
        # 1. Modified data
        # print("modified data")
        for N_wedge in self.algorithms_dict[key]['hyperparam']:
            spibb = algorithm_name_dict[key](pi_b=self.pi_b, gamma=self.gamma, nb_states=self.nb_states,
                                             nb_actions=self.nb_actions, data=self.data, R=self.R_state_state,
                                             N_wedge=N_wedge, episodic=self.episodic, shield=self.shielder, 
                                             speed_up_dict=self.speed_up_dict,estimate_baseline=self.estimate_baseline, shield_data=True, shield_action=False)
            t_0 = time.time()
            spibb.fit()
            t_1 = time.time()
            spibb_perf = self._policy_evaluation_exact(spibb.pi)
            method = spibb.NAME + "_Modified_Data"
            method_perf = spibb_perf
            hyperparam = N_wedge
            run_time = t_1 - t_0
            
            self.results.append(self.to_append + [method, hyperparam, method_perf, run_time])
        
        # 2. Shield on Actions 
        # print("shield Actions")
        for N_wedge in self.algorithms_dict[key]['hyperparam']:
            spibb = algorithm_name_dict[key](pi_b=self.pi_b, gamma=self.gamma, nb_states=self.nb_states,
                                             nb_actions=self.nb_actions, data=self.data, R=self.R_state_state,
                                             N_wedge=N_wedge, episodic=self.episodic, shield=self.shielder, 
                                             speed_up_dict=self.speed_up_dict,estimate_baseline=self.estimate_baseline, 
                                             shield_baseline = False, shield_data=False, shield_action=True)
            t_0 = time.time()
            spibb.fit()
            t_1 = time.time()
            spibb_perf = self._policy_evaluation_exact(spibb.pi)
            method = spibb.NAME + "_Shield_Actions"
            method_perf = spibb_perf
            hyperparam = N_wedge
            run_time = t_1 - t_0
            
            self.results.append(self.to_append + [method, hyperparam, method_perf, run_time])
        
        # 3. Shield on Actions + Modified Data
        # print("shield_actions + modified data")
        for N_wedge in self.algorithms_dict[key]['hyperparam']:
            spibb = algorithm_name_dict[key](pi_b=self.pi_b, gamma=self.gamma, nb_states=self.nb_states,
                                             nb_actions=self.nb_actions, data=self.data, R=self.R_state_state,
                                             N_wedge=N_wedge, episodic=self.episodic, shield=self.shielder, 
                                             speed_up_dict=self.speed_up_dict,estimate_baseline=self.estimate_baseline, 
                                             shield_baseline = False, shield_data=True, shield_action=True)
            t_0 = time.time()
            spibb.fit()
            t_1 = time.time()
            spibb_perf = self._policy_evaluation_exact(spibb.pi)
            method = spibb.NAME + "_Shield_Actions+Modified_Data"
            method_perf = spibb_perf
            hyperparam = N_wedge
            run_time = t_1 - t_0
            
            self.results.append(self.to_append + [method, hyperparam, method_perf, run_time])
        
        # 4. Shield on Baseline
        # print("shield baseline")
        for N_wedge in self.algorithms_dict[key]['hyperparam']:
            spibb = algorithm_name_dict[key](pi_b=self.pi_b, gamma=self.gamma, nb_states=self.nb_states,
                                             nb_actions=self.nb_actions, data=self.data, R=self.R_state_state,
                                             N_wedge=N_wedge, episodic=self.episodic, shield=self.shielder, 
                                             speed_up_dict=self.speed_up_dict,estimate_baseline=self.estimate_baseline, 
                                             shield_baseline = True, shield_data=False, shield_action=False)
            t_0 = time.time()
            spibb.fit()
            t_1 = time.time()
            spibb_perf = self._policy_evaluation_exact(spibb.pi)
            method = spibb.NAME + "_Shield_Baseline"
            method_perf = spibb_perf
            hyperparam = N_wedge
            run_time = t_1 - t_0
            
            self.results.append(self.to_append + [method, hyperparam, method_perf, run_time])
        
        # 5. Shield on Baseline + Modified Data
        # print("shield baseline + modified data")
        for N_wedge in self.algorithms_dict[key]['hyperparam']:
            spibb = algorithm_name_dict[key](pi_b=self.pi_b, gamma=self.gamma, nb_states=self.nb_states,
                                             nb_actions=self.nb_actions, data=self.data, R=self.R_state_state,
                                             N_wedge=N_wedge, episodic=self.episodic, shield=self.shielder, 
                                             speed_up_dict=self.speed_up_dict,estimate_baseline=self.estimate_baseline, 
                                             shield_baseline = True, shield_data=True, shield_action=False)
            t_0 = time.time()
            spibb.fit()
            t_1 = time.time()
            spibb_perf = self._policy_evaluation_exact(spibb.pi)
            method = spibb.NAME + "_Shield_Baseline+Modified_Data"
            method_perf = spibb_perf
            hyperparam = N_wedge
            run_time = t_1 - t_0
            
            self.results.append(self.to_append + [method, hyperparam, method_perf, run_time])
        
        # 6. Shield on Baseline + Shield on sactions
        # print("shield baseline + shield actions")
        for N_wedge in self.algorithms_dict[key]['hyperparam']:
            spibb = algorithm_name_dict[key](pi_b=self.pi_b, gamma=self.gamma, nb_states=self.nb_states,
                                             nb_actions=self.nb_actions, data=self.data, R=self.R_state_state,
                                             N_wedge=N_wedge, episodic=self.episodic, shield=self.shielder, 
                                             speed_up_dict=self.speed_up_dict,estimate_baseline=self.estimate_baseline, 
                                             shield_baseline = True, shield_data=False, shield_action=True)
            t_0 = time.time()
            spibb.fit()
            t_1 = time.time()
            spibb_perf = self._policy_evaluation_exact(spibb.pi)
            method = spibb.NAME + "_Shield_Baseline+Shield_Actions"
            method_perf = spibb_perf
            hyperparam = N_wedge
            run_time = t_1 - t_0
            
            self.results.append(self.to_append + [method, hyperparam, method_perf, run_time])
        
        # 7. Shield on Baseline + Shield on actions + Modified Data
        # print("shield baseline + shield actions + modified data")
        for N_wedge in self.algorithms_dict[key]['hyperparam']:
            spibb = algorithm_name_dict[key](pi_b=self.pi_b, gamma=self.gamma, nb_states=self.nb_states,
                                             nb_actions=self.nb_actions, data=self.data, R=self.R_state_state,
                                             N_wedge=N_wedge, episodic=self.episodic, shield=self.shielder, 
                                             speed_up_dict=self.speed_up_dict,estimate_baseline=self.estimate_baseline, 
                                             shield_baseline = True, shield_data=True, shield_action=True)
            t_0 = time.time()
            spibb.fit()
            t_1 = time.time()
            spibb_perf = self._policy_evaluation_exact(spibb.pi)
            method = spibb.NAME + "_Shield_Baseline+Shield_Actions+Modified_Data"
            method_perf = spibb_perf
            hyperparam = N_wedge
            run_time = t_1 - t_0
            
            self.results.append(self.to_append + [method, hyperparam, method_perf, run_time])
        
    
            
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
            spibb_perf = self._policy_evaluation_exact(spibb.pi)
            method = spibb.NAME
            method_perf = spibb_perf
            hyperparam = N_wedge
            run_time = t_1 - t_0
            self.results.append(self.to_append + [method, hyperparam, method_perf, run_time])
            
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
                                self.to_append + [method, hyperparam, method_perf, run_time, delta, bound])
                        else:
                            self.results.append(self.to_append + [method, hyperparam, method_perf, run_time])

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
        self.results.append(self.to_append + [method, hyperparam, method_perf, run_time])
        
    def _run_r_min(self, key):
        """
        Runs R-MIN for one data set, with all hyper-parameters.
        :param key: RMIN.NAME
        """
        for N_wedge in self.algorithms_dict[key]['hyperparam']:
            r_min = algorithm_name_dict[key](pi_b=self.pi_b, gamma=self.gamma, nb_states=self.nb_states,
                                             nb_actions=self.nb_actions, data=self.data, R=self.R_state_state,
                                             N_wedge=N_wedge, episodic=self.episodic, speed_up_dict=self.speed_up_dict,
                                             estimate_baseline=self.estimate_baseline)
            t_0 = time.time()
            r_min.fit()
            t_1 = time.time()
            r_min_perf = self._policy_evaluation_exact(r_min.pi)
            method = r_min.NAME
            method_perf = r_min_perf
            hyperparam = N_wedge
            run_time = t_1 - t_0
            self.results.append(self.to_append + [method, hyperparam, method_perf, run_time])
            
    def _run_r_min_shielded(self, key):
        """
        Runs R-MIN for one data set, with all hyper-parameters.
        :param key: RMIN.NAME
        """
        for N_wedge in self.algorithms_dict[key]['hyperparam']:
            r_min = algorithm_name_dict[key](pi_b=self.pi_b, gamma=self.gamma, nb_states=self.nb_states,
                                             nb_actions=self.nb_actions, data=self.data, R=self.R_state_state,
                                             N_wedge=N_wedge, episodic=self.episodic, shield=self.shielder, speed_up_dict=self.speed_up_dict,
                                             estimate_baseline=self.estimate_baseline)
            t_0 = time.time()
            r_min.fit()
            t_1 = time.time()
            r_min_perf = self._policy_evaluation_exact(r_min.pi)
            method = r_min.NAME
            method_perf = r_min_perf
            hyperparam = N_wedge
            run_time = t_1 - t_0
            self.results.append(self.to_append + [method, hyperparam, method_perf, run_time])

    def _run_mbie(self, key):
        """
        Runs MBIE for one data set, with all hyper-parameters.
        :param key: MBIE.NAME
        """
        if self.safety_deltas:
            deltas = self.safety_deltas
        else:
            deltas = self.algorithms_dict[key]['deltas']
        for delta in deltas:
            mbie = algorithm_name_dict[key](pi_b=self.pi_b, gamma=self.gamma, nb_states=self.nb_states,
                                            nb_actions=self.nb_actions, data=self.data, R=self.R_state_state,
                                            delta=delta, episodic=self.episodic, speed_up_dict=self.speed_up_dict,
                                            estimate_baseline=self.estimate_baseline)
            t_0 = time.time()
            mbie.fit()
            t_1 = time.time()
            mbie_perf = self._policy_evaluation_exact(mbie.pi)
            method = mbie.NAME
            method_perf = mbie_perf
            hyperparam = delta
            run_time = t_1 - t_0
            if self.safety_deltas:
                bound = mbie.get_v[self.initial_state]
                self.results.append(self.to_append + [method, hyperparam, method_perf, run_time, delta, bound])
            else:
                self.results.append(self.to_append + [method, hyperparam, method_perf, run_time])
                
    def _run_mbie_shielded(self, key):
        """
        Runs MBIE for one data set, with all hyper-parameters.
        :param key: MBIE.NAME
        """
        if self.safety_deltas:
            deltas = self.safety_deltas
        else:
            deltas = self.algorithms_dict[key]['deltas']
        for delta in deltas:
            mbie = algorithm_name_dict[key](pi_b=self.pi_b, gamma=self.gamma, nb_states=self.nb_states,
                                            nb_actions=self.nb_actions, data=self.data, R=self.R_state_state,
                                            delta=delta, episodic=self.episodic, shield=self.shielder, speed_up_dict=self.speed_up_dict,
                                            estimate_baseline=self.estimate_baseline)
            t_0 = time.time()
            mbie.fit()
            t_1 = time.time()
            mbie_perf = self._policy_evaluation_exact(mbie.pi)
            method = mbie.NAME
            method_perf = mbie_perf
            hyperparam = delta
            run_time = t_1 - t_0
            if self.safety_deltas:
                bound = mbie.get_v[self.initial_state]
                self.results.append(self.to_append + [method, hyperparam, method_perf, run_time, delta, bound])
            else:
                self.results.append(self.to_append + [method, hyperparam, method_perf, run_time])

    def _run_ramdp(self, key):
        """
        Runs RaMDP for one data set, with all hyper-parameters.
        :param key: RaMDP.NAME
        """
        for kappa in self.algorithms_dict[key]['hyperparam']:
            ramdp = algorithm_name_dict[key](pi_b=self.pi_b, gamma=self.gamma, nb_states=self.nb_states,
                                             nb_actions=self.nb_actions, data=self.data, R=self.R_state_state,
                                             kappa=kappa, episodic=self.episodic, speed_up_dict=self.speed_up_dict,
                                             estimate_baseline=self.estimate_baseline)
            t_0 = time.time()
            ramdp.fit()
            t_1 = time.time()
            ramdp_perf = self._policy_evaluation_exact(ramdp.pi)
            method = ramdp.NAME
            method_perf = ramdp_perf
            hyperparam = kappa
            run_time = t_1 - t_0
            self.results.append(self.to_append + [method, hyperparam, method_perf, run_time])
            
    def _run_ramdp_shielded(self, key):
        """
        Runs RaMDP for one data set, with all hyper-parameters.
        :param key: RaMDP.NAME
        """
        for kappa in self.algorithms_dict[key]['hyperparam']:
            ramdp = algorithm_name_dict[key](pi_b=self.pi_b, gamma=self.gamma, nb_states=self.nb_states,
                                             nb_actions=self.nb_actions, data=self.data, R=self.R_state_state,
                                             kappa=kappa, episodic=self.episodic, shield=self.shielder, speed_up_dict=self.speed_up_dict,
                                             estimate_baseline=self.estimate_baseline)
            t_0 = time.time()
            ramdp.fit()
            t_1 = time.time()
            ramdp_perf = self._policy_evaluation_exact(ramdp.pi)
            method = ramdp.NAME
            method_perf = ramdp_perf
            hyperparam = kappa
            run_time = t_1 - t_0
            self.results.append(self.to_append + [method, hyperparam, method_perf, run_time])

    def _policy_evaluation_exact(self, pi):
        """
        Evaluates policy pi exactly.
        :param pi: policy as numpy array with shape (nb_states, nb_actions)
        """
        return policy_evaluation_exact(pi, self.R_state_action, self.P, self.gamma)[0][self.initial_state]
    
    def compute_r_state_action(self, P, R):
        if isinstance(P, dict):
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


class SimplifiedPacmanExperiment(Experiment):
    fixed_params_exp_columns = ['seed', 'gamma', 'width', 'height', 'lag_p', 'pi_rand_perf', 'pi_star_perf']   
    variable_params_exp_columns = ['iteration', 'epsilon_baseline', 'pi_b_perf', 'nb_trajectories'] 
    
    def _set_env_params(self):
        self.episodic = True
        self.gamma = float(self.experiment_config['ENV_PARAMETERS']['GAMMA'])
        self.lag = float(self.experiment_config['ENV_PARAMETERS']['LAG'])
        self.env = pacmanSimplified(self.lag)
        self.width = self.env.width
        self.height = self.env.height
        
        self.nb_states = self.env.nb_states
        self.nb_actions = 4
        
        self.initial_state = self.env.state_to_int()
        print("init = ", self.initial_state)
        print("create transition function")
        self.P = self.env.get_transition_function()
        self.traps = []
        self.goal = []
        for state in range(self.nb_states):
            x,y,gx1,gy1,gx2,gy2 = self.env.decode_int(state)
            if self.env._is_terminal_state(x,y,gx1,gy1,gx2,gy2):
                if self.env.get_reward_from_int(state) > 0:
                    self.goal.append(state)
                else:
                    self.traps.append(state)
        
        print("create reward function")
        self.R_state_state = self.env.get_reward_function()
        self.R_state_action = self.compute_r_state_action(self.P, self.R_state_state)
        
        # for i, state in enumerate(self.R_state_action):
        #     print(f"for state {i} we have {state}")
        
        self.fixed_params_exp_list = [self.seed, self.gamma, self.width, self.height, self.lag]
        
        pi_rand = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        print("calcing pi rand")
        pi_rand_perf = self._policy_evaluation_exact(pi_rand)
        # pi_rand_perf = 0
        print("pi_rand_perf:", pi_rand_perf)
        print("calcing pi perf")

        self.fixed_params_exp_list.append(pi_rand_perf)
        
        pi_star = PiStar(pi_b=None, gamma=self.gamma, nb_states=self.nb_states, nb_actions=self.nb_actions,
                         data=[[]], R=self.R_state_state, episodic=self.episodic, P=self.P)
        pi_star.fit()    
            
        pi_star_perf = self._policy_evaluation_exact(pi_star.pi)
        # pi_star_perf = 7.7
        print("pi_star_perf:", pi_star_perf)
        self.fixed_params_exp_list.append(pi_star_perf)
        
        pi_b = PacmanBaselinePolicy(env=self.env, epsilon=0).pi
        pi_b_perf = self._policy_evaluation_exact(pi_b)
        print("pi_b_perf:", pi_b_perf)
        self.epsilons_baseline = ast.literal_eval(self.experiment_config['BASELINE']['epsilons_baseline'])
        self.nb_trajectories_list = ast.literal_eval(self.experiment_config['BASELINE']['nb_trajectories_list'])

        self.estimate_baseline=bool((util.strtobool(self.experiment_config['ENV_PARAMETERS']['estimate_baseline'])))
        
    def generate_batch(self, trajectories, nb_trajectories, env, pi, max_steps=2500):
        """
        Generates a data batch for an episodic MDP.
        :param nb_steps: number of steps in the data batch
        :param env: environment to be used to generate the batch on
        :param pi: policy to be used to generate the data as numpy array with shape (nb_states, nb_actions)
        :return: data batch as a list of sublists of the form [state, action, next_state, reward]
        """
        current_nb_trajectories = len(trajectories)
        nb_itt =nb_trajectories - current_nb_trajectories
        # print(pi)
        env.reset()
        for _ in np.arange(nb_itt):
            nb_steps = 0
            trajectorY = []
            
            state = env.state_to_int()
            is_done = False
            while nb_steps < max_steps and not is_done:
                action_choice = np.random.choice(pi.shape[1], p=pi[state])
                state, next_state, reward = env.step(action_choice)
                # print(f"in state {env.decode_int(state)}, action {action_choice} leads to state {env.decode_int(next_state)} with reward of {reward}")
                is_done = env.is_done()                    
                # print(is_done)
                trajectorY.append([action_choice, state, next_state, reward])
                state = next_state
                nb_steps += 1
                    
            trajectories.append(trajectorY)
            env.reset()
            env.set_random_state()
        batch_traj = [val for sublist in trajectories for val in sublist]
        return trajectories, batch_traj
    
    def _run_one_iteration(self):
        for epsilon_baseline in self.epsilons_baseline:
            print(f'Process with seed {self.seed} starting with epsilon_baseline {epsilon_baseline} out of'
                  f' {self.epsilons_baseline}')
            
            print("creating Baseline Policy")
            self.pi_b = PacmanBaselinePolicy(env=self.env, epsilon=epsilon_baseline).pi
            self.to_append_run_one_iteration = self.to_append_run + [epsilon_baseline,
                                                                        self._policy_evaluation_exact(self.pi_b)]
            trajectories = []
            for nb_trajectories in self.nb_trajectories_list:
                print(
                    f'Process with seed {self.seed} starting with nb_trajectories {nb_trajectories} out of '
                    f'{self.nb_trajectories_list}')
                # Generate trajectories, both stored as trajectories and (s,a,s',r) transition samples

                print("Generating Trajectories")
                trajectories, batch_traj = self.generate_batch(trajectories, nb_trajectories, self.env, self.pi_b)
                self.data = trajectories
                self.to_append = self.to_append_run_one_iteration + [nb_trajectories]

                # print("----------------------------------------------------------------")    
                print("Estimating Intervals")  
          
                self.structure = self.reduce_transition_matrix(self.P)   
                self.estimator = PACIntervalEstimator(self.structure, 0.1, self.data, self.nb_actions, alpha=2)
                self.estimator.calculate_intervals()
                self.intervals = self.estimator.get_intervals()
                # print(intervals)
                print("Calculating Shield")  
                self.shielder = ShieldSimplifiedPacman(self.structure, self.traps, self.goal, self.intervals, self.width, self.height)
                self.shielder.calculateShield()

                # self.shielder.printShield()
                # print("----------------------------------------------------------------")
                print("Running Algorithms")
                self._run_algorithms()

                
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

        # Prepare the reduced matrix to hold the indices of possible states
        reduced_matrix = np.empty((self.nb_states, self.nb_actions), dtype=object)

        for state in range(self.nb_states):
            for action in range(self.nb_actions):
                reduced_matrix[state, action] = []
        
        # Loop through each state and action to populate the reduced matrix
        for (state,action,next_state) in transition_matrix.keys():
            reduced_matrix[state][action].append(next_state)
            
        # for state in range(num_states):
        #     for action in range(num_actions):
        #         # Get indices of nonzero probabilities (possible end states)
        #         possible_states = np.nonzero(transition_matrix[state, action])[0]
        #         reduced_matrix[state, action] = np.array(possible_states)
        return reduced_matrix
        
class SlipperyGridworldExperiment(Experiment):
    fixed_params_exp_columns = ['seed', 'gamma', 'width', 'height', 'slip_p', 'escape_p', 'pi_rand_perf', 'pi_star_perf']   
    variable_params_exp_columns = ['iteration', 'epsilon_baseline', 'pi_b_perf', 'nb_trajectories']     
    
    def _set_env_params(self):
        self.episodic = True
        self.gamma = float(self.experiment_config['ENV_PARAMETERS']['GAMMA'])
        self.width = int(self.experiment_config['ENV_PARAMETERS']['WIDTH'])
        self.height = int(self.experiment_config['ENV_PARAMETERS']['HEIGHT'])
        self.slip_p = float(self.experiment_config['ENV_PARAMETERS']['SLIP_P'])
        self.escape_p = float(self.experiment_config['ENV_PARAMETERS']['ESCAPE_P'])
        
        self.nb_states = self.width * self.height
        self.nb_actions = 4
        
        self.env = gridWorld(self.height, self.width, self.slip_p, self.escape_p)
        self.initial_state = self.env.get_state_int()
        self.P = self.env.get_transition_function()
        self.traps = self.env.get_traps()
        self.goal = self.env.get_int_from_state(self.env.goal)
        # print("real transition model = ")
        # print(self.P)
        # for state in range(len(self.P)):
        #     print(f"in state {self.env.decode_int(state)}, we have the following actions")
        #     for action in range(len(self.P[state])):
        #         print(self.P[state][action])
        self.R_state_state = self.env.get_reward_function()
        # print(self.R_state_state)
        self.R_state_action = self.compute_r_state_action(self.P, self.R_state_state)
        self.fixed_params_exp_list = [self.seed, self.gamma, self.width, self.height, self.slip_p, self.escape_p]
        
        pi_rand = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        pi_rand_perf = self._policy_evaluation_exact(pi_rand)
        print("pi_Rand_perf:", pi_rand_perf)
        self.fixed_params_exp_list.append(pi_rand_perf)
        
        pi_star = PiStar(pi_b=None, gamma=self.gamma, nb_states=self.nb_states, nb_actions=self.nb_actions,
                         data=[[]], R=self.R_state_state, episodic=self.episodic, P=self.P)
        print("pi-star")
        pi_star.fit()    
            
        pi_star_perf = self._policy_evaluation_exact(pi_star.pi)
        print("pi_star_perf:", pi_star_perf)
        self.fixed_params_exp_list.append(pi_star_perf)

        self.epsilons_baseline = ast.literal_eval(self.experiment_config['BASELINE']['epsilons_baseline'])
        self.nb_trajectories_list = ast.literal_eval(self.experiment_config['BASELINE']['nb_trajectories_list'])

        self.estimate_baseline=bool((util.strtobool(self.experiment_config['ENV_PARAMETERS']['estimate_baseline'])))
        
    def generate_batch(self, trajectories, nb_trajectories, env, pi, max_steps=2500):
        """
        Generates a data batch for an episodic MDP.
        :param nb_steps: number of steps in the data batch
        :param env: environment to be used to generate the batch on
        :param pi: policy to be used to generate the data as numpy array with shape (nb_states, nb_actions)
        :return: data batch as a list of sublists of the form [state, action, next_state, reward]
        """
        current_trajectories = len(trajectories)
        nb_itt = nb_trajectories - current_trajectories
        # print(pi)
        counter = 0
        for _ in np.arange(nb_itt):
            nb_steps = 0
            trajectorY = []
            env.reset()
            state = env.get_state_int()
            is_done = False
            while nb_steps < max_steps and not is_done:
                # print(pi[state])
                action_choice = np.random.choice(pi.shape[1], p=pi[state])
                state, next_state, reward = env.step(action_choice)
                # print(f"in state {env.decode_int(state)}, action {action_choice} leads to state {env.decode_int(next_state)} with reward of {reward}")
                is_done = env.is_finished()                    
                # print(is_done)
                trajectorY.append([action_choice, state, next_state, reward])
                state = next_state
                nb_steps += 1
            trajectories.append(trajectorY)
            if is_done:
                counter+=1
        print(counter)
        batch_traj = [val for sublist in trajectories for val in sublist]
        return trajectories, batch_traj  

    def _run_one_iteration(self):
        for epsilon_baseline in self.epsilons_baseline:
            print(f'Process with seed {self.seed} starting with epsilon_baseline {epsilon_baseline} out of'
                  f' {self.epsilons_baseline}')
            
            self.pi_b = GridworldBaselinePolicy(env=self.env, epsilon=epsilon_baseline).pi
            self.to_append_run_one_iteration = self.to_append_run + [epsilon_baseline,
                                                                        self._policy_evaluation_exact(self.pi_b)]
            trajectories = []
            for nb_trajectories in self.nb_trajectories_list:
                print(
                    f'Process with seed {self.seed} starting with nb_trajectories {nb_trajectories} out of '
                    f'{self.nb_trajectories_list}')
                # Generate trajectories, both stored as trajectories and (s,a,s',r) transition samples
                print("Generating Trajectories")
                print("before ", len(trajectories))
                trajectories, batch_traj = self.generate_batch(trajectories, nb_trajectories, self.env, self.pi_b)
                print("after ", len(trajectories))
                self.data = trajectories
                self.to_append = self.to_append_run_one_iteration + [nb_trajectories]

                # print("----------------------------------------------------------------")    
                print("Estimating Intervals")            
                self.structure = self.reduce_transition_matrix(self.P)   
                self.estimator = PACIntervalEstimator(self.structure, 0.1, self.data, self.nb_actions, alpha=2)
                self.estimator.calculate_intervals()
                self.intervals = self.estimator.get_intervals()
                # print(intervals)
                print("Calculating Shield")  
                self.shielder = ShieldSlipperyGridworld(self.structure, self.traps, [self.goal], self.intervals, self.width, self.height)
                self.shielder.calculateShield()
                # self.shielder.printShield()
                # print("----------------------------------------------------------------")
                print("Running Algorithms")
                self._run_algorithms()
                
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
        num_states = self.nb_states
        num_actions = self.nb_actions
        # Prepare the reduced matrix to hold the indices of possible states
        reduced_matrix = np.empty((num_states, num_actions), dtype=object)
        
        # Loop through each state and action to populate the reduced matrix
        for state in range(num_states):
            for action in range(num_actions):
                # Get indices of nonzero probabilities (possible end states)
                possible_states = np.nonzero(transition_matrix[state, action])[0]
                reduced_matrix[state, action] = np.array(possible_states)
        
        return reduced_matrix

class AirplaneExperiment(Experiment):
    fixed_params_exp_columns = ['seed', 'gamma', 'maxX', 'maxY', 'response', 'adv_prob', 'pi_rand_perf', 'pi_star_perf']   
    variable_params_exp_columns = ['iteration', 'epsilon_baseline', 'pi_b_perf', 'nb_trajectories']

    def _set_env_params(self):
        self.episodic = True
        self.gamma = float(self.experiment_config['ENV_PARAMETERS']['GAMMA'])
        self.maxX = int(self.experiment_config['ENV_PARAMETERS']['MAXX'])
        self.maxY = int(self.experiment_config['ENV_PARAMETERS']['MAXY'])
        self.response = float(self.experiment_config['ENV_PARAMETERS']['RESPONSE'])
        self.adv_prob = float(self.experiment_config['ENV_PARAMETERS']['ADV_PROB'])
        
        self.nb_states = self.maxX * self.maxY * self.maxY
        self.nb_actions = 3
        
        self.env = Airplane(self.maxY, self.maxX, self.response, self.adv_prob)
        self.initial_state = self.env.get_state_int()
        self.P = self.env.get_transition_function()
        print("real transition model = ")
        # print(self.P)
        # for state in range(len(self.P)):
        #     print(f"in state {self.env.decode_int(state)}, we have the following actions")
        #     for action in range(len(self.P[state])):
        #         print(self.P[state][action])
        self.R_state_state = self.env.get_reward_function()
        # print(self.R_state_state)
        self.R_state_action = self.compute_r_state_action(self.P, self.R_state_state)
        self.set_success_and_crash_states()
        self.fixed_params_exp_list = [self.seed, self.gamma, self.maxX, self.maxY, self.response, self.adv_prob]
        
        pi_rand = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        pi_rand_perf = self._policy_evaluation_exact(pi_rand)
        print("pi_Rand_perf:", pi_rand_perf)
        self.fixed_params_exp_list.append(pi_rand_perf)
        
        pi_star = PiStar(pi_b=None, gamma=self.gamma, nb_states=self.nb_states, nb_actions=self.nb_actions,
                         data=[[]], R=self.R_state_state, episodic=self.episodic, P=self.P)
        pi_star.fit()
        pi_star_perf = self._policy_evaluation_exact(pi_star.pi)
        print("pi_star_perf:", pi_star_perf)
        self.fixed_params_exp_list.append(pi_star_perf)

        self.epsilons_baseline = ast.literal_eval(self.experiment_config['BASELINE']['epsilons_baseline'])
        self.nb_trajectories_list = ast.literal_eval(self.experiment_config['BASELINE']['nb_trajectories_list'])

        self.estimate_baseline=bool((util.strtobool(self.experiment_config['ENV_PARAMETERS']['estimate_baseline'])))
        
    def set_success_and_crash_states(self):
        success = []
        crash = []
        for state in range(self.nb_states):
            x, y, ay = self.env.decode_int(state)
            if self.env.is_done(x):
                reward = self.env.get_reward(x, y, ay)
                if reward > 0:
                    success.append(state)
                else:
                    crash.append(state)
        self.success = success
        self.crash = crash
        
    def _run_one_iteration(self):
        for epsilon_baseline in self.epsilons_baseline:
            print(f'Process with seed {self.seed} starting with epsilon_baseline {epsilon_baseline} out of'
                  f' {self.epsilons_baseline}')
            
            self.pi_b = AirplaneBaselinePolicy(env=self.env, epsilon=epsilon_baseline).pi
            self.to_append_run_one_iteration = self.to_append_run + [epsilon_baseline,
                                                                        self._policy_evaluation_exact(self.pi_b)]
            
            
            for nb_trajectories in self.nb_trajectories_list:
                print(
                    f'Process with seed {self.seed} starting with nb_trajectories {nb_trajectories} out of '
                    f'{self.nb_trajectories_list}')
                # Generate trajectories, both stored as trajectories and (s,a,s',r) transition samples
                self.data, batch_traj = self.generate_batch(nb_trajectories, self.env, self.pi_b)
                self.to_append = self.to_append_run_one_iteration + [nb_trajectories]

                # print("----------------------------------------------------------------")                
                self.structure = self.reduce_transition_matrix(self.P)   
                self.estimator = PACIntervalEstimator(self.structure, 0.1, self.data, self.nb_actions, alpha=2)
                self.estimator.calculate_intervals()
                self.intervals = self.estimator.get_intervals()
                # print(intervals)
                self.shielder = ShieldAirplane(self.structure, self.crash, self.success, self.intervals, self.maxX, self.maxY)
                self.shielder.calculateShield()
                # self.shielder.printShield()
                # print("----------------------------------------------------------------")
                self._run_algorithms()
                
    
    def generate_batch(self, nb_trajectories, env, pi, max_steps=50):
        """
        Generates a data batch for an episodic MDP.
        :param nb_steps: number of steps in the data batch
        :param env: environment to be used to generate the batch on
        :param pi: policy to be used to generate the data as numpy array with shape (nb_states, nb_actions)
        :return: data batch as a list of sublists of the form [state, action, next_state, reward]
        """
        trajectories = []
        # print(pi)
        for _ in np.arange(nb_trajectories):
            nb_steps = 0
            trajectorY = []
            env.reset()
            state = env.get_state_int()
            is_done = False
            while nb_steps < max_steps and not is_done:
                # print(pi[state])
                action_choice = np.random.choice(pi.shape[1], p=pi[state])
                state, next_state, reward = env.step(action_choice)
                # print(f"in state {env.decode_int(state)}, action {action_choice} leads to state {env.decode_int(next_state)} with reward of {reward}")
                is_done = env.is_done_state()
                # print(is_done)
                trajectorY.append([action_choice, state, next_state, reward])
                state = next_state
                nb_steps += 1
            trajectories.append(trajectorY)
        batch_traj = [val for sublist in trajectories for val in sublist]
        return trajectories, batch_traj
    
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
        num_states = self.nb_states
        num_actions = self.nb_actions
        # Prepare the reduced matrix to hold the indices of possible states
        reduced_matrix = np.empty((num_states, num_actions), dtype=object)
        
        # Loop through each state and action to populate the reduced matrix
        for state in range(num_states):
            for action in range(num_actions):
                # Get indices of nonzero probabilities (possible end states)
                possible_states = np.nonzero(transition_matrix[state, action])[0]
                reduced_matrix[state, action] = np.array(possible_states)
        
        return reduced_matrix

class WetChickenExperiment(Experiment):
    # Inherits from the base class Experiment to implement the Wet Chicken experiment specifically.
    fixed_params_exp_columns = ['seed', 'gamma', 'length', 'width', 'max_turbulence', 'max_velocity', 'baseline_method',
                                'pi_rand_perf', 'pi_star_perf']
    
    def _set_env_params(self):
        """
        Reads in all parameters necessary from self.experiment_config to set up the Wet Chicken experiment.
        """
        self.episodic = False
        self.gamma = float(self.experiment_config['ENV_PARAMETERS']['GAMMA'])
        self.length = int(self.experiment_config['ENV_PARAMETERS']['LENGTH'])
        self.width = int(self.experiment_config['ENV_PARAMETERS']['WIDTH'])
        self.max_turbulence = float(self.experiment_config['ENV_PARAMETERS']['MAX_TURBULENCE'])
        self.max_velocity = float(self.experiment_config['ENV_PARAMETERS']['MAX_VELOCITY'])

        self.nb_states = self.length * self.width + 1
        self.nb_actions = 5

        self.env = WetChicken(length=self.length, width=self.width, max_turbulence=self.max_turbulence,
                              max_velocity=self.max_velocity)
        self.initial_state = self.env.get_state_int()
        self.P = self.env.get_transition_function()
        # for state in range(len(self.P)):
        #     x = int(state / self.length)
        #     y = state % self.width
        #     print(f"in state {x},{y}, we have the following actions")
        #     for action in range(len(self.P[state])):
        #         print(self.P[state][action])
        self.R_state_state = self.env.get_reward_function()
        
        self.R_state_action = self.compute_r_state_action(self.P, self.R_state_state)
        self.baseline_method = self.experiment_config['BASELINE']['method']
        self.fixed_params_exp_list = [self.seed, self.gamma, self.length, self.width, self.max_turbulence,
                                      self.max_velocity, self.baseline_method]

        pi_rand = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        pi_rand_perf = self._policy_evaluation_exact(pi_rand)
        print(f"pi_rand_perf = {pi_rand_perf}")
        self.fixed_params_exp_list.append(pi_rand_perf)

        pi_star = PiStar(pi_b=None, gamma=self.gamma, nb_states=self.nb_states, nb_actions=self.nb_actions,
                         data=[[]], R=self.R_state_state, episodic=self.episodic, P=self.P)
        pi_star.fit()
        pi_star_perf = self._policy_evaluation_exact(pi_star.pi)
        print(f"pi_star_perf = {pi_star_perf}")
        self.fixed_params_exp_list.append(pi_star_perf)

        self.epsilons_baseline = ast.literal_eval(self.experiment_config['BASELINE']['epsilons_baseline'])
        self.lengths_trajectory = ast.literal_eval(self.experiment_config['BASELINE']['lengths_trajectory'])
        if self.baseline_method == 'heuristic':
            self.variable_params_exp_columns = ['i', 'epsilon_baseline', 'pi_b_perf', 'length_trajectory']
        else:
            self.learning_rates = ast.literal_eval(self.experiment_config['BASELINE']['learning_rates'])
            self.variable_params_exp_columns = ['i', 'epsilon_baseline', 'learning_rate', 'pi_b_perf',
                                                'length_trajectory']
        self.estimate_baseline=bool((util.strtobool(self.experiment_config['ENV_PARAMETERS']['estimate_baseline'])))

        
    def _run_one_iteration(self):
        """
        Runs one iteration on the Wet Chicken benchmark, so iterates through different baseline and data set parameters
        and then starts the computation for each algorithm.
        """

        for epsilon_baseline in self.epsilons_baseline:
            print(f'Process with seed {self.seed} starting with epsilon_baseline {epsilon_baseline} out of'
                  f' {self.epsilons_baseline}')
            if self.baseline_method == 'heuristic':
                self.pi_b = WetChickenBaselinePolicy(env=self.env, gamma=self.gamma, method=self.baseline_method,
                                                     epsilon=epsilon_baseline).pi
                self.to_append_run_one_iteration = self.to_append_run + [epsilon_baseline,
                                                                         self._policy_evaluation_exact(self.pi_b)]
                for length_trajectory in self.lengths_trajectory:
                    print(f'Starting with length_trajectory {length_trajectory} out of {self.lengths_trajectory}.')
                    self.data = self.generate_batch(length_trajectory, self.env, self.pi_b)
                    self.to_append = self.to_append_run_one_iteration + [length_trajectory]
                    # print("----------------------------------------------------------------")    
                    self.structure = self.reduce_transition_matrix(self.P)
                    # actions = [
                    #     "drift",
                    #     "hold",
                    #     "paddle_back",
                    #     "right",
                    #     "left",
                    # ]
                    # for state in range(len(self.structure)):
                    #     for action in range(len(self.structure[state])):
                    #         x = int(state / 5)
                    #         y = state % 5
                    #         print(f"In state ({x},{y}), using action {actions[action]} has the following possible next states: {self.structure[state][action]}")
                    goal_states = self.find_closest_states(list(range(len(self.structure))), self.length)
                    self.estimator = PACIntervalEstimator(self.structure, 0.1, [self.data], self.nb_actions, alpha=10)
                    self.intervals = self.estimator.get_intervals()
                    self.shielder = ShieldWetChicken(self.structure, self.width, self.length, goal_states, self.intervals)
                    self.shielder.calculateShield()
                    # self.shielder.printShield()
                    # print("----------------------------------------------------------------")
                    self._run_algorithms()
                

    def generate_batch(self, nb_steps, env, pi):
        """
        Generates a data batch for a non-episodic MDP.
        :param nb_steps: number of steps in the data batch
        :param env: environment to be used to generate the batch on
        :param pi: policy to be used to generate the data as numpy array with shape (nb_states, nb_actions)
        :return: data batch as a list of sublists of the form [state, action, next_state, reward]
        """
        trajectory = []
        state = env.get_state_int()
        for _ in np.arange(nb_steps):
            action_choice = np.random.choice(pi.shape[1], p=pi[state])
            state, reward, next_state = env.step(action_choice)
            trajectory.append([action_choice, state, next_state, reward])
            state = next_state
            if state == self.length * self.width:
                state, reward, next_state = env.step(action_choice)
                trajectory.append([0, state, next_state, reward])
        return trajectory
            

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

        # Prepare the reduced matrix to hold the indices of possible states
        reduced_matrix = np.empty((self.nb_states, self.nb_actions), dtype=object)

        for state in range(self.nb_states):
            for action in range(self.nb_actions):
                reduced_matrix[state, action] = []
        
        # Loop through each state and action to populate the reduced matrix
        for (state,action,next_state) in transition_matrix.keys():
            reduced_matrix[state][action].append(next_state)
            
        # for state in range(num_states):
        #     for action in range(num_actions):
        #         # Get indices of nonzero probabilities (possible end states)
        #         possible_states = np.nonzero(transition_matrix[state, action])[0]
        #         reduced_matrix[state, action] = np.array(possible_states)
        return reduced_matrix
    
    def find_closest_states(self, states_list, length):
        """
        Given a list of states in single integer representation, identifies the states where the boat
        falls off the waterfall (i.e., where x > 4).

        Parameters:
            states (list[int]): List of states in single integer representation.
            width (int): The width of the river, used to calculate the x-coordinate.

        Returns:
            list[int]: List of states where the boat would fall off the waterfall.
        """
        falling_states = []
        
        for state in states_list:
            x = state // length  # Calculate the x-coordinate (position along the river)
            
            if x >= length-1:  # The boat falls off the waterfall if x > 4
                falling_states.append(state)
        
        return falling_states
    
class RandomMDPsExperiment(Experiment):
    # Inherits from the base class Experiment to implement the Wet Chicken experiment specifically.
    fixed_params_exp_columns = ['seed', 'gamma', 'nb_states', 'nb_actions', 'nb_next_state_transition']
    variable_params_exp_columns = ['iteration', 'softmax_target_perf_ratio',
                                   'baseline_target_perf_ratio', 'baseline_perf', 'pi_rand_perf', 'pi_star_perf',
                                   'nb_trajectories']

    def _set_env_params(self):
        """
        Reads in all parameters necessary from self.experiment_config to set up the Random MDPs experiment.
        """
        self.episodic = True
        self.gamma = float(self.experiment_config['ENV_PARAMETERS']['GAMMA'])
        self.nb_states = int(self.experiment_config['ENV_PARAMETERS']['nb_states'])
        self.nb_actions = int(self.experiment_config['ENV_PARAMETERS']['nb_actions'])
        self.nb_next_state_transition = int(self.experiment_config['ENV_PARAMETERS']['nb_next_state_transition'])
        self.env_type = int(self.experiment_config['ENV_PARAMETERS']['env_type'])
        self.self_transitions = int(self.experiment_config['ENV_PARAMETERS']['self_transitions'])
        self.fixed_params_exp_list = [self.seed, self.gamma, self.nb_states, self.nb_actions,
                                      self.nb_next_state_transition]
        self.estimate_baseline=bool((util.strtobool(self.experiment_config['ENV_PARAMETERS']['estimate_baseline'])))
        self.initial_state = 0
        self.pi_rand = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions

        self.baseline_target_perf_ratios = ast.literal_eval(
            self.experiment_config['BASELINE']['baseline_target_perf_ratios'])
        self.nb_trajectories_list = ast.literal_eval(self.experiment_config['BASELINE']['nb_trajectories_list'])

        self.log = bool(util.strtobool(self.experiment_config['META']['log']))

    def _run_one_iteration(self):
        """
        Runs one iteration on the Random MDPs benchmark, so iterates through different baseline and data set parameters
        and then starts the computation for each algorithm.
        """
        path_config = configparser.ConfigParser()
        path_config.read(os.path.join(directory, 'paths.ini'))
        spibb_path = path_config['PATHS']['spibb_path']
        sys.path.append(spibb_path)
        import SPIBBmaster.garnets as garnets
        self.garnet = garnets.Garnets(self.nb_states, self.nb_actions, self.nb_next_state_transition,
                                env_type=self.env_type, self_transitions=self.self_transitions, nb_traps=5, gamma=self.gamma)
        for baseline_target_perf_ratio in self.baseline_target_perf_ratios:
            print(f'Process with seed {self.seed} starting with baseline_target_perf_ratio {baseline_target_perf_ratio}'
                  f' out of {self.baseline_target_perf_ratios}')

            softmax_target_perf_ratio = (baseline_target_perf_ratio + 1) / 2
            self.to_append_run_one_iteration = self.to_append_run + [softmax_target_perf_ratio,
                                                                     baseline_target_perf_ratio]
            self.pi_b, self._q_pi_b, self.pi_star_perf, self.pi_b_perf, self.pi_rand_perf = \
                self.garnet.generate_baseline_policy(self.gamma,
                                                     softmax_target_perf_ratio=softmax_target_perf_ratio,
                                                     baseline_target_perf_ratio=baseline_target_perf_ratio)
            self.R_state_state = self.garnet.compute_reward()
            self.P = self.garnet.transition_function
            # for state in range(len(self.P)):
            #     print(f"in state {state}, we have the following actions")
            #     for action in range(len(self.P[state])):
            #         print(self.P[state][action])
            # print("The final state = ", self.garnet.final_state, "which has the following")
            # for action in range(len(self.P[self.garnet.final_state])):
            #     print(self.P[self.garnet.final_state][action])
            # print(self.R_state_state)
            # print(self.R_state_state[:, self.garnet.final_state])
            self.traps = self.garnet.get_traps()
            self.easter_egg = None

            self.R_state_action = self.compute_r_state_action(self.P, self.R_state_state)
            self.to_append_run_one_iteration += [self.pi_b_perf, self.pi_rand_perf, self.pi_star_perf]


            for nb_trajectories in self.nb_trajectories_list:
                print(
                    f'Process with seed {self.seed} starting with nb_trajectories {nb_trajectories} out of '
                    f'{self.nb_trajectories_list}')
                # Generate trajectories, both stored as trajectories and (s,a,s',r) transition samples
                self.data, batch_traj = self.generate_batch(nb_trajectories, self.garnet, self.pi_b,
                                                            easter_egg=self.easter_egg)
                self.to_append = self.to_append_run_one_iteration + [nb_trajectories]

                # print("----------------------------------------------------------------")                
                self.structure = self.reduce_transition_matrix(self.P)
                self.estimator = PACIntervalEstimator(self.structure, 0.1, self.data, self.nb_actions, alpha=10)
                self.estimator.calculate_intervals()
                self.intervals = self.estimator.get_intervals()
                self.shielder = ShieldRandomMDP(self.structure, self.traps, [self.garnet.final_state], self.intervals)
                self.shielder.calculateShield()
                # self.shielder.printShield()
                # print("----------------------------------------------------------------")
                self._run_algorithms()


    def _set_traps(self, n, reward):
        # set n traps
        potential_final_states = [s for s in range(self.nb_states) if s != self.garnet.final_state and s != 0]
        
        #
        for _ in range(n):
            trap = np.random.choice(potential_final_states)
            self.traps.append(trap)
            self.R_state_state[:, trap] = reward
            self.P[trap, :, :] = 0
            self.R_state_action = self.compute_r_state_action(self.P, self.R_state_state)
            potential_final_states.remove(trap)
        
        pi_star = PiStar(pi_b=None, gamma=self.gamma, nb_states=self.nb_states, nb_actions=self.nb_actions,
                         data=[[]], R=self.R_state_state, episodic=self.episodic, P=self.P)
        pi_star.fit()
        self.pi_star_perf = self._policy_evaluation_exact(pi_star.pi)
        self.pi_b_perf = self._policy_evaluation_exact(self.pi_b)
        self.pi_rand_perf = self._policy_evaluation_exact(self.pi_rand)
        if self.log:
            print(f"Optimal perf in trapped environment:\t\t\t" + str(self.pi_star_perf))
            print(f"Baseline perf in trapped environment:\t\t\t" + str(self.pi_b_perf))
            
        
    def _set_easter_egg(self, reward):
        """
        Sets up the easter egg if one is used (also possible to use a bad easter egg with negative reward).
        :param reward: the reward of the easter egg
        """
        # Randomly pick a second terminal state and update model parameters
        potential_final_states = [s for s in range(self.nb_states) if s != self.garnet.final_state and s != 0]
        self.easter_egg = np.random.choice(potential_final_states)
        # Or pick the one with the least transitions
        # current_proba_sum = current_proba.reshape(-1, current_proba.shape[-1]).sum(axis=0)
        # mask_easter = np.ma.array(current_proba_sum, mask=False)
        # mask_easter.mask[garnet.final_state] = True
        # easter_egg = np.argmin(mask_easter)
        assert (self.garnet.final_state != self.easter_egg)
        self.R_state_state[:, self.easter_egg] = reward
        self.P[self.easter_egg, :, :] = 0
        self.R_state_action = self.compute_r_state_action(self.P, self.R_state_state)
        # Compute optimal policy in this new environment
        pi_star = PiStar(pi_b=None, gamma=self.gamma, nb_states=self.nb_states, nb_actions=self.nb_actions,
                         data=[[]], R=self.R_state_state, episodic=self.episodic, P=self.P)
        pi_star.fit()
        self.pi_star_perf = self._policy_evaluation_exact(pi_star.pi)
        self.pi_b_perf = self._policy_evaluation_exact(self.pi_b)
        self.pi_rand_perf = self._policy_evaluation_exact(self.pi_rand)
        if self.log:
            if reward > 0:
                property_easter_egg = 'good'
            else:
                property_easter_egg = 'bad'
            print(f"Optimal perf in {property_easter_egg} easter egg environment:\t\t\t" + str(self.pi_star_perf))
            print(f"Baseline perf in {property_easter_egg} easter egg environment:\t\t\t" + str(self.pi_b_perf))

            

            

    def generate_batch(self, nb_trajectories, env, pi, easter_egg=None, max_steps=50):
        """
        Generates a data batch for an episodic MDP.
        :param nb_steps: number of steps in the data batch
        :param env: environment to be used to generate the batch on
        :param pi: policy to be used to generate the data as numpy array with shape (nb_states, nb_actions)
        :return: data batch as a list of sublists of the form [state, action, next_state, reward]
        """
        print(self.traps)
        trajectories = []
        for _ in np.arange(nb_trajectories):
            nb_steps = 0
            trajectorY = []
            state = env.reset()
            is_done = False
            while nb_steps < max_steps and not is_done:
                action_choice = np.random.choice(pi.shape[1], p=pi[state])
                state, reward, next_state, is_done = env.step(action_choice, easter_egg)
                trajectorY.append([action_choice, state, next_state, reward])
                state = next_state
                nb_steps += 1
            trajectories.append(trajectorY)
        batch_traj = [val for sublist in trajectories for val in sublist]
        return trajectories, batch_traj
    
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
        num_states = self.nb_states
        num_actions = self.nb_actions
        max_transitions = self.nb_next_state_transition
        # Prepare the reduced matrix to hold the indices of possible states
        reduced_matrix = np.zeros((num_states, num_actions, max_transitions), dtype=int)
        
        # Loop through each state and action to populate the reduced matrix
        for state in range(num_states):
            for action in range(num_actions):
                # Get indices of nonzero probabilities (possible end states)
                possible_states = np.nonzero(transition_matrix[state, action])[0]
                reduced_matrix[state, action, :len(possible_states)] = possible_states
        
        return reduced_matrix
    
class PrismExperiment(Experiment):
    # Inherits from the base class Experiment to implement the Wet Chicken experiment specifically.
    fixed_params_exp_columns = ['seed', 'gamma', 'baseline_method','pi_rand_perf', 'pi_star_perf']
    
    def compute_r_state_action(self, P, R):
        result = defaultdict(float)

        for (i, j, k), p_val in P.items():
            r_val = R.get((i, k), 0.0)
            result[(i, j)] += p_val * r_val

        # Convert result to dense NumPy array

        dense_result = np.zeros((self.nb_states, self.nb_actions))

        for (i, j), val in result.items():
            dense_result[i, j] = val

        return dense_result
    
    def _set_env_params(self):
        """
        Reads in all parameters necessary from self.experiment_config to set up the Wet Chicken experiment.
        """
        self.episodic = True
        self.gamma = float(self.experiment_config['ENV_PARAMETERS']['GAMMA'])
        
        print("start env")
        self.env = prism_env()
        print("get values")
        self.nb_states = self.env.get_nb_states()
        self.nb_actions = self.env.get_nb_actions()
        self.traps = self.env.get_traps()
        self.goal = self.env.get_goal()
        self.initial_state = self.env.get_init_state()
        
        self.P = self.env.get_transition_function()
        self.R_state_state = self.env.get_reward_function()

        print("calcing r_sa")
        self.R_state_action = self.compute_r_state_action(self.P, self.R_state_state)
        self.baseline_method = self.experiment_config['BASELINE']['method']
        self.fixed_params_exp_list = [self.seed, self.gamma, self.baseline_method]

        pi_rand = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        print("calcing rand perf")
        pi_rand_perf = self._policy_evaluation_exact(pi_rand)
        print(f"pi_rand_perf = {pi_rand_perf}")
        
        self.fixed_params_exp_list.append(pi_rand_perf)

        pi_star = PiStar(pi_b=None, gamma=self.gamma, nb_states=self.nb_states, nb_actions=self.nb_actions,
                         data=[[]], R=self.R_state_state, episodic=self.episodic, P=self.P)
        pi_star.fit()
        pi_star_perf = self._policy_evaluation_exact(pi_star.pi)
        print(f"pi_star_perf = {pi_star_perf}")
        self.fixed_params_exp_list.append(pi_star_perf)

        self.epsilons_baseline = ast.literal_eval(self.experiment_config['BASELINE']['epsilons_baseline'])
        self.nb_trajectories_list = ast.literal_eval(self.experiment_config['BASELINE']['nb_trajectories_list'])
        if self.baseline_method == 'heuristic':
            self.variable_params_exp_columns = ['i', 'epsilon_baseline', 'pi_b_perf', 'length_trajectory']
        else:
            self.learning_rates = ast.literal_eval(self.experiment_config['BASELINE']['learning_rates'])
            self.variable_params_exp_columns = ['i', 'epsilon_baseline', 'learning_rate', 'pi_b_perf',
                                                'length_trajectory']
        self.estimate_baseline=bool((util.strtobool(self.experiment_config['ENV_PARAMETERS']['estimate_baseline'])))

        
    def _run_one_iteration(self):
        for epsilon_baseline in self.epsilons_baseline:
            print(f'Process with seed {self.seed} starting with epsilon_baseline {epsilon_baseline} out of'
                  f' {self.epsilons_baseline}')
            
            print("creating Baseline Policy")
            self.pi_b = self.env.get_baseline_policy(epsilon=epsilon_baseline)
            print("a")
            self.to_append_run_one_iteration = self.to_append_run + [epsilon_baseline,
                                                                        self._policy_evaluation_exact(self.pi_b)]
            print("b")
            
            for nb_trajectories in self.nb_trajectories_list:
                print(
                    f'Process with seed {self.seed} starting with nb_trajectories {nb_trajectories} out of '
                    f'{self.nb_trajectories_list}')
                # Generate trajectories, both stored as trajectories and (s,a,s',r) transition samples
                print("Generating Trajectories")
                self.data, batch_traj = self.generate_batch(nb_trajectories, self.env, self.pi_b)
                self.to_append = self.to_append_run_one_iteration + [nb_trajectories]

                # print("----------------------------------------------------------------")    
                print("Estimating Intervals")            
                self.structure = self.reduce_transition_matrix(self.P)   
                self.estimator = PACIntervalEstimator(self.structure, 0.1, self.data, self.nb_actions, alpha=2)
                self.estimator.calculate_intervals()
                self.intervals = self.estimator.get_intervals()
                # print(intervals)
                print("Calculating Shield")  
                self.shielder = ShieldPrism(self.structure, self.traps, self.goal, self.intervals)
                self.shielder.calculateShield()
                # self.shielder.printShield()
                # print("----------------------------------------------------------------")
                print("Running Algorithms")
                self._run_algorithms()
                

    def generate_batch(self, nb_trajectories, env, pi, max_steps=50):
        """
        Generates a data batch for an episodic MDP.
        :param nb_steps: number of steps in the data batch
        :param env: environment to be used to generate the batch on
        :param pi: policy to be used to generate the data as numpy array with shape (nb_states, nb_actions)
        :return: data batch as a list of sublists of the form [state, action, next_state, reward]
        """
        trajectories = []
        
        for _ in np.arange(nb_trajectories):
            nb_steps = 0
            trajectorY = []
            env.reset()
            state = self.initial_state
            is_done = False
            while nb_steps < max_steps and not is_done:
                action_choice = np.random.choice(pi.shape[1], p=pi[state])
                state, next_state, reward = env.step(action_choice)
                while state == -1:
                    action_choice = np.random.choice(pi.shape[1], p=pi[state])
                    state, next_state, reward = env.step(action_choice)
                    print("help!!", action_choice)
                is_done = env.is_done()                    
                trajectorY.append([action_choice, state, next_state, reward])
                state = next_state
                nb_steps += 1
            trajectories.append(trajectorY)
        batch_traj = [val for sublist in trajectories for val in sublist]
        # temppp = np.array(trajectories)
        # print(temppp)
        return trajectories, batch_traj
            
    

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

        # Prepare the reduced matrix to hold the indices of possible states
        reduced_matrix = np.empty((self.nb_states, self.nb_actions), dtype=object)

        for state in range(self.nb_states):
            for action in range(self.nb_actions):
                reduced_matrix[state, action] = []
        
        # Loop through each state and action to populate the reduced matrix
        for (state,action,next_state) in transition_matrix.keys():
            reduced_matrix[state][action].append(next_state)
            
        # for state in range(num_states):
        #     for action in range(num_actions):
        #         # Get indices of nonzero probabilities (possible end states)
        #         possible_states = np.nonzero(transition_matrix[state, action])[0]
        #         reduced_matrix[state, action] = np.array(possible_states)
        return reduced_matrix

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
      num_states: number of states
      num_actions: number of actions

    Returns:
      v: 1D array with updated state values
      q: 2D array with updated action values
    """
    num_states, num_actions = pi.shape
    # Compute expected rewards under policy
    r_pi = np.einsum('ij,ij->i', pi, r)
    print("a")
    # Build sparse p_pi as a dictionary {(s, s'): prob}
    p_pi_sparse = defaultdict(float)

    for (s, a, s_prime), prob in p_sparse.items():
        p_pi_sparse[(s, s_prime)] += pi[s, a] * prob
    print("b")
    # Convert sparse p_pi to a scipy sparse matrix for solving
    P_pi = dok_matrix((num_states, num_states), dtype=np.float64)
    for (s, s_prime), prob in p_pi_sparse.items():
        P_pi[s, s_prime] = prob
    print("c")
    # Solve (I - gamma * P_pi) * v = r_pi
    A = identity(num_states, format='csr') - gamma * P_pi.tocsr()
    v, info = bicgstab(A, r_pi, atol=1e-6)
    if info != 0:
        print("Warning: Iterative solver did not converge")
    print("d")
    # Compute Q-values
    q = np.zeros((num_states, num_actions))
    for (s, a, s_prime), prob in p_sparse.items():
        q[s, a] += prob * v[s_prime]
    q = r + gamma * q
    print("e")
    return v, q, dict(p_pi_sparse)

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
algorithm_name_dict = {SPIBB.NAME: SPIBB, Lower_SPIBB.NAME: Lower_SPIBB,
                       ApproxSoftSPIBB.NAME: ApproxSoftSPIBB, ExactSoftSPIBB.NAME: ExactSoftSPIBB,
                       AdvApproxSoftSPIBB.NAME: AdvApproxSoftSPIBB, LowerApproxSoftSPIBB.NAME: LowerApproxSoftSPIBB,
                       DUIPI.NAME: DUIPI, shield_DUIPI.NAME: shield_DUIPI, Basic_rl.NAME: Basic_rl, 
                       RMin.NAME: RMin, Shield_RMin.NAME: Shield_RMin, MBIE.NAME: MBIE, shield_MBIE.NAME : shield_MBIE, 
                       RaMDP.NAME: RaMDP, Shield_RaMDP.NAME : Shield_RaMDP, Shield_SPIBB.NAME: Shield_SPIBB, 
                       Shield_Lower_SPIBB.NAME: Shield_Lower_SPIBB,
                       WorstCaseRMDP.NAME : WorstCaseRMDP, Shield_WorstCaseRMDP.NAME : Shield_WorstCaseRMDP, PiStar.NAME: PiStar}




