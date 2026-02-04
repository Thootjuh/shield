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
from environments.gym_environment import gymTaxi, gymIce
from environments.gym_cartpole_env import cartPole, cartPolePolicy
from environments.gym_maze.gym_maze_env import maze, mazePolicy
from environments.gym_crashing_mountain_car import crashingMountainCar, crashingMountainCarPolicy


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

from shield import ShieldRandomMDP, ShieldCartpole, ShieldCrashingMountainCar, ShieldMaze
from PACIntervalEstimator import PACIntervalEstimator
from evaluate_cartpole import evaluate_policy
from discretization.grid.define_imdp import imdp_builder

from discretization.MRL.helper_functions import trajToDF, state2region
from discretization.MRL.model import MDP_model
from discretization.MRL.testing import predict_cluster
from discretization.MRL_scratch.mrl_scratch import MRL_scratch
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
    algorithms_columns = ['method', 'hyperparam', 'method_perf', 'run_time', 'nb_states', 'success_rate', 'failure_rate']

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
        # self._run_shielded_baseline()
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
            

    def _run_shielded_baseline(self):
        pi_b_s = shieldedBaseline(pi_b=self.pi_b, gamma=self.gamma, nb_states=self.nb_states,
                                            nb_actions=self.nb_actions, data=self.data, R=self.R_state_state,
                                            episodic=self.episodic, shield=self.shielder, speed_up_dict=self.speed_up_dict, estimate_baseline=self.estimate_baseline)
        t_0 = time.time()
        pi_b_s.fit()
        t_1 = time.time()
        if self.discretization_method=='mrl':
            basic_rl_perf = evaluate_policy(self.env, pi_b_s.pi, 1, 500, self.discretization_method, predictor=self.predictor, dimensions=self.dimensions)
        elif self.discretization_method=='grid':
            basic_rl_perf = evaluate_policy(self.env, pi_b_s.pi, 1, 500, self.discretization_method)
        
        # basic_rl_perf = self._policy_evaluation_exact(pi_b_s.pi)
        method = pi_b_s.NAME + "_" + self.discretization_method
        method_perf = basic_rl_perf
        hyperparam = None
        run_time = t_1 - t_0
        self.results.append(self.to_append + [method, hyperparam, method_perf, run_time, self.nb_states])
    
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
                spibb_perf, succ_rate, failure_rate = evaluate_policy(self.env, spibb.pi, 1, 100, self.discretization_method, predictor=self.predictor, dimensions=self.dimensions)
            elif self.discretization_method=='grid':
                spibb_perf, succ_rate, failure_rate = evaluate_policy(self.env, spibb.pi, 1, 100, self.discretization_method)
            
            method = spibb.NAME + "_" + self.discretization_method
            method_perf = spibb_perf
            hyperparam = N_wedge
            run_time = t_1 - t_0
            write_policy_to_file(spibb.pi, "policy_shielded.txt")
            self.results.append(self.to_append + [method, hyperparam, method_perf, run_time, self.nb_states, succ_rate, failure_rate])
                  
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
                spibb_perf, succ_rate, failure_rate = evaluate_policy(self.env, spibb.pi, 1, 100, self.discretization_method, predictor=self.predictor, dimensions=self.dimensions)
            elif self.discretization_method=='grid':
                spibb_perf, succ_rate, failure_rate = evaluate_policy(self.env, spibb.pi, 1, 100, self.discretization_method)
            print("evaluated policy")
            # spibb_perf = self._policy_evaluation_exact(spibb.pi)
            method = spibb.NAME + "_" + self.discretization_method
            method_perf = spibb_perf
            hyperparam = N_wedge
            run_time = t_1 - t_0
            write_policy_to_file(spibb.pi, "policy.txt")
            self.results.append(self.to_append + [method, hyperparam, method_perf, run_time, self.nb_states, succ_rate, failure_rate])
    
    def _run_spibb_dqn(self, key):
        for N_wedge in self.algorithms_dict[key]['hyperparam']:
            spibb = spibb_dqn(baseline=self.pi_b, gamma=self.gamma, dataset=self.data_cont, env=self.env, minimum_count=N_wedge)
            t_0 = time.time()
            spibb.learn(passes_on_dataset = 25)
            t_1 = time.time()
            spibb_perf, succ_rate, failure_rate = evaluate_policy(self.env, None, 1, 100, self.discretization_method, ai=spibb.ai)
            method = 'spibb_dqn'
            method_perf = spibb_perf
            hyperparam = N_wedge
            run_time = t_1 - t_0
            self.results.append(self.to_append + [method, hyperparam, method_perf, run_time, self.nb_states, succ_rate, failure_rate])
            
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

                # Generate the shield: First, we compute the PAC estimate IMDP, then we compute the shield
                self.structure = self.reduce_transition_matrix(self.P)
                self.estimator = PACIntervalEstimator(self.structure, 0.1, self.data, self.nb_actions, alpha=5)
                self.estimator.calculate_intervals()
                self.intervals = self.estimator.get_intervals()
                self.shielder = ShieldRandomMDP(self.structure, self.traps, [self.garnet.final_state], self.intervals)
                self.shielder.calculateShield()
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
    


class GymCartPoleExperiment(Experiment):
    fixed_params_exp_columns = ['seed', 'gamma']
    def _set_env_params(self):
        """
        Reads in all parameters necessary from self.experiment_config to set up the Wet Chicken experiment.
        """
        self.episodic = True
        self.gamma = float(self.experiment_config['ENV_PARAMETERS']['GAMMA'])
        
        print("start env")
        self.env = cartPole()
        print("get values")
        self.nb_states = self.env.get_nb_states()
        self.nb_actions = self.env.get_nb_actions()
        

        self.traps = self.env.get_traps()
        self.goal = self.env.get_goal_state()

        self.initial_state = self.env.get_init_state()
        
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
                    plot=True,
                    verbose=False
                )
                print("Trained the model!!")
                
                # discretize data
                self.predictor = predict_cluster(m.df_trained, self.dimensions)
                d_data = self.discretize_data(self.data_cont, self.predictor)
                
                nb_states = m.df_trained["CLUSTER"].nunique()
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
                self.shielder = ShieldCartpole(self.structure, traps, goal, self.intervals, self.initial_state)
                self.shielder.calculateShield()
                # self.shielder.printShield()
                
                # # Run the algoirhtm
                self.nb_states = nb_states
                self.data = d_data
                self.pi_b = cartPolePolicy(self.env, epsilon=epsilon_baseline).compute_baseline_size(nb_states)

                print("Running Algorithms")
                self._run_algorithms()
                
                
                # ----------------------------- GRID ---------------------------------
                self.discretization_method = 'grid'
                self.pi_b = cartPolePolicy(self.env, epsilon=epsilon_baseline).pi
                print("The baseline has length:", len(self.pi_b))
                self.nb_states = self.env.get_nb_states()
                self.data = data_grid
                self.R_state_state = self.env.get_reward_function()
                
                print("Estimating Intervals")            
                self._count(self.data)
                self._build_model()
                self.structure = self._tm_to_next_states()
                self.estimator = PACIntervalEstimator(self.structure, 0.1, d_data, self.nb_actions, alpha=5)
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
                # ----------------------------- SPIBB-DQN ----------------------------------
                self.discretization_method = 'SPIBB-DQN'
                self.data = self.data_cont
                self._run_spibb_dqn('SPIBB-DQN')
         
        
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
                s_d = state2region(predictor, s, 4)
                ns_d = state2region(predictor, ns, 4)
                traj.append([a, s_d, ns_d, r])
            data_disc.append(traj)
        return data_disc
                             
    def get_empty_structure(self, nb_states):
        empty_structure = np.empty((nb_states, self.nb_actions), dtype=object)                      
        
        for state in range(nb_states):
            for action in range(self.nb_actions):
                empty_structure[state, action] = np.array([])
                
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
                    poss_next = np.append(poss_next, [ns])
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
    

class GymMazeExperiment(Experiment):
    fixed_params_exp_columns = ['seed', 'gamma']
    def _set_env_params(self):
        """
        Reads in all parameters necessary from self.experiment_config to set up the Wet Chicken experiment.
        """
        self.episodic = True
        self.gamma = float(self.experiment_config['ENV_PARAMETERS']['GAMMA'])
        
        print("start env")
        self.env = maze()
        print("get values")
        self.nb_states = self.env.get_nb_states()
        self.nb_actions = self.env.get_nb_actions()
        

        self.traps = self.env.get_traps()
        self.goal = self.env.get_goal_state()

        self.initial_state = self.env.get_init_state()
        
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
                m = MDP_model()
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
                    verbose=False
                )
                print("Trained the model!!")
                
                # discretize data
                self.predictor = predict_cluster(m.df_trained, self.dimensions)
                d_data = self.discretize_data(self.data_cont, self.predictor)
                
                nb_states = m.df_trained["CLUSTER"].nunique()
                print("nb states = ", nb_states)                
                # get discrete reward function
                self.R_state_state = np.zeros((nb_states, nb_states))
                traps = []
                goal = []
                for state in range(len(self.R_state_state)):
                    r = m.R_df[state]
                    self.R_state_state[:, state] = r
                    if r == -1.0:
                        traps.append(state)
                    if r == 1.0:
                        goal.append(state)

                # get structure transition function
                self.structure = self.get_empty_structure(nb_states)
                self.structure = self.add_trans_from_data(self.structure, d_data)
                
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
                self.nb_states = nb_states
                self.data = d_data
                self.pi_b = mazePolicy(self.env, epsilon=epsilon_baseline).compute_baseline_size(nb_states)

                print("Running Algorithms")
                self._run_algorithms()
                
                # ----------------------------- GRID ---------------------------------
                # self.discretization_method = 'grid'
                # self.pi_b = cartPolePolicy(self.env, epsilon=epsilon_baseline).pi
                # self.nb_states = self.env.get_nb_states()
                # self.data = data_grid
                # self.R_state_state = self.env.get_reward_function()
                
                # print("Estimating Intervals")            
                # self._count()
                # self.estimator = imdp_builder(self.data, self.count_state_action_state, self.count_state_action, self.episodic, beta=1e-4, kstep=1)
                # self.intervals = self.estimator.get_intervals()
                
                
                # print("Calculating Shield")  
                # self.structure = self.build_transition_matrix()
                # self.shielder = ShieldCartpole(self.structure, [self.traps], self.goal, self.intervals, self.initial_state)
                # self.shielder.calculateShield()
                # # self.shielder.printShield()
                # print("Running Algorithms")
                # self._run_algorithms()
                # ----------------------------- SPIBB-DQN ----------------------------------
                # self._run_spibb_dqn('SPIBB-DQN')
                
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
            is_done = False
            # print(state)
            while nb_steps < max_steps and not is_done:
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
                trajectorY_cont.append([state, action_choice, next_state, reward, terminated, truncated])
                region = next_region
                nb_steps += 1
            trajectories_cont.append(trajectorY_cont)
            trajectories.append(trajectorY)
            env.reset()
            env.set_random_state()
        batch_traj = [val for sublist in trajectories for val in sublist]
        print("number of successes = ", succ_count)
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
                empty_structure[state, action] = np.array([])
                
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
                    poss_next = np.append(poss_next, [ns])
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
                s_d = state2region(predictor, s, 2)
                ns_d = state2region(predictor, ns, 2)
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
                    poss_next = np.append(poss_next, [ns])
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
      num_states: number of states
      num_actions: number of actions

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
    if info != 0:
        print("Warning: Iterative solver did not converge")
    # Compute Q-values
    q = np.zeros((num_states, num_actions))
    for (s, a, s_prime), prob in p_sparse.items():
        q[s, a] += prob * v[s_prime]
    q = r + gamma * q
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
algorithm_name_dict = {SPIBB.NAME: SPIBB, Lower_SPIBB.NAME: Lower_SPIBB, 'SPIBB-DQN': spibb_dqn,
                       ApproxSoftSPIBB.NAME: ApproxSoftSPIBB, ExactSoftSPIBB.NAME: ExactSoftSPIBB,
                       AdvApproxSoftSPIBB.NAME: AdvApproxSoftSPIBB, LowerApproxSoftSPIBB.NAME: LowerApproxSoftSPIBB,
                       DUIPI.NAME: DUIPI, shield_DUIPI.NAME: shield_DUIPI, Basic_rl.NAME: Basic_rl, 
                       RMin.NAME: RMin, Shield_RMin.NAME: Shield_RMin, MBIE.NAME: MBIE, shield_MBIE.NAME : shield_MBIE, 
                       RaMDP.NAME: RaMDP, Shield_RaMDP.NAME : Shield_RaMDP, Shield_SPIBB.NAME: Shield_SPIBB, 
                       Shield_Lower_SPIBB.NAME: Shield_Lower_SPIBB,
                       WorstCaseRMDP.NAME : WorstCaseRMDP, Shield_WorstCaseRMDP.NAME : Shield_WorstCaseRMDP, PiStar.NAME: PiStar}




