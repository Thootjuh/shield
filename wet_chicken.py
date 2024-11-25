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

        self.nb_states = self.length * self.width
        self.nb_actions = 5

        self.env = WetChicken(length=self.length, width=self.width, max_turbulence=self.max_turbulence,
                              max_velocity=self.max_velocity)
        self.initial_state = self.env.get_state_int()
        self.P = self.env.get_transition_function()
        self.R_state_state = self.env.get_reward_function()
        self.R_state_action = compute_r_state_action(self.P, self.R_state_state)

        self.baseline_method = self.experiment_config['BASELINE']['method']
        self.fixed_params_exp_list = [self.seed, self.gamma, self.length, self.width, self.max_turbulence,
                                      self.max_velocity, self.baseline_method]

        pi_rand = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        pi_rand_perf = self._policy_evaluation_exact(pi_rand)
        self.fixed_params_exp_list.append(pi_rand_perf)

        pi_star = PiStar(pi_b=None, gamma=self.gamma, nb_states=self.nb_states, nb_actions=self.nb_actions,
                         data=[[]], R=self.R_state_state, episodic=self.episodic, P=self.P)
        pi_star.fit()
        pi_star_perf = self._policy_evaluation_exact(pi_star.pi)
        self.fixed_params_exp_list.append(pi_star_perf)

        self.epsilons_baseline = ast.literal_eval(self.experiment_config['BASELINE']['epsilons_baseline'])
        self.lengths_trajectory = ast.literal_eval(self.experiment_config['BASELINE']['lengths_trajectory'])
        if self.baseline_method == 'heuristic':
            self.variable_params_exp_columns = ['i', 'epsilon_baseline', 'pi_b_perf', 'length_trajectory']
        else:
            self.learning_rates = ast.literal_eval(self.experiment_config['BASELINE']['learning_rates'])
            self.variable_params_exp_columns = ['i', 'epsilon_baseline', 'learning_rate', 'pi_b_perf',
                                                'length_trajectory']

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
        return trajectory