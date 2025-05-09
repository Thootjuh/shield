from collections import defaultdict
import stormpy
import stormpy.examples
import stormpy.examples.files
import numpy as np

class prism_env:
    def __init__(self):
        # self.file = prism_file
        # self.program = stormpy.examples.files.prism_mdp_maze
        self.program = "model.nm"
        self.prop = 'Pmax=? [F "goal"]'
        self.initial_calculations()
        self._state = self.initial_state
        
    def reset(self):
        self._state = self.initial_state
            
    def is_done(self):
        if self._state in self.goal_states or self._state in self.trap_states:
            return True
        return False
    
    def step(self, action_choice):
        old_state = self._state
        candidates = [(next_state, prob)
                for (s, a, next_state), prob in self.transition_function.items()
                if s == self._state and a == action_choice]
        if len(candidates) == 0:
            return -1, 0, 0
        next_states, probs = zip(*candidates)
        probs = np.array(probs, dtype=float)
        
        next_state = np.random.choice(next_states, p=probs)
        self._state = next_state
        return old_state, next_state, self.reward_function.get((old_state, next_state), 0)
    
    def get_nb_states(self):
        return self.nb_states
    def get_nb_actions(self):
        return self.nb_actions
    def get_traps(self):
        return self.trap_states
    def get_goal(self):
        return self.goal_states
    def get_init_state(self):
        return self.initial_state
    
    def initial_calculations(self):
        # set up the model
        print("running the model")
        program = stormpy.parse_prism_program(self.program)
        properties = stormpy.parse_properties_for_prism_program(self.prop, program)
        model = stormpy.build_model(program)
        result = stormpy.model_checking(model, properties[0], extract_scheduler=True)
        self.goal_states = []
        self.trap_states = []
        
        transition_function = defaultdict(float)
        # reward_model = model.reward_models.get(model.reward_models.keys()[0])
        reward_model = next(iter(model.reward_models.values()))

        nb_states = 0
        max_actions = 0
        scheduler = result.scheduler
        print("generating the transition function")
        # Get the transition model

        for state in model.states:
            choice = scheduler.get_choice(state.id)
            labels = state.labels
            nb_states += 1
            if state.id in model.initial_states:
                self.initial_state = state.id
            if "goal" in labels:
                self.goal_states.append(state.id)
            elif "crash" in labels: 
                self.trap_states.append(state.id)
            else:
                action_counter = 0
                for action in state.actions:
                    action_counter+=1
                    for transition in action.transitions:
                        transition_function[(state.id, action.id, transition.column)] = transition.value()
                max_actions = max(max_actions, action_counter)
        self.transition_function = transition_function
        counter = 0
        
        self.nb_states = nb_states
        self.nb_actions = max_actions 
        
        # get the reward model and baseline policy
        print("generating the reward model and pi_b")
        pi = np.full((self.nb_states, self.nb_actions), 0)
        reward_dict = {}
        for next_state in range(self.nb_states):
            reward = reward_model.get_state_reward(next_state)
            if reward != 0:
                for state in range(self.nb_states):
                    reward_dict[(state, next_state)] = reward
            choice = scheduler.get_choice(next_state).get_deterministic_choice()
            pi[next_state][choice] = 1
                         

        self.pi = pi     
        self.reward_model = reward_dict
        self.reward_function = reward_dict
        print(self.goal_states)
        print(self.trap_states)
        # generate a baseline policy
        print("finished calculations")
        
        
    def get_transition_function(self):
        return self.transition_function
    
    def get_reward_function(self):
        return self.reward_model
    
    def get_baseline_policy(self, epsilon):
        pi_r = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        pi_b = (1 - epsilon) * self.pi + epsilon * pi_r
        return pi_b