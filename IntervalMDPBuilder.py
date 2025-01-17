import stormpy
import pycarl

class IntervalMDPBuilder:
    def __init__(self, transitions, intervals, goal, traps):
        self.transitions = transitions
        self.intervals = intervals
        self.traps = traps
        self.goal = goal
        self.groups_nr = 0
        self.num_states= len(self.transitions)
        self.num_actions = len(self.transitions[0])
        
    def build_wetChicken_model_with_init(self, state_init, action_init, next_states_init):
        builder = stormpy.IntervalSparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False, has_custom_row_grouping=True, row_groups=0)
        actions = [
            "drift",
            "hold",
            "paddle_back",
            "right",
            "left",
        ]
        # Create sparse matrix
        counter = 0
        # Build Initial state
        builder.new_row_group(counter)
        for next_state_init in next_states_init:
            bounds = self.intervals[(state_init, action_init, next_state_init)]
            builder.add_next_value(counter, next_state_init+1, pycarl.Interval(bounds[0], bounds[1])) 
        counter += 1 
        # Build 25x25 grid
        for state in range(self.num_states-1):
            builder.new_row_group(counter)
            for action in range(self.num_actions):
                next_states = self.transitions[state][action]
                for next_state in next_states:
                    bounds = self.intervals[(state, action, next_state)]
                    # Check if you have fallen of the waterfall
                    # if next_state == 25:
                    #     x = int(state / 5)
                    #     y = state % 5
                    #     print(f"In state ({x},{y}), using action {actions[action]} has a chance of falling off the waterfall between {bounds[0]} and {bounds[1]}")
                    #     builder.add_next_value(counter, self.num_states+1, pycarl.Interval(bounds[0], bounds[1]))
                    # else:
                    builder.add_next_value(counter, next_state+1, pycarl.Interval(bounds[0], bounds[1]))
                    
                    # print(next_state)
                counter+=1
        
                
        # Build the waterfall
        builder.new_row_group(counter) 
        builder.add_next_value(counter, 1, pycarl.Interval(1, 1))
       
        

        transition_matrix = builder.build()
        state_labels = self.set_state_labels_with_init_WetChicken()
        choice_labels = self.set_choice_labels_with_init_WetChicken(action_init)
        # choice_labels = self.set_choice_labels_MDP()
        reward_model = self.set_reward_model_MDP()
        
        # Collect components
        components = stormpy.SparseIntervalModelComponents(
            transition_matrix=transition_matrix, state_labeling=state_labels, reward_models=reward_model, rate_transitions=False
        )
        components.choice_labeling = choice_labels
        
        # Build the model
        mdp = stormpy.storage.SparseIntervalMdp(components)
        
        return mdp
    
    def build_MDP_model_with_init(self, state_init, action_init, next_states_init):
        # initialize builder
        builder = stormpy.IntervalSparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False, has_custom_row_grouping=True, row_groups=0)
        
        # Create sparse matrix
        counter = 0
        # Add initial state corresponding to the given state-action pair
        builder.new_row_group(counter)
        if(sum(next_states_init) > 0):
            for next_state_init in next_states_init:
                bounds = self.intervals[(state_init, action_init, next_state_init)]
                builder.add_next_value(counter, next_state_init+1, pycarl.Interval(bounds[0], bounds[1]))
        else:
            builder.add_next_value(counter, 0, pycarl.Interval(1, 1)) 
        counter += 1 
        for state in range(self.num_states):
            builder.new_row_group(counter)
            for action in range(self.num_actions):
                next_states = self.transitions[state][action]
                # check if next state is a terminal state. If so, only add a transition to itself
                if(sum(next_states) > 0):
                    for next_state in next_states:
                        bounds = self.intervals[(state, action, next_state)]
                        builder.add_next_value(counter, next_state+1, pycarl.Interval(bounds[0], bounds[1]))
                else:
                      builder.add_next_value(counter, state+1, pycarl.Interval(1, 1))  
                counter+=1
                

        transition_matrix = builder.build()
        state_labels = self.set_state_labels_with_init_MDP()
        choice_labels = self.set_choice_labels_with_init_MDP(action_init)
        # choice_labels = self.set_choice_labels_MDP()
        reward_model = self.set_reward_model_MDP()
        
        # Collect components
        components = stormpy.SparseIntervalModelComponents(
            transition_matrix=transition_matrix, state_labeling=state_labels, reward_models=reward_model, rate_transitions=False
        )
        components.choice_labeling = choice_labels
        
        # Build the model
        mdp = stormpy.storage.SparseIntervalMdp(components)
        return mdp
    
    def build_airplane_model_with_init(self, state_init, action_init, next_states_init):
        builder = stormpy.IntervalSparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False, has_custom_row_grouping=True, row_groups=0)
        
        counter = 0
        
        builder.new_row_group(counter)
        if(sum(next_states_init) > 0):
            for next_state_init in next_states_init:
                bounds = self.intervals[(state_init, action_init, next_state_init)]
                builder.add_next_value(counter, next_state_init+1, pycarl.Interval(bounds[0], bounds[1]))
            
        else:
            builder.add_next_value(counter, 0, pycarl.Interval(1, 1)) 
            
        # Add the rest of the model
        counter += 1 
        for state in range(self.num_states):
            builder.new_row_group(counter)
            for action in range(self.num_actions):
                next_states = self.transitions[state][action]
                # check if next state is a terminal state. If so, only add a transition to itself
                if(sum(next_states) > 0):
                    for next_state in next_states:
                        bounds = self.intervals[(state, action, next_state)]
                        builder.add_next_value(counter, next_state+1, pycarl.Interval(bounds[0], bounds[1]))
                else:
                      builder.add_next_value(counter, state+1, pycarl.Interval(1, 1))  
                counter+=1
                
        transition_matrix = builder.build()
        state_labels = self.set_state_labels_with_init_airplane()
        choice_labels = self.set_choice_labels_with_init_airplane(action_init)
        # choice_labels = self.set_choice_labels_MDP()
        reward_model = self.set_reward_model_MDP()
        
        # Collect components
        components = stormpy.SparseIntervalModelComponents(
            transition_matrix=transition_matrix, state_labeling=state_labels, reward_models=reward_model, rate_transitions=False
        )
        components.choice_labeling = choice_labels
        
        # Build the model
        mdp = stormpy.storage.SparseIntervalMdp(components)
        return mdp

    def get_nr_groups(self):
        return self.groups_nr
    
    def set_state_labels_with_init_MDP(self):
        state_labeling = stormpy.storage.StateLabeling(self.num_states+1)
        labels = {"goal", "trap", "init"}
        for label in labels:
            state_labeling.add_label(label)
        for i in self.traps:
            state_labeling.add_label_to_state("trap", i+1)
        for i in self.goal:
            state_labeling.add_label_to_state("goal", i+1)
        state_labeling.add_label_to_state("init", 0)
        
        return state_labeling
    
    def set_state_labels_with_init_airplane(self):
        state_labeling = stormpy.storage.StateLabeling(self.num_states+1)
        labels = {"success", "crash", "init"}
        for label in labels:
            state_labeling.add_label(label)
        for i in self.traps:
            state_labeling.add_label_to_state("crash", i+1)
        for i in self.goal:
            state_labeling.add_label_to_state("success", i+1)
        state_labeling.add_label_to_state("init", 0)
        
        return state_labeling
    
    
    def find_closest_states(self, states_list, length):
        """
        Given a list of states in single integer representation, identifies the states where the boat
        is on the edge off the waterfall (i.e., where x = 4).

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
    
    def set_state_labels_with_init_WetChicken(self):
        state_labeling = stormpy.storage.StateLabeling(self.num_states+1)
        labels = {"waterfall", "init", "goal"}
        for label in labels:
            state_labeling.add_label(label)
        edge_states = self.find_closest_states(range(self.num_states-1), 5)
        for i in edge_states:
            state_labeling.add_label_to_state("goal", i+1)
        state_labeling.add_label_to_state("waterfall", self.num_states)
        
        state_labeling.add_label_to_state("init", 0)
        
        return state_labeling
    
    def set_choice_labels_with_init_MDP(self, action):    
        choice_labeling = stormpy.storage.ChoiceLabeling(self.num_actions*self.num_states+1)
        choice_labels = {"action 1", "action 2", "action 3", "action 4"}
        for label in choice_labels:
            choice_labeling.add_label(label)
            
        for i in range(self.num_states):
            choice_labeling.add_label_to_choice("action 1", 1+i*4)
            choice_labeling.add_label_to_choice("action 2", 1+i*4+1)
            choice_labeling.add_label_to_choice("action 3", 1+i*4+2)
            choice_labeling.add_label_to_choice("action 4", 1+i*4+3) 
        
        init_action = ""
        match action:
            case 0:
                init_action = "action 1"
            case 1:
                init_action = "action 2"
            case 2:
                init_action = "action 3"
            case 3: 
                init_action = "action 4"
        choice_labeling.add_label_to_choice(init_action, 0)
        return choice_labeling
    
    def set_choice_labels_with_init_WetChicken(self, action):    
        choice_labeling = stormpy.storage.ChoiceLabeling(self.num_actions*(self.num_states-1)+2)
        choice_labels = {"drift", "hold", "paddle_back", "right", "left", "reset"}
        for label in choice_labels:
            choice_labeling.add_label(label)
            
        for i in range(self.num_states-1):
            choice_labeling.add_label_to_choice("drift", 1+i*5)
            choice_labeling.add_label_to_choice("hold", 1+i*5+1)
            choice_labeling.add_label_to_choice("paddle_back", 1+i*5+2)
            choice_labeling.add_label_to_choice("right", 1+i*5+3) 
            choice_labeling.add_label_to_choice("left", 1+i*5+4) 
        
        choice_labeling.add_label_to_choice("reset", self.num_actions*(self.num_states-1)+1)
        init_action = ""
        match action:
            case 0:
                init_action = "drift"
            case 1:
                init_action = "hold"
            case 2:
                init_action = "paddle_back"
            case 3: 
                init_action = "right"
            case 4: 
                init_action = "left"
        choice_labeling.add_label_to_choice(init_action, 0)
        return choice_labeling
    
    def set_choice_labels_with_init_airplane(self, action):    
        choice_labeling = stormpy.storage.ChoiceLabeling(self.num_actions*self.num_states+1)
        choice_labels = {"down", "up", "stay"}
        for label in choice_labels:
            choice_labeling.add_label(label)
            
        for i in range(self.num_states):
            choice_labeling.add_label_to_choice("down", 1+i*3)
            choice_labeling.add_label_to_choice("up", 1+i*3+1)
            choice_labeling.add_label_to_choice("stay", 1+i*3+2)
        
        init_action = ""
        match action:
            case 0:
                init_action = "down"
            case 1:
                init_action = "up"
            case 2:
                init_action = "stay"
        choice_labeling.add_label_to_choice(init_action, 0)
        return choice_labeling
            
    def set_reward_model_MDP(self):
        reward_models = {}
        reward_models["random MDP"] = stormpy.SparseIntervalRewardModel()
        return reward_models
    
    def buildMDP(self):
        # initialize builder
        builder = stormpy.IntervalSparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False, has_custom_row_grouping=True, row_groups=0)
        
        # Create sparse matrix
        counter = 0
        for state in range(self.num_states):
            builder.new_row_group(counter)
            for action in range(self.num_actions):
                next_states = self.transitions[state][action]
                # check if next state is a terminal state. If so, only add a transition to itself
                if(sum(next_states) > 0):
                    for next_state in next_states:
                        bounds = self.intervals[(state, action, next_state)]
                        builder.add_next_value(counter, next_state, pycarl.Interval(bounds[0], bounds[1]))
                else:
                      builder.add_next_value(counter, state, pycarl.Interval(1, 1))  
                counter+=1
        transition_matrix = builder.build()
        # Set State Labels
        state_labeling = stormpy.storage.StateLabeling(self.num_states)
        labels = {"goal", "trap"}
        for label in labels:
            state_labeling.add_label(label)
        for i in self.traps:
            state_labeling.add_label_to_state("trap", i)
        for i in self.goal:
            state_labeling.add_label_to_state("goal", i)
        
        # set choice labels, hard coded for 4 actions
        choice_labeling = stormpy.storage.ChoiceLabeling(self.num_actions*self.num_states)
        choice_labels = {"action 1", "action 2", "action 3", "action 4"}
        for label in choice_labels:
            choice_labeling.add_label(label)
            
        for i in range(self.num_states):
            choice_labeling.add_label_to_choice("action 1", i*4)
            choice_labeling.add_label_to_choice("action 2", i*4+1)
            choice_labeling.add_label_to_choice("action 3", i*4+2)
            choice_labeling.add_label_to_choice("action 4", i*4+3) 
        
        # set reward model   
        reward_models = {}
        reward_models["coin_flips"] = stormpy.SparseIntervalRewardModel()
        
        # Collect components
        components = stormpy.SparseIntervalModelComponents(
            transition_matrix=transition_matrix, state_labeling=state_labeling, reward_models=reward_models, rate_transitions=False
        )
        components.choice_labeling = choice_labeling
        
        # Build the model
        mdp = stormpy.storage.SparseIntervalMdp(components)
        return mdp
    