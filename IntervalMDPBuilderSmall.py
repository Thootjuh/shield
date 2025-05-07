import stormpy
import pycarl

class IntervalMDPBuilderSmall:
    def __init__(self, transitions, intervals, goal, traps):
        self.transitions = transitions
        self.intervals = intervals
        self.traps = traps
        self.goal = goal
        self.groups_nr = 0
        self.num_states= len(self.transitions)
        self.num_actions = len(self.transitions[0])
    
    def build_model(self):
        pass
    
    def set_state_labels(self):
        pass
    
    def set_choice_labels(self):
        pass
    
    def set_reward_model_MDP(self):
        reward_models = {}
        reward_models["random MDP"] = stormpy.SparseIntervalRewardModel()
        return reward_models
    
class IntervalMDPBuilderRandomMDPSmall(IntervalMDPBuilderSmall):
    def build_model(self):
        # initialize builder
        builder = stormpy.IntervalSparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False, has_custom_row_grouping=True, row_groups=0)
        
        # Create sparse matrix
        counter = 0
        # Add initial state corresponding to the given state-action pair
        for state in range(self.num_states):
            builder.new_row_group(counter)
            for action in range(self.num_actions):
                next_states = self.transitions[state][action]
                # check if next state is a terminal state. If so, only add a transition to itself
                if(len(next_states) > 0):
                    for next_state in next_states:
                        bounds = self.intervals[(state, action, next_state)]
                        builder.add_next_value(counter, next_state, pycarl.Interval(bounds[0], bounds[1]))
                else:
                      builder.add_next_value(counter, state, pycarl.Interval(1, 1))  
                counter+=1
                

        transition_matrix = builder.build()
        state_labels = self.set_state_labels()
        choice_labels = self.set_choice_labels()
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
    
    def set_state_labels(self):
        state_labeling = stormpy.storage.StateLabeling(self.num_states)
        labels = {"goal", "trap", "init"}
        for label in labels:
            state_labeling.add_label(label)
        for i in self.traps:
            state_labeling.add_label_to_state("trap", i)
        for i in self.goal:
            state_labeling.add_label_to_state("goal", i)
        state_labeling.add_label_to_state("init", 0)
        
        return state_labeling
    
    def set_choice_labels(self):
        choice_labeling = stormpy.storage.ChoiceLabeling(self.num_actions*self.num_states)
        choice_labels = {"action 1", "action 2", "action 3", "action 4"}
        for label in choice_labels:
            choice_labeling.add_label(label)
            
        for i in range(self.num_states):
            choice_labeling.add_label_to_choice("action 1", i*4)
            choice_labeling.add_label_to_choice("action 2", i*4+1)
            choice_labeling.add_label_to_choice("action 3", i*4+2)
            choice_labeling.add_label_to_choice("action 4", i*4+3) 
        
        return choice_labeling
    
class IntervalMDPBuilderWetChickenSmall(IntervalMDPBuilderSmall):
    def build_model(self):
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
                    builder.add_next_value(counter, next_state, pycarl.Interval(bounds[0], bounds[1]))
                    
                    # print(next_state)
                counter+=1
        
                
        # Build the waterfall
        builder.new_row_group(counter) 
        builder.add_next_value(counter, 1, pycarl.Interval(1, 1))

        transition_matrix = builder.build()
        state_labels = self.set_state_labels()
        choice_labels = self.set_choice_labels()
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
    
    def set_state_labels(self):
        state_labeling = stormpy.storage.StateLabeling(self.num_states)
        labels = {"waterfall", "init", "goal"}
        for label in labels:
            state_labeling.add_label(label)
        edge_states = self.find_closest_states(range(self.num_states-1), 5)
        for i in edge_states:
            state_labeling.add_label_to_state("goal", i)
        state_labeling.add_label_to_state("waterfall", self.num_states-1)
        
        state_labeling.add_label_to_state("init", 0)
        
        return state_labeling
    
    def set_choice_labels(self):
        choice_labeling = stormpy.storage.ChoiceLabeling(self.num_actions*(self.num_states-1)+1)
        choice_labels = {"drift", "hold", "paddle_back", "right", "left", "reset"}
        for label in choice_labels:
            choice_labeling.add_label(label)
            
        for i in range(self.num_states-1):
            choice_labeling.add_label_to_choice("drift", i*5)
            choice_labeling.add_label_to_choice("hold", i*5+1)
            choice_labeling.add_label_to_choice("paddle_back", i*5+2)
            choice_labeling.add_label_to_choice("right", i*5+3) 
            choice_labeling.add_label_to_choice("left", i*5+4) 
        
        choice_labeling.add_label_to_choice("reset", self.num_actions*(self.num_states-1))

        return choice_labeling
    
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
    
class IntervalMDPBuilderPacmanSmall(IntervalMDPBuilderSmall):
    def build_model(self):
        # initialize builder
        # print(next_states_init)
        builder = stormpy.IntervalSparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False, has_custom_row_grouping=True, row_groups=0)
        
        # Create sparse matrix
        counter = 0
        # Add initial state corresponding to the given state-action pair
        for state in range(self.num_states):
            builder.new_row_group(counter)
            for action in range(self.num_actions):
                next_states = self.transitions[state][action]
                if(len(next_states) > 0):
                    for next_state in next_states:
                        bounds = self.intervals[(state, action, next_state)]
                        builder.add_next_value(counter, next_state, pycarl.Interval(bounds[0], bounds[1]))
                else:
                      builder.add_next_value(counter, state, pycarl.Interval(1, 1))
                counter+=1
                

        transition_matrix = builder.build()
        state_labels = self.set_state_labels()
        choice_labels = self.set_choice_labels()
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
    
    def set_state_labels(self):
        state_labeling = stormpy.storage.StateLabeling(self.num_states)
        labels = {"goal", "eaten", "maze"}
        for label in labels:
            state_labeling.add_label(label)
        for state in range(self.num_states):
            if state in self.traps:
                state_labeling.add_label_to_state("eaten", state)
            elif state in self.goal:
                state_labeling.add_label_to_state("goal", state)
            else:
                state_labeling.add_label_to_state("maze", state)
        return state_labeling
    
    def set_choice_labels(self):
        choice_labeling = stormpy.storage.ChoiceLabeling(self.num_actions*self.num_states)
        choice_labels = {"Up", "Right", "Down", "Left"}
        for label in choice_labels:
            choice_labeling.add_label(label)
            
        for i in range(self.num_states):
            choice_labeling.add_label_to_choice("Up", i*4)
            choice_labeling.add_label_to_choice("Right", i*4+1)
            choice_labeling.add_label_to_choice("Down", i*4+2)
            choice_labeling.add_label_to_choice("Left", i*4+3) 
        
        return choice_labeling
    
class IntervalMDPBuilderAirplaneSmall(IntervalMDPBuilderSmall):
    def build_model(self):
        builder = stormpy.IntervalSparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False, has_custom_row_grouping=True, row_groups=0)
        
        counter = 0    
        # Add the model
        for state in range(self.num_states):
            builder.new_row_group(counter)
            for action in range(self.num_actions):
                next_states = self.transitions[state][action]
                # check if next state is a terminal state. If so, only add a transition to itself
                if(len(next_states) > 0):
                    for next_state in next_states:
                        bounds = self.intervals[(state, action, next_state)]
                        builder.add_next_value(counter, next_state, pycarl.Interval(bounds[0], bounds[1]))
                else:
                      builder.add_next_value(counter, state, pycarl.Interval(1, 1))  
                counter+=1
                
        transition_matrix = builder.build()
        state_labels = self.set_state_labels()
        choice_labels = self.set_choice_labels()
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
    
    def set_state_labels(self):
        state_labeling = stormpy.storage.StateLabeling(self.num_states)
        labels = {"success", "crash", "init"}
        for label in labels:
            state_labeling.add_label(label)
        for i in self.traps:
            state_labeling.add_label_to_state("crash", i)
        for i in self.goal:
            state_labeling.add_label_to_state("success", i)
        state_labeling.add_label_to_state("init", 0)
        
        return state_labeling
    
    def set_choice_labels(self):
        choice_labeling = stormpy.storage.ChoiceLabeling(self.num_actions*self.num_states)
        choice_labels = {"down", "up", "stay"}
        for label in choice_labels:
            choice_labeling.add_label(label)
            
        for i in range(self.num_states):
            choice_labeling.add_label_to_choice("down", i*3)
            choice_labeling.add_label_to_choice("up", i*3+1)
            choice_labeling.add_label_to_choice("stay", i*3+2)
        
        return choice_labeling
    
class IntervalMDPBuilderSlipperyGridworldSmall(IntervalMDPBuilderSmall):
    def build_model(self):
        # initialize builder
        # print(next_states_init)
        builder = stormpy.IntervalSparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False, has_custom_row_grouping=True, row_groups=0)
        
        # Create sparse matrix
        counter = 0
        for state in range(self.num_states):
            builder.new_row_group(counter)
            for action in range(self.num_actions):
                next_states = self.transitions[state][action]
                if(len(next_states) > 0):
                    for next_state in next_states:
                        bounds = self.intervals[(state, action, next_state)]
                        builder.add_next_value(counter, next_state, pycarl.Interval(bounds[0], bounds[1]))
                else:
                      builder.add_next_value(counter, state, pycarl.Interval(1, 1))
                    #   print(f"We do this for state {state}")  
                counter+=1
                

        transition_matrix = builder.build()
        state_labels = self.set_state_labels()
        choice_labels = self.set_choice_labels()
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
    
    def set_state_labels(self):
        state_labeling = stormpy.storage.StateLabeling(self.num_states)
        labels = {"goal", "trap", "init", "save"}
        for label in labels:
            state_labeling.add_label(label)
        for state in range(self.num_states):
            if state in self.traps:
                state_labeling.add_label_to_state("trap", state)
            if state in self.goal:
                state_labeling.add_label_to_state("goal", state)
            else:
                state_labeling.add_label_to_state("save", state)
        state_labeling.add_label_to_state("init", 0)
        
        return state_labeling
    
    def set_choice_labels(self):
        choice_labeling = stormpy.storage.ChoiceLabeling(self.num_actions*self.num_states)
        choice_labels = {"North", "East", "South", "West", "Escape", "Reset"}
        for label in choice_labels:
            choice_labeling.add_label(label)
            
        for i in range(self.num_states):
            if i in self.traps:
                choice_labeling.add_label_to_choice("Escape", i*4)
                choice_labeling.add_label_to_choice("Escape", i*4+1)
                choice_labeling.add_label_to_choice("Reset", i*4+2)
                choice_labeling.add_label_to_choice("Reset", i*4+3)                 
            else:    
                choice_labeling.add_label_to_choice("North", i*4)
                choice_labeling.add_label_to_choice("East", i*4+1)
                choice_labeling.add_label_to_choice("South", i*4+2)
                choice_labeling.add_label_to_choice("West", i*4+3) 
        return choice_labeling
    
    
    
class IntervalMDPBuilderPrism(IntervalMDPBuilderSmall):
    def build_model(self):
        # initialize builder
        # print(next_states_init)
        builder = stormpy.IntervalSparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False, has_custom_row_grouping=True, row_groups=0)
        
        # Create sparse matrix
        counter = 0
        for state in range(self.num_states):
            builder.new_row_group(counter)
            for action in range(self.num_actions):
                next_states = self.transitions[state][action]
                if(len(next_states) > 0):
                    for next_state in next_states:
                        bounds = self.intervals[(state, action, next_state)]
                        builder.add_next_value(counter, next_state, pycarl.Interval(bounds[0], bounds[1]))
                else:
                      builder.add_next_value(counter, state, pycarl.Interval(1, 1))
                    #   print(f"We do this for state {state}")  
                counter+=1
                

        transition_matrix = builder.build()
        state_labels = self.set_state_labels()
        choice_labels = self.set_choice_labels()
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
    
    def set_state_labels(self):
        state_labeling = stormpy.storage.StateLabeling(self.num_states)
        labels = {"goal", "trap", "init"}
        for label in labels:
            state_labeling.add_label(label)
        for state in range(self.num_states):
            if state in self.traps:
                state_labeling.add_label_to_state("trap", state)
            if state in self.goal:
                state_labeling.add_label_to_state("goal", state)
        state_labeling.add_label_to_state("init", 0)
        
        return state_labeling
    
    def set_choice_labels(self):
        choice_labeling = stormpy.storage.ChoiceLabeling(self.num_actions*self.num_states)
        choice_labels = {"action"}
        for label in choice_labels:
            choice_labeling.add_label(label)
            
        for i in range(self.num_actions*self.num_states):
            choice_labeling.add_label_to_choice("action", i)

        return choice_labeling