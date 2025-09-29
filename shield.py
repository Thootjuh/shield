import stormpy
import PrismEncoder
import tempfile
import os
import time
import numpy as np
from IntervalMDPBuilder import IntervalMDPBuilderPacman, IntervalMDPBuilderRandomMDP, IntervalMDPBuilderAirplane, IntervalMDPBuilderSlipperyGridworld, IntervalMDPBuilderWetChicken, IntervalMDPBuilderPrism, IntervalMDPBuilderTaxi, IntervalMDPBuilderFrozenLake
import pycarl
 
class Shield:
    def __init__(self, transition_matrix, traps, goal, intervals):
        """
        args:
        transition_matrix (np.ndarray): 
            A NumPy array of shape (num_states, num_actions), where each element is a list of possible
            next states for the corresponding state-action pair.

        traps (list[int]):
            A list of state indices that represent trap states (undesirable or absorbing states).

        goal (list[int]):
            A list of state indices that represent goal states (desirable or target states).

        intervals (dict[tuple[int, int, int], tuple[float, float]])
            A dictionary mapping (state, action, next_state) tuples to a (min, max) interval tuple representing
            the range of possible transition probabilities due to uncertainty.
        """
        self.traps = traps
        self.goal = goal
        self.structure = transition_matrix
        self.intervals = intervals
        self.num_states= len(self.structure)
        self.num_actions = len(self.structure[0])
        self.shield = np.full((self.num_states, self.num_actions), -1, dtype=np.float64)
    
    def get_probs(self, model, prop):
        """
        calculate the probability of satisfying the reach-avoid specification using the STORM model checker

        Args:
            model (stormpy.storage.storage.SparseIntervalMdp): 
                stormpy representation of the intervalMDP
            prop (str): 
                the reach-avoid specification writen as a temporal logic property

        Returns:
            probs (list[float]): calculated probability of each state leading to a violation of the property
            transition_probs(dict[int, pycarl.Interval]): the chance that each action is taken
        """
        properties = stormpy.parse_properties(prop)
        env = stormpy.Environment()
        env.solver_environment.minmax_solver_environment.method = stormpy.MinMaxMethod.value_iteration
        task = stormpy.CheckTask(properties[0].raw_formula, only_initial_states=False)
        task.set_produce_schedulers()
        result = stormpy.check_interval_mdp(model, task, env)
        
        probs = result.get_values()
        transition_probs = {}
        transition_probs = {}
        for state in model.states:
            for action in state.actions:
                transitions = {}
                for transition in action.transitions:
                    transition.value
                    transitions[transition.column] = transition.value()
                transition_probs[(state.id, action.id)] = transitions
        return probs, transition_probs
    
    def calculateShieldInterval(self, prop, model):
        """
        Calculate the probability of violating the given specification for each state-action pair

        Args:
            model (stormpy.storage.storage.SparseIntervalMdp): 
                stormpy representation of the intervalMDP
            prop (str): 
                the reach-avoid specification writen as a temporal logic property
        """
        start_total_time = time.time()
        state_probabilities, transition_probabilities = self.get_probs(model, prop)
        for state in range(len(self.structure)):
            for action in range(len(self.structure[state])):
                if state in self.traps:
                    self.shield[state][action] = 1
                elif state in self.goal:
                    self.shield[state][action] = 0
                else:
                    try:
                        trans = transition_probabilities[(state, action)]
                    except KeyError:
                        self.shield[state][action] = 1
                    # Find the worst case transition probabilities
                    worst_case_transitions = {}
                    for next_state, trans_prob in trans.items():
                        worst_case_transitions[next_state] = trans_prob.lower()
                    remaining_states = list(worst_case_transitions.keys())
                    total_mass = sum(worst_case_transitions.values())
                    
                    #get the worst next state and add mass untill we reach upper
                    while total_mass < 1 and len(remaining_states) >= 1:
                        # get the worst next state
                        worst_next_state = min(remaining_states, key=lambda i: state_probabilities[i])
                        remaining_states = [state for state in remaining_states if state != worst_next_state]
                        # add mass untill we reach max_prob or total_mass = 1
                        bounds = trans[worst_next_state]
                        difference = bounds.upper()-bounds.lower()
                        added_mass = min(difference, 1-total_mass)
                        total_mass+= added_mass
                        worst_case_transitions[worst_next_state] = worst_case_transitions[worst_next_state]+added_mass
                    
                    value = 0  
                    for next_state, trans_prob in worst_case_transitions.items():                        
                        value += trans_prob*state_probabilities[next_state]
                    self.shield[state][action] = max(min(1.0, 1-value), 0.0)
        end_total_time = time.time()
        

        print("Total time needed to create the Shield:", end_total_time - start_total_time)     
                        
        
    def printShield(self):
        """
        print the shield
        """
        sorted_arr = self.shield[self.shield[:, 2].argsort()]
        np.set_printoptions(suppress=True, precision=6)
        print(sorted_arr)
        print(self.traps)


class ShieldRandomMDP(Shield):
    # Calculate the shield for the Random MDPs environment
    def __init__(self, transition_matrix, traps, goal, intervals):
        """
        args:
        transition_matrix (np.ndarray): 
            A NumPy array of shape (num_states, num_actions), where each element is a list of possible
            next states for the corresponding state-action pair.

        traps (list[int]):
            A list of state indices that represent trap states (undesirable or absorbing states).

        goal (list[int]):
            A list of state indices that represent goal states (desirable or target states).

        intervals (dict[tuple[int, int, int], tuple[float, float]])
            A dictionary mapping (state, action, next_state) tuples to a (min, max) interval tuple representing
            the range of possible transition probabilities due to uncertainty.
        """
        self.model_builder = IntervalMDPBuilderRandomMDP(transition_matrix, intervals, goal, traps)
        super().__init__(transition_matrix, traps, goal, intervals)
        
    def calculateShield(self):
        """
        calculate the probability of violating the safety specification for the Random MDPs environment
        """
        # How likely are we to step into a trap
        prop = "Pmax=? [!\"trap\" U \"goal\"]"
        # prop = "Pmin=? [  F<4 \"trap\" ]"
        # prop1 = "Pmax=? [  F \"trap\" ]"
        
        # Is it possible to reach the goal
        # prop2 = "Pmax=? [F \"goal\" & !F<4 \"trap\"]"
        # prop2 = "Pmax=? [F \"reach\" & !F \"trap\"]"
        # prop3 = "Pmin=? [F<=5 \"reach\"]"
        super().calculateShieldInterval(prop, self.model_builder.build_model())
        
    def get_safe_actions_from_shield(self, state, threshold=0.2, buffer = 0.05):
        """
        calculate the actions allowed by the shield for a given state
        Args:
            state (int): the state for which we want compute the safe actions
            threshold (float, optional): maximum probability for which we will accept an action. Defaults to 0.2.
            buffer (float, optional): we allow any actions within the buffer of the best action. Defaults to 0.05.

        Returns:
            safe_actions (list[int]): list containing the actions deemed to be 'safe' by the shield
        """
        probs = self.shield[state]
        safe_actions = []
        for i, prob in enumerate(probs):
            if prob >= 0 and prob <= threshold:
                safe_actions.append(i)

        if len(safe_actions) == 0:
            min_value = np.min(probs)
            safe_actions = np.where(probs <= min_value+buffer)[0].tolist()
        return safe_actions
    
    def printShield(self):
        """
        print the probabilities associated with each state-action pair
        """
        state_action_prob_pairs = []
        for state in range(len(self.shield)):
            for action in range(len(self.shield[state])):
                prob = self.shield[state][action]
                state_action_prob_pairs.append([state, action, prob])
        # state_action_prob_pairs = sorted(state_action_prob_pairs, key=lambda x: x[2])      
        
        for pair in state_action_prob_pairs:
            state = pair[0]
            action = pair[1]
            prob = pair[2]
                        
            print(f"State: {state}, action: {action}, with probability of falling: {prob}")
        print(f"with crash states being {self.traps} and success states being {self.goal}")

class ShieldWetChicken(Shield):
    # Calculate the shield for the wet chicken environment

    def __init__(self, transition_matrix, width, length, goals, intervals):
        """
        args:
        transition_matrix (np.ndarray): 
            A NumPy array of shape (num_states, num_actions), where each element is a list of possible
            next states for the corresponding state-action pair.

        width (int): 
            width of the environment
        
        length (int):
            length of the environment

        goals (list[int]):
            A list of state indices that represent goal states (desirable or target states).

        intervals (dict[tuple[int, int, int], tuple[float, float]])
            A dictionary mapping (state, action, next_state) tuples to a (min, max) interval tuple representing
            the range of possible transition probabilities due to uncertainty.
        """
        self.width = width
        self.length = length
        self.model_builder = IntervalMDPBuilderWetChicken(transition_matrix, intervals, [], [])
        super().__init__(transition_matrix, [], [], intervals)
    
    def printShield(self):
        """
        print the probabilities associated with each state-action pair
        """
        actions = [
            "drift",
            "hold",
            "paddle_back",
            "right",
            "left",
        ]
        state_action_prob_pairs = []
        for state in range(len(self.shield)):
            for action in range(len(self.shield[state])):
                prob = self.shield[state][action]
                state_action_prob_pairs.append([state, action, prob])
        # state_action_prob_pairs = sorted(state_action_prob_pairs, key=lambda x: x[2])      
        
        for pair in state_action_prob_pairs:
                state = pair[0]
                action = actions[pair[1]]
                prob = pair[2]
                x = int(state / self.length)
                y = int(state % self.width)
                print(f"State: ({x}, {y}), action: {action}, with probability of falling: {prob}")
            
        print("with falling states being")
        for state in self.traps:
            x = int(state / self.length)
            y = int(state % self.width)
            print(f"({x},{y})")
            
    def calculateShield(self):
        """
        calculate the probability of violating the safety specification for the Wet Chicken environment
        """
        # How likely are we to step into a trap
        prop = "Pmax=? [  !\"waterfall\" U \"goal\"]"
        # prop = "Pmin=? [  F\"waterfall\"]"
        # prop = "Pmax=? [  !F<2\"waterfall\"]"
        super().calculateShieldInterval(prop, self.model_builder.build_model())
        
    def get_safe_actions_from_shield(self, state, threshold=0.2, buffer = 0.05):
        """
        calculate the actions allowed by the shield for a given state
        Args:
            state (int): the state for which we want compute the safe actions
            threshold (float, optional): maximum probability for which we will accept an action. Defaults to 0.2.
            buffer (float, optional): we allow any actions within the buffer of the best action. Defaults to 0.05.

        Returns:
            safe_actions (list[int]): list containing the actions deemed to be 'safe' by the shield
        """
        probs = self.shield[state]
        safe_actions = []
        for i, prob in enumerate(probs):
            if prob >= 0 and prob <= threshold:
                safe_actions.append(i)

        if len(safe_actions) == 0:
            min_value = np.min(probs)
            safe_actions = np.where(probs <= min_value+buffer)[0].tolist()
        print(safe_actions)
        return safe_actions
    
class ShieldAirplane(Shield):
    def __init__(self, transition_matrix, traps, goal, intervals, maxX, maxY):
        """
        Args:
            transition_matrix (np.ndarray): 
                A NumPy array of shape (num_states, num_actions), where each element is a list of possible
                next states for the corresponding state-action pair.
            traps (list[int]):
                A list of state indices that represent trap states (undesirable or absorbing states).
            goal (list[int]):
                A list of state indices that represent goal states (desirable or target states).
            intervals (dict[tuple[int, int, int], tuple[float, float]])
                A dictionary mapping (state, action, next_state) tuples to a (min, max) interval tuple representing
                the range of possible transition probabilities due to uncertainty.
            maxX (int): maximum x position of the airplane
            maxY (int): maximum y position of the airplane
        """
        self.maxX = maxX
        self.maxY = maxY
        self.model_builder = IntervalMDPBuilderAirplane(transition_matrix, intervals, goal, traps)
        super().__init__(transition_matrix, traps, goal, intervals)
        
    def calculateShield(self):
        """
        calculate the probability of violating the safety specification for the Airplane environment
        """
        # How likely are we to step into a trap
        prop = "Pmax=? [  !\"crash\" U \"success\"]"
        # prop = "Pmin=? [  F\"waterfall\"]"
        # prop = "Pmax=? [  !F<2\"waterfall\"]"
        super().calculateShieldInterval(prop, self.model_builder.build_model())
        
    def decode_int(self, state_int):   
        """
        get the x, y, and adversarial y coordinates for a given state

        Args:
            state_int (int): the state we want to decode

        Returns:
            x (int): x position of both planes
            y (int): y position of the agent airplane
            ay (int): y position of the adversarial airplane
        """
        ay = state_int % self.maxY
        state_int //= self.maxY
        
        y = state_int % self.maxY
        state_int //= self.maxY
        
        x = state_int % self.maxX
        
        return x, y, ay
    
    def printShield(self):
        """
        print the probabilities associated with each state-action pair
        """
        actions = [
            "down",
            "up",
            "stay",
        ]
        state_action_prob_pairs = []
        for state in range(len(self.shield)):
            for action in range(len(self.shield[state])):
                prob = self.shield[state][action]
                state_action_prob_pairs.append([state, action, prob])
        # state_action_prob_pairs = sorted(state_action_prob_pairs, key=lambda x: x[2])      
        
        for pair in state_action_prob_pairs:
            state = pair[0]
            action = actions[pair[1]]
            prob = pair[2]
            
            x, y, ay = self.decode_int(state)
            
            print(f"State: {state}: {x}, {y}, {ay}), action: {action}, with probability of crashing: {prob}")
        print(f"with crash states being {self.traps} and success states being {self.goal}")
                
    def get_safe_actions_from_shield(self, state, threshold=0.2, buffer = 0.1):
        """
        calculate the actions allowed by the shield for a given state
        Args:
            state (int): the state for which we want compute the safe actions
            threshold (float, optional): maximum probability for which we will accept an action. Defaults to 0.2.
            buffer (float, optional): we allow any actions within the buffer of the best action. Defaults to 0.1.

        Returns:
            safe_actions (list[int]): list containing the actions deemed to be 'safe' by the shield
        """
        probs = self.shield[state]
        safe_actions = []
        for i, prob in enumerate(probs):
            if prob >= 0 and prob <= threshold:
                safe_actions.append(i)

        if len(safe_actions) == 0:
            min_value = np.min(probs)
            safe_actions = np.where(probs <= min_value+buffer)[0].tolist()
        return safe_actions
    
class ShieldSlipperyGridworld(Shield):
    # Calculate the shield for the Slippery Gridworld environment
    def __init__(self, transition_matrix, traps, goal, intervals, width, height):
        """
        Args:
            transition_matrix (np.ndarray): 
                A NumPy array of shape (num_states, num_actions), where each element is a list of possible
                next states for the corresponding state-action pair.
            traps (list[int]):
                A list of state indices that represent trap states (undesirable or absorbing states).
            goal (list[int]):
                A list of state indices that represent goal states (desirable or target states).
            intervals (dict[tuple[int, int, int], tuple[float, float]])
                A dictionary mapping (state, action, next_state) tuples to a (min, max) interval tuple representing
                the range of possible transition probabilities due to uncertainty.
            width (int): 
                width of the gridworld
            height (int): 
                height of the gridworld
        """
        self.width = width
        self.height = height
        self.model_builder = IntervalMDPBuilderSlipperyGridworld(transition_matrix, intervals, goal, traps)
        super().__init__(transition_matrix, traps, goal, intervals)
    
    def calculateShield(self):
        """
        calculate the probability of violating the safety specification for the Slippery Gridworld environment
        """
        # How likely are we to step into a trap
        prop = "Pmax=? [!\"trap\"U\"goal\"]"
        # prop = "Pmax=? [!\"trap\"U\"save\"]"
        # prop = "Pmax=? [!\"save\"U\"trap\"]"
        
        super().calculateShieldInterval(prop, self.model_builder.build_model())
        # self.printShield()
        # self.shield = 1-self.shield
    
    def get_safe_actions_from_shield(self, state, threshold=0.1, buffer = 0.1):
        """
        calculate the actions allowed by the shield for a given state
        Args:
            state (int): the state for which we want compute the safe actions
            threshold (float, optional): maximum probability for which we will accept an action. Defaults to 0.1.
            buffer (float, optional): we allow any actions within the buffer of the best action. Defaults to 0.1.

        Returns:
            safe_actions (list[int]): list containing the actions deemed to be 'safe' by the shield
        """
        probs = self.shield[state]
        safe_actions = []
        for i, prob in enumerate(probs):
            if prob >= 0 and prob <= threshold:
                safe_actions.append(i)

        if len(safe_actions) == 0:
            min_value = np.min(probs)
            safe_actions = np.where(probs <= min_value+buffer)[0].tolist()
        return safe_actions
    
    def get_state_from_int(self, number):
        """
        get the x and y coordinate for a given state

        Args:
            number (int): the state

        Returns:
            x (int): x coordinate
            y (int): y coordinate
        """
        x = number // (self.width)
        y = number % (self.width)
        
        return x, y
    
    def printShield(self):
        """
        print the probabilities associated with each state-action pair
        """
        actions = [
            "North",
            "East",
            "South",
            "West"
        ]
        state_action_prob_pairs = []
        for state in range(len(self.shield)):
            for action in range(len(self.shield[state])):
                prob = self.shield[state][action]
                state_action_prob_pairs.append([state, action, prob])
        # state_action_prob_pairs = sorted(state_action_prob_pairs, key=lambda x: x[2])      
        
        for pair in state_action_prob_pairs:
            state = pair[0]
            prob = pair[2]
            
            x, y = self.get_state_from_int(state)
            if state in self.traps:
                if pair[1] < 2:
                    action = "escape"
                else:
                    action = "reset"
                print(f"Trap State: {state}: {x}, {y}), action: {action}, with probability of getting trapped: {prob}")
            else:
                action = actions[pair[1]]
                print(f"State: {state}: {x}, {y}), action: {action}, with probability of getting trapped: {prob}")
        print(f"with crash states being {self.traps} and success states being {self.goal}")
    
class ShieldSimplifiedPacman(Shield):
    # Calculate the shield for the Simplified Pacman environment
    def __init__(self, transition_matrix, traps, goal, intervals, width, height):
        """
        args:
            transition_matrix (np.ndarray): 
                A NumPy array of shape (num_states, num_actions), where each element is a list of possible
                next states for the corresponding state-action pair.
            traps (list[int]):
                A list of state indices that represent trap states (undesirable or absorbing states).
            goal (list[int]):
                A list of state indices that represent goal states (desirable or target states).
            intervals (dict[tuple[int, int, int], tuple[float, float]])
                A dictionary mapping (state, action, next_state) tuples to a (min, max) interval tuple representing
                the range of possible transition probabilities due to uncertainty.
            width (int): 
                width of the maze
            height (int): 
                height of the maze
        """
        self.width = width
        self.height = height
        self.model_builder = IntervalMDPBuilderPacman(transition_matrix, intervals, goal, traps)
        super().__init__(transition_matrix, traps, goal, intervals)
        

    def calculateShield(self):
        """
        calculate the probability of violating the safety specification for the Simplified Pacman environment
        """
        # How likely are we to step into a trap
        prop = "Pmax=? [!\"eaten\"U\"goal\"]"
        
        super().calculateShieldInterval(prop, self.model_builder.build_model())
    
    
    def get_safe_actions_from_shield(self, state, threshold=0.01, buffer = 0.05):
        """
        calculate the actions allowed by the shield for a given state
        Args:
            state (int): the state for which we want compute the safe actions
            threshold (float, optional): maximum probability for which we will accept an action. Defaults to 0.2.
            buffer (float, optional): we allow any actions within the buffer of the best action. Defaults to 0.05.

        Returns:
            safe_actions (list[int]): list containing the actions deemed to be 'safe' by the shield
        """
        probs = self.shield[state]
        safe_actions = []
        for i, prob in enumerate(probs):
            if prob >= 0 and prob <= threshold:
                safe_actions.append(i)

        if len(safe_actions) == 0:
            min_value = np.min(probs)
            safe_actions = np.where(probs <= min_value+buffer)[0].tolist()
        return safe_actions  
    
    def printShield(self):
        """
        print the probabilities associated with each state-action pair
        """
        actions = [
            "Up",
            "Right",
            "Down",
            "Left"
        ]
        state_action_prob_pairs = []
        for state in range(len(self.shield)):
            for action in range(len(self.shield[state])):
                prob = self.shield[state][action]
                state_action_prob_pairs.append([state, action, prob])
        # state_action_prob_pairs = sorted(state_action_prob_pairs, key=lambda x: x[2])      
        
        for pair in state_action_prob_pairs:
            state = pair[0]
            prob = pair[2]
            action = actions[pair[1]]
            x, y, gx1, gy1 = self.decode_int(state)
            if prob > 0.01:
                print(f"Trap State: {state} action: {action}, with probability of getting trapped: {prob}")
        print(f"with crash states being {self.traps} and success states being {self.goal}")
        
class ShieldTaxi(Shield):
    # Calculate the shield for the Taxi environment
    def __init__(self, transition_matrix, traps, goal, intervals, init=0):
        """
        args:
            transition_matrix (np.ndarray): 
                A NumPy array of shape (num_states, num_actions), where each element is a list of possible
                next states for the corresponding state-action pair.
            traps (list[int]):
                A list of state indices that represent trap states (undesirable or absorbing states).
            goal (list[int]):
                A list of state indices that represent goal states (desirable or target states).
            intervals (dict[tuple[int, int, int], tuple[float, float]])
                A dictionary mapping (state, action, next_state) tuples to a (min, max) interval tuple representing
                the range of possible transition probabilities due to uncertainty.
            init(int, optional): the initial state
        """
        self.model_builder = IntervalMDPBuilderTaxi(transition_matrix, intervals, goal, traps)
        self.init = init
        super().__init__(transition_matrix, traps, goal, intervals)
        
   
    def calculateShield(self):
        """
        calculate the probability of violating the safety specification for the Taxi environment
        """
        # How likely are we to step into a trap
        prop = "Pmax=? [!\"crash\"U\"goal\"]"
    
        # self.printShield()
        # self.shield[:] = 0
        super().calculateShieldInterval(prop, self.model_builder.build_model())
        # self.printShield()
        # self.printShield()
    
    def get_safe_actions_from_shield(self, state, threshold=0.3, buffer = 0.05):
        """
        calculate the actions allowed by the shield for a given state
        Args:
            state (int): the state for which we want compute the safe actions
            threshold (float, optional): maximum probability for which we will accept an action. Defaults to 0.3.
            buffer (float, optional): we allow any actions within the buffer of the best action. Defaults to 0.05.

        Returns:
            safe_actions (list[int]): list containing the actions deemed to be 'safe' by the shield
        """
        probs = self.shield[state]
        safe_actions = []
        for i, prob in enumerate(probs):
            if prob >= 0 and prob <= threshold:
                safe_actions.append(i)

        if len(safe_actions) == 0:
            min_value = np.min(probs)
            safe_actions = np.where(probs <= min_value+buffer)[0].tolist()
        return safe_actions  
    
    def decode(self, i):
        """
        decode a given state into its components

        Args:
            i (int): state

        Returns:
            list[int]: list containing the x and y positions of the taxi, the destination and the passengers location
        """
        if i == self.num_states-1:
            out = [-1, -1, -1, -1]
            return reversed(out)
        if i == self.num_states-2:
            out = [-2, -2, -2, -2]
            return reversed(out)
        if i == self.num_states-3:
            out = [-3, -3, -3, -3]
            return reversed(out) 
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        return reversed(out)
    
    def printShield(self):
        """
        print the probabilities associated with each state-action pair
        """
        ACTIONS = {
            0 : "south",
            1 : "north",
            2 : "east",
            3 : "west",
            4 : "pickup",
            5 : "drop off"
        }
            
        
        state_action_prob_pairs = []
        for state in range(len(self.shield)):
            for action in range(len(self.shield[state])):
                prob = self.shield[state][action]
                state_action_prob_pairs.append([state, action, prob])
        # state_action_prob_pairs = sorted(state_action_prob_pairs, key=lambda x: x[2])      
        
        for pair in state_action_prob_pairs:
            state = pair[0]
            act = pair[1]
            prob = pair[2]
            taxi_row, taxi_col, pass_loc, dest_idx = self.decode(state)
            if (prob > 0.1):
                print(f"State: {state}: ({taxi_row},{taxi_col}, pass at {pass_loc}, dest at {dest_idx}), taking action {ACTIONS[act]} with probability of getting trapped: {prob}")
        
        print(f"with crash states being {self.traps} and success states being {self.goal}")


class ShieldFrozenLake(Shield):
    # Calculate the shield for the Frozen Lake environment
    def __init__(self, transition_matrix, traps, goal, intervals):
        """
        args:
            transition_matrix (np.ndarray): 
                A NumPy array of shape (num_states, num_actions), where each element is a list of possible
                next states for the corresponding state-action pair.
            traps (list[int]):
                A list of state indices that represent trap states (undesirable or absorbing states).
            goal (list[int]):
                A list of state indices that represent goal states (desirable or target states).
            intervals (dict[tuple[int, int, int], tuple[float, float]])
                A dictionary mapping (state, action, next_state) tuples to a (min, max) interval tuple representing
                the range of possible transition probabilities due to uncertainty.
        """
        self.model_builder = IntervalMDPBuilderFrozenLake(transition_matrix, intervals, goal, traps)
        super().__init__(transition_matrix, traps, goal, intervals)
        self.grid_size = np.sqrt(self.num_states)
        
    def calculateShield(self):
        """
        calculate the probability of violating the safety specification for the Frozen Lake environment
        """
        # How likely are we to step into a trap
        prop = "Pmax=? [!\"hole\"U\"goal\"]"
        
        # self.printShield()
        # self.shield[:] = 0
        super().calculateShieldInterval(prop, self.model_builder.build_model())
        # self.printShield()
        # self.printShield()
    
    def get_safe_actions_from_shield(self, state, threshold=0.2, buffer = 0.02):
        """
        calculate the actions allowed by the shield for a given state
        Args:
            state (int): the state for which we want compute the safe actions
            threshold (float, optional): maximum probability for which we will accept an action. Defaults to 0.2.
            buffer (float, optional): we allow any actions within the buffer of the best action. Defaults to 0.02.

        Returns:
            safe_actions (list[int]): list containing the actions deemed to be 'safe' by the shield
        """
        probs = self.shield[state]
        safe_actions = []
        for i, prob in enumerate(probs):
            if prob >= 0 and prob <= threshold:
                safe_actions.append(i)

        if len(safe_actions) == 0:
            min_value = np.min(probs)
            safe_actions = np.where(probs <= min_value+buffer)[0].tolist()
        return safe_actions  
    
    def decode(self, state):
        """
        Decode a state into its row and collumn position

        Args:
            state (int): the state to be decoded

        Returns:
            col (int): the current collumn the agent is in
            row (int): the current row the agent is in
        """
        col = int(state % self.grid_size)
        row = int(state // self.grid_size)
        return col, row
    def printShield(self):
        """
        print the probabilities associated with each state-action pair
        """
        ACTIONS = {
            0 : "left",
            1 : "down",
            2 : "right",
            3 : "up",
        }   
        
        state_action_prob_pairs = []
        for state in range(len(self.shield)):
            for action in range(len(self.shield[state])):
                prob = self.shield[state][action]
                state_action_prob_pairs.append([state, action, prob])
        # state_action_prob_pairs = sorted(state_action_prob_pairs, key=lambda x: x[2])      
        
        for pair in state_action_prob_pairs:
            state = pair[0]
            act = pair[1]
            prob = pair[2]
            col, row = self.decode(state)
            print(f"State: {state}: ({row},{col}, taking action {ACTIONS[act]} with probability of getting trapped: {prob}")
        
        print(f"with crash states being {self.traps} and success states being {self.goal}")
                    
class ShieldPrism(Shield):
    def __init__(self, transition_matrix, traps, goal, intervals):
        """
        args:
            transition_matrix (np.ndarray): 
                A NumPy array of shape (num_states, num_actions), where each element is a list of possible
                next states for the corresponding state-action pair.
            traps (list[int]):
                A list of state indices that represent trap states (undesirable or absorbing states).
            goal (list[int]):
                A list of state indices that represent goal states (desirable or target states).
            intervals (dict[tuple[int, int, int], tuple[float, float]])
                A dictionary mapping (state, action, next_state) tuples to a (min, max) interval tuple representing
                the range of possible transition probabilities due to uncertainty.
        """
        self.model_builder = IntervalMDPBuilderPrism(transition_matrix, intervals, goal, traps)
        super().__init__(transition_matrix, traps, goal, intervals)
        
   
    def calculateShield(self):
        """
        calculate the probability of violating the safety specification for any prism environment
        """
        # How likely are we to step into a trap
        prop = "Pmax=? [!\"trap\"U\"goal\"]"
        
        # self.printShield()
        # self.shield[:] = 0
        super().calculateShieldInterval(prop, self.model_builder.build_model())
        # self.printShield()
        # self.printShield()
    
    def get_safe_actions_from_shield(self, state, threshold=0.2, buffer = 0.05):
        """
        calculate the actions allowed by the shield for a given state
        Args:
            state (int): the state for which we want compute the safe actions
            threshold (float, optional): maximum probability for which we will accept an action. Defaults to 0.2.
            buffer (float, optional): we allow any actions within the buffer of the best action. Defaults to 0.05.

        Returns:
            safe_actions (list[int]): list containing the actions deemed to be 'safe' by the shield
        """
        probs = self.shield[state]
        safe_actions = []
        for i, prob in enumerate(probs):
            if prob >= 0 and prob <= threshold:
                safe_actions.append(i)

        if len(safe_actions) == 0:
            min_value = np.min(probs)
            safe_actions = np.where(probs <= min_value+buffer)[0].tolist()
        return safe_actions  
    
    def printShield(self):
        """
        print the probabilities associated with each state-action pair
        """
        state_action_prob_pairs = []
        for state in range(len(self.shield)):
            for action in range(len(self.shield[state])):
                prob = self.shield[state][action]
                state_action_prob_pairs.append([state, action, prob])
        # state_action_prob_pairs = sorted(state_action_prob_pairs, key=lambda x: x[2])      
        
        for pair in state_action_prob_pairs:
            state = pair[0]
            prob = pair[2]
            print(f"State: {state}:  with probability of getting trapped: {prob}")
        
        print(f"with crash states being {self.traps} and success states being {self.goal}")
        

    
    
    
    
