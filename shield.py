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
                # elif state in self.goal:
                #     self.shield[state][action] = 0
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
        

class ShieldCartpole(Shield):
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
        self.model_builder = IntervalMDPBuilderPrism(transition_matrix, intervals, goal, traps) 
        super().__init__(transition_matrix, traps, goal, intervals)
        
    def calculateShield(self):
        """
        calculate the probability of violating the safety specification for the Random MDPs environment
        """
        # How likely are we to step into a trap
        # prop = "Pmax=? [!\"trap\" U \"goal\"]"
        prop = "Pmax=? [!\"goal\"U\"trap\"]"
        # prop = "Pmin=? [  F<4 \"trap\" ]"
        # prop1 = "Pmax=? [  F \"trap\" ]"
        
        # Is it possible to reach the goal
        # prop2 = "Pmax=? [F \"goal\" & !F<4 \"trap\"]"
        # prop2 = "Pmax=? [F \"reach\" & !F \"trap\"]"
        # prop3 = "Pmin=? [F<=5 \"reach\"]"
        super().calculateShieldInterval(prop, self.model_builder.build_model())
        self.shield = 1-self.shield
        
    def get_safe_actions_from_shield(self, state, threshold=0.0, buffer = 0.05):
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
        state_action_prob_pairs = sorted(state_action_prob_pairs, key=lambda x: x[2])      
        state_action_prob_pairs = [i for i in state_action_prob_pairs if i[2] < 0.95 and i[2]> 0]
        for pair in state_action_prob_pairs:
            state = pair[0]
            action = pair[1]
            prob = pair[2]
                        
            print(f"State: {state}, action: {action}, with probability of falling: {prob}")
        print(f"with crash states being {self.traps} and success states being {self.goal}")


        

    
    
    
    
