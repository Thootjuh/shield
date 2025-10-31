import stormpy
import PrismEncoder
import tempfile
import os
import time
import numpy as np
from IntervalMDPBuilder import IntervalMDPBuilderPacman, IntervalMDPBuilderRandomMDP, IntervalMDPBuilderAirplane, IntervalMDPBuilderSlipperyGridworld, IntervalMDPBuilderWetChicken, IntervalMDPBuilderPrism, IntervalMDPBuilderTaxi, IntervalMDPBuilderFrozenLake
from PrismEncoder import encodeCartPole, encodeCrashingMountainCar
import pycarl
import subprocess
from collections import defaultdict
import re

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
        # self.num_states = 10000 + 1
        # self.num_actions = 2
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
            transition_probs(dict[int, pycarl.Interval]): probability interval
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
        for state in range(self.num_states):
            for action in range(self.num_actions):
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
                    
                    # print(worst_case_transitions)
                    value = 0  
                    for next_state, trans_prob in worst_case_transitions.items():                        
                        value += trans_prob*state_probabilities[next_state]
                    self.shield[state][action] = max(min(1.0, 1-value), 0.0)
        end_total_time = time.time()
        

        print("Total time needed to create the Shield:", end_total_time - start_total_time) 
            
    def calculateShieldPrism(self, prism_txt, prop, export_dir="./shield", java_mem=8):
        """
        Compute per-state-action property satisfaction probabilities using PRISM,
        when the model is provided as a string.

        Args:
            prism_executable (str): path to prism executable
            prism_txt (str): PRISM model contents as a string
            prop (str): property string, e.g. 'Pmin=? [ F<4 "trap" ]'
            export_dir (str): directory to store temporary export files
            java_mem (int): Java memory limit in GB

        Returns:
            dict: nested dictionary { state -> { action -> value } }
        """
        start_total_time = time.time()
        prism_executable="/internship/prism/prism/bin/prism"
        
        os.makedirs(export_dir, exist_ok=True)

        # Write the PRISM model string to a temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".prism", delete=False, dir=export_dir) as tmp_file:
            tmp_file.write(prism_txt)
            model_file = tmp_file.name
        with tempfile.NamedTemporaryFile("w", suffix=".pctl", delete=False, dir=export_dir) as f:
            filter = f"filter(printall, {prop}, true)"
            f.write(filter)
            prop_file = f.name
    
        trans_file = os.path.join(export_dir, "transitions.txt")
        results_file = os.path.join(export_dir, "state_values.txt")

        
        # Run PRISM
        cmd = (
            f"{prism_executable} -javamaxmem {java_mem}g "
            f"{model_file} {prop_file} "
            f"-exporttrans {trans_file} "
            f"| awk '/Results \\(including zeros\\) for filter true:/ {{flag=1; next}} "
            f"flag && /^Range of values/ {{exit}} flag {{print}}' "
            f"> {results_file}"
        )
        
        # print(repr(model_file))
        # print(repr(prop_file))
        # print(repr(trans_file))
        # print(repr(results_file))
        # print("CMD:", cmd)
        # print(repr(cmd))
        subprocess.Popen(cmd, shell=True).wait()

        # Parse state values
        state_values = {}
        pattern = re.compile(r'^(\d+):\([^)]*\)=(\S+)$')
        with open(results_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                match = pattern.match(line)
                if match:
                    state = int(match.group(1))
                    val = float(match.group(2))
                    state_values[state] = val  
                    
                       
        # Parse transitions
        transition_probs = {}
        with open(trans_file, "r") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            
        trans_pattern = re.compile(r'^(\d+)\s+(\d+)\s+(\d+)\s+\[([^\]]+)\]\s+(\S+)$')
        for idx, line in enumerate(lines[1:], start=2):
            m = trans_pattern.match(line)
            if not m:
                print(f"Warning: could not parse transition line #{idx}: {repr(line)}")
                continue

            s = int(m.group(1))
            a = int(m.group(2))
            s_next = int(m.group(3))
            bounds = m.group(4).split(",")
            try:
                lower, upper = float(bounds[0]), float(bounds[1])
            except Exception:
                print(f"Warning: bad bounds on line #{idx}: {repr(bounds)}")
                continue

            transition_probs.setdefault((s, a), {})[s_next] = {"lower": lower, "upper": upper}
        
        # Fill self.shield with computed state-action values
        for state in range(self.num_states):
            for action in range(self.num_actions):
                trans = transition_probs[(state, action)]
                worst_case_transitions = {}
                for next_state, trans_prob in trans.items():
                    worst_case_transitions[next_state] = trans_prob["lower"]
                remaining_states = list(worst_case_transitions.keys())
                total_mass = sum(worst_case_transitions.values())
                
                #get the worst next state and add mass untill we reach upper
                while total_mass < 1 and len(remaining_states) >= 1:
                    # get the worst next state
                    worst_next_state = min(remaining_states, key=lambda i: state_values[i])
                    remaining_states = [state for state in remaining_states if state != worst_next_state]
                    # add mass untill we reach max_prob or total_mass = 1
                    bounds = trans[worst_next_state]
                    difference = bounds["upper"]-bounds["lower"]
                    added_mass = min(difference, 1-total_mass)
                    total_mass+= added_mass
                    worst_case_transitions[worst_next_state] = worst_case_transitions[worst_next_state]+added_mass
                
                value = 0  
                for next_state, trans_prob in worst_case_transitions.items():                        
                    value += trans_prob*state_values[next_state]
                self.shield[state][action] = max(min(1.0, 1-value), 0.0)
        # print(self.shield)
        # Clean up temporary files
        os.remove(model_file)  
        os.remove(prop_file)   
        os.remove(trans_file)
        os.remove(results_file)   
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
    def __init__(self, transition_matrix, traps, goal, intervals, init):
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
        # self.model_builder = IntervalMDPBuilderPrism(transition_matrix, intervals, goal, traps) 
        self.prism_text = encodeCartPole(transition_matrix, intervals, init)
        super().__init__(transition_matrix, traps, goal, intervals)
        
    def calculateShield(self):
        """
        calculate the probability of violating the safety specification for the Random MDPs environment
        """
        # How likely are we to step into a trap
        # prop = "Pmax=? [!\"trap\" U \"goal\"]"
        # prop = "Pmax=? [!\"goal\"U\"trap\"]"
        prop = "Pmax=? [  F<4 \"trap\" ]"
        # prop1 = "Pmax=? [  F \"trap\" ]"
        
        # Is it possible to reach the goal
        # prop2 = "Pmax=? [F \"goal\" & !F<4 \"trap\"]"
        # prop2 = "Pmax=? [F \"reach\" & !F \"trap\"]"
        # prop3 = "Pmin=? [F<=5 \"reach\"]"
        # super().calculateShieldInterval(prop, self.model_builder.build_model())
        super().calculateShieldPrism(self.prism_text, prop)
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
        
        
class ShieldCrashingMountainCar(Shield):
    # Calculate the shield for the Random MDPs environment
    def __init__(self, transition_matrix, traps, goal, intervals, init):
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
        # self.model_builder = IntervalMDPBuilderPrism(transition_matrix, intervals, goal, traps) 
        self.prism_text = encodeCrashingMountainCar(transition_matrix, intervals, init)
        super().__init__(transition_matrix, traps, goal, intervals)
        
    def calculateShield(self):
        """
        calculate the probability of violating the safety specification for the Random MDPs environment
        """
        # How likely are we to step into a trap
        prop = "Pmax=? [  F<4 \"trap\" ]"
        # prop = "Pmax=? [!\"trap\" U \"goal\"]"
        
        # Is it possible to reach the goal
        # prop2 = "Pmax=? [F \"goal\" & !F<4 \"trap\"]"
        # prop2 = "Pmax=? [F \"reach\" & !F \"trap\"]"
        # prop3 = "Pmin=? [F<=5 \"reach\"]"
        # super().calculateShieldInterval(prop, self.model_builder.build_model())
        super().calculateShieldPrism(self.prism_text, prop)
        self.shield = 1-self.shield
        
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
        # state_action_prob_pairs = [i for i in state_action_prob_pairs if i[2] < 0.95 and i[2]> 0]
        for pair in state_action_prob_pairs:
            state = pair[0]
            action = pair[1]
            prob = pair[2]
                        
            print(f"State: {state}, action: {action}, with probability of falling: {prob}")
        print(f"with crash states being {self.traps} and success states being {self.goal}")


        

    
    
    
    
