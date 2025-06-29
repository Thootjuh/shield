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
        self.traps = traps
        self.goal = goal
        self.structure = transition_matrix
        self.intervals = intervals
        self.num_states= len(self.structure)
        self.num_actions = len(self.structure[0])
        self.shield = np.full((self.num_states, self.num_actions), -1, dtype=np.float64)
    
        
    def createPrismStr(self):
        Tempstr = PrismEncoder.encodeMDP(self.structure)
        Tempstr = PrismEncoder.add_reach_label(Tempstr, self.goal)
        Tempstr = PrismEncoder.add_avoid_label(Tempstr, self.traps)
        with open("output.txt", 'w') as file:
            file.write(Tempstr)
        return Tempstr
    
    def get_probs(self, model, prop):
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
        sorted_arr = self.shield[self.shield[:, 2].argsort()]
        np.set_printoptions(suppress=True, precision=6)
        print(sorted_arr)
        print(self.traps)
                
    
    def invokeStorm(self, model, prop):
        properties = stormpy.parse_properties(prop)
        env = stormpy.Environment()
        env.solver_environment.minmax_solver_environment.method = stormpy.MinMaxMethod.value_iteration
        task = stormpy.CheckTask(properties[0].raw_formula, only_initial_states=True)
        task.set_produce_schedulers()
        task.set_robust_uncertainty(True)
        result = stormpy.check_interval_mdp(model, task, env)
        initial_state = model.initial_states[0]
        
        return result.at(initial_state)


class ShieldRandomMDP(Shield):
    def __init__(self, transition_matrix, traps, goal, intervals):
        self.model_builder = IntervalMDPBuilderRandomMDP(transition_matrix, intervals, goal, traps)
        super().__init__(transition_matrix, traps, goal, intervals)
        
    def calculateShield(self):
        # How likely are we to step into a trap
        prop = "Pmin=? [  F<4 \"trap\" ]"
        # prop1 = "Pmax=? [  F \"trap\" ]"
        
        # Is it possible to reach the goal
        prop2 = "Pmax=? [F \"goal\" & !F<4 \"trap\"]"
        # prop2 = "Pmax=? [F \"reach\" & !F \"trap\"]"
        prop3 = "Pmax=? [!\"trap\" U \"goal\"]"
        # prop3 = "Pmin=? [F<=5 \"reach\"]"
        super().calculateShieldInterval(prop3, self.model_builder.build_model())
    def get_safe_actions_from_shield(self, state, threshold=0.2, buffer = 0.05):
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
    def __init__(self, transition_matrix, width, length, goals, intervals):
        self.width = width
        self.length = length
        self.model_builder = IntervalMDPBuilderWetChicken(transition_matrix, intervals, [], [])
        super().__init__(transition_matrix, [], [], intervals)
        
    def createPrismStr(self):
        Tempstr = PrismEncoder.encodeWetChicken(self.structure, self.width, self.length)
        Tempstr = PrismEncoder.add_avoid_label(Tempstr, self.traps)
        with open("output.txt", 'w') as file:
            file.write(Tempstr)
        return Tempstr
    
    def printShield(self):
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
        # How likely are we to step into a trap
        prop = "Pmax=? [  !\"waterfall\" U \"goal\"]"
        # prop = "Pmin=? [  F\"waterfall\"]"
        # prop = "Pmax=? [  !F<2\"waterfall\"]"
        super().calculateShieldInterval(prop, self.model_builder.build_model())
    def get_safe_actions_from_shield(self, state, threshold=0.2, buffer = 0.05):
        probs = self.shield[state]
        safe_actions = []
        for i, prob in enumerate(probs):
            if prob >= 0 and prob <= threshold:
                safe_actions.append(i)

        if len(safe_actions) == 0:
            min_value = np.min(probs)
            safe_actions = np.where(probs <= min_value+buffer)[0].tolist()
        return safe_actions
    
class ShieldAirplane(Shield):
    def __init__(self, transition_matrix, traps, goal, intervals, maxX, maxY):
        self.maxX = maxX
        self.maxY = maxY
        self.model_builder = IntervalMDPBuilderAirplane(transition_matrix, intervals, goal, traps)
        super().__init__(transition_matrix, traps, goal, intervals)
        
    def calculateShield(self):
        # How likely are we to step into a trap
        prop = "Pmax=? [  !\"crash\" U \"success\"]"
        # prop = "Pmin=? [  F\"waterfall\"]"
        # prop = "Pmax=? [  !F<2\"waterfall\"]"
        super().calculateShieldInterval(prop, self.model_builder.build_model())
    def decode_int(self, state_int):   
        ay = state_int % self.maxY
        state_int //= self.maxY
        
        y = state_int % self.maxY
        state_int //= self.maxY
        
        x = state_int % self.maxX
        
        return x, y, ay
    
    def printShield(self):
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
    def __init__(self, transition_matrix, traps, goal, intervals, width, height):
        self.width = width
        self.height = height
        self.model_builder = IntervalMDPBuilderSlipperyGridworld(transition_matrix, intervals, goal, traps)
        super().__init__(transition_matrix, traps, goal, intervals)
    
    def calculateShield(self):
        # How likely are we to step into a trap
        prop = "Pmax=? [!\"trap\"U\"goal\"]"
        # prop = "Pmax=? [!\"trap\"U\"save\"]"
        # prop = "Pmax=? [!\"save\"U\"trap\"]"
        
        super().calculateShieldInterval(prop, self.model_builder.build_model())
        # self.printShield()
        # self.shield = 1-self.shield
    
    def get_safe_actions_from_shield(self, state, threshold=0.1, buffer = 0.1):
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
        x = number // (self.width)
        y = number % (self.width)
        
        return x, y
    
    def printShield(self):
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
    def __init__(self, transition_matrix, traps, goal, intervals, width, height):
        self.width = width
        self.height = height
        self.model_builder = IntervalMDPBuilderPacman(transition_matrix, intervals, goal, traps)
        super().__init__(transition_matrix, traps, goal, intervals)
        

    def calculateShield(self):
        # How likely are we to step into a trap
        prop = "Pmax=? [!\"eaten\"U\"goal\"]"
        
        super().calculateShieldInterval(prop, self.model_builder.build_model())
    
    def decode_int(self, state_int):
        ay = state_int % self.height
        state_int //= self.height

        ax = state_int % self.width
        state_int //= self.width

        y = state_int % self.height
        state_int //= self.height

        x = state_int % self.width

        return x, y, ax, ay
    
    def get_safe_actions_from_shield(self, state, threshold=0.01, buffer = 0.05):
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
                print(f"Trap State: {state}: a:({x}, {y}), ghost:({gx1},{gy1}) action: {action}, with probability of getting trapped: {prob}")
        print(f"with crash states being {self.traps} and success states being {self.goal}")
        
class ShieldTaxi(Shield):
    def __init__(self, transition_matrix, traps, goal, intervals, init=0):
        self.model_builder = IntervalMDPBuilderTaxi(transition_matrix, intervals, goal, traps)
        self.init = init
        super().__init__(transition_matrix, traps, goal, intervals)
        
   
    def calculateShield(self):
        # How likely are we to step into a trap
        prop = "Pmax=? [!\"crash\"U\"goal\"]"
    
        # self.printShield()
        # self.shield[:] = 0
        super().calculateShieldInterval(prop, self.model_builder.build_model())
        # self.printShield()
        # self.printShield()
    
    def get_safe_actions_from_shield(self, state, threshold=0.3, buffer = 0.05):
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

    def __init__(self, transition_matrix, traps, goal, intervals):
        self.model_builder = IntervalMDPBuilderFrozenLake(transition_matrix, intervals, goal, traps)
        super().__init__(transition_matrix, traps, goal, intervals)
        self.grid_size = np.sqrt(self.num_states)
        
    def calculateShield(self):
        # How likely are we to step into a trap
        prop = "Pmax=? [!\"hole\"U\"goal\"]"
        
        # self.printShield()
        # self.shield[:] = 0
        super().calculateShieldInterval(prop, self.model_builder.build_model())
        # self.printShield()
        # self.printShield()
    
    def get_safe_actions_from_shield(self, state, threshold=0.2, buffer = 0.02):
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
        col = int(state % self.grid_size)
        row = int(state // self.grid_size)
        return col, row
    def printShield(self):
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
        self.model_builder = IntervalMDPBuilderPrism(transition_matrix, intervals, goal, traps)
        super().__init__(transition_matrix, traps, goal, intervals)
        
   
    def calculateShield(self):
        # How likely are we to step into a trap
        prop = "Pmax=? [!\"trap\"U\"goal\"]"
        
        # self.printShield()
        # self.shield[:] = 0
        super().calculateShieldInterval(prop, self.model_builder.build_model())
        # self.printShield()
        # self.printShield()
    
    def get_safe_actions_from_shield(self, state, threshold=0.2, buffer = 0.05):
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
        

    
    
    
    
