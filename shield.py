import stormpy
import PrismEncoder
import tempfile
import os
import time
import numpy as np
from IntervalMDPBuilder import IntervalMDPBuilder
from IntervalMDPBuilder import IntervalMDPBuilderRandomMDPs, IntervalMDPBuilderAirplane, IntervalMDPBuilderWetChicken, IntervalMDPBuilderSlipperyGridworld,  IntervalMDPBuilderPacman

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
        
    
    def calculateShieldInterval2(self, prop, model_function):
        start_total_time = time.time()
        for state in range(len(self.structure)):
            for action in range(len(self.structure[state])):
                # print(f"shield for state {state} and action {action}")
                if state in self.traps:
                    self.shield[state][action] = 1
                elif state in self.goal:
                    self.shield[state][action] = 0
                else:
                    next_states = self.structure[state][action]
                    model = model_function(state, action, next_states)
                    r1 = self.invokeStorm(model, prop)
                    self.shield[state][action] = 1-r1
                    # raise("hold on a god damn second")
                # time.sleep(100)
        end_total_time = time.time()
        

        print("Total time needed to create the Shield:", end_total_time - start_total_time)   
    def calculateShieldInterval(self, prop, model_function):
        
        start_total_time = time.time()
        self.use_hint = False
        for state in range(len(self.structure)):
            for action in range(len(self.structure[state])):
                # print(f"shield for state {state} and action {action}")
                if state in self.traps:
                    self.shield[state][action] = 1
                elif state in self.goal:
                    self.shield[state][action] = 0
                else:
                    next_states = self.structure[state][action]
                    model = model_function(state, action, next_states)
                    r1 = self.invokeStorm(model, prop)
                    self.shield[state][action] = 1-r1
                    # raise("hold on a god damn second")
                # time.sleep(100)
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
        if self.use_hint:
            print("DABABY")
            hint = stormpy.ExplicitModelCheckerHintDouble().set_scheduler_hint(self.scheduler)
            task.set_hint(hint)
        task.set_produce_schedulers()
        task.set_robust_uncertainty(True)
        result = stormpy.check_interval_mdp(model, task, env)
        if not self.use_hint:
            self.scheduler = result.scheduler
            # self.use_hint = True
        # print(self.scheduler)
        initial_state = model.initial_states[0]
        
        # for state in model.states:
        #     for action in state.actions:
        #         for transition in action.transitions:
        #             print("From state {} by action {}, with probability {}, go to state {}".format(state, action, transition.value(), transition.column))
        
        return result.at(initial_state)


class ShieldRandomMDP(Shield):
    def __init__(self, transition_matrix, traps, goal, intervals):
        self.builder = IntervalMDPBuilderRandomMDPs(transition_matrix, intervals, goal, traps)
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
        return super().calculateShieldInterval(prop3, self.builder.build_model_with_init)
    
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
        self.builder = IntervalMDPBuilderWetChicken(transition_matrix, intervals, [], [])
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
        return super().calculateShieldInterval(prop, self.builder.build_model_with_init)
    
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
        self.builder = IntervalMDPBuilderAirplane(transition_matrix, intervals, goal, traps)
        super().__init__(transition_matrix, traps, goal, intervals)
        
    def calculateShield(self):
        # How likely are we to step into a trap
        prop = "Pmax=? [  !\"crash\" U \"success\"]"
        # prop = "Pmin=? [  F\"waterfall\"]"
        # prop = "Pmax=? [  !F<2\"waterfall\"]"
        return super().calculateShieldInterval(prop, self.builder.build_model_with_init)
    
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
        self.builder = IntervalMDPBuilderSlipperyGridworld(transition_matrix, intervals, goal, traps)
        super().__init__(transition_matrix, traps, goal, intervals)
    
    def calculateShield(self):
        # How likely are we to step into a trap
        # prop = "Pmax=? [!\"trap\"U\"goal\"]"
        prop = "Pmax=? [!\"trap\"U\"save\"]"
        prop = "Pmax=? [!\"save\"U\"trap\"]"
        
        super().calculateShieldInterval(prop, self.builder.build_model_with_init)
        self.shield = 1-self.shield
    
    def get_safe_actions_from_shield(self, state, threshold=0.0, buffer = 0.15):
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
    def __init__(self, transition_matrix, traps, goal, intervals, width, height, walls):
        self.walls = walls
        self.width = width
        self.height = height
        self.builder = IntervalMDPBuilderPacman(transition_matrix, intervals, goal, traps)
        super().__init__(transition_matrix, traps, goal, intervals)
        
    def calculateShieldInterval(self, prop, model_function):
        start_total_time = time.time()
        self.use_hint = False
        for state in range(len(self.structure)):
            if state in self.traps or state in self.walls:
                self.shield[state][:] = 1
            elif state in self.goal:
                self.shield[state][:] = 0
            else:
                for action in range(len(self.structure[state])):
                    # print(f"shield for state {state} and action {action}")
                        next_states = self.structure[state][action]
                        model = model_function(state, action, next_states)
                        r1 = self.invokeStorm(model, prop)
                        self.shield[state][action] = 1-r1
                        # raise("hold on a god damn second")
                    # time.sleep(100)
        end_total_time = time.time()
        

        print("Total time needed to create the Shield:", end_total_time - start_total_time) 
    def calculateShield(self):
        # How likely are we to step into a trap
        prop = "Pmax=? [!\"eaten\"U\"goal\"]"
        
        super().calculateShieldInterval(prop, self.builder.build_model_with_init)
    
    def decode_int(self, state_int):
        ay = state_int % self.height
        state_int //= self.height

        ax = state_int % self.width
        state_int //= self.width

        y = state_int % self.height
        state_int //= self.height

        x = state_int % self.width

        return x, y, ax, ay 
    def get_safe_actions_from_shield(self, state, threshold=0.0, buffer = 0.00000001):
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
            x, y, gx, gy = self.decode_int(state)
            print(f"Trap State: {state}: a:({x}, {y}), ghost:({gx},{gy}) action: {action}, with probability of getting trapped: {prob}")
        
        print(f"with crash states being {self.traps} and success states being {self.goal}")

    
    
    
    
