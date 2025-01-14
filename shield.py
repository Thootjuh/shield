import stormpy
import PrismEncoder
import tempfile
import os
import time
import numpy as np
from IntervalMDPBuilder import IntervalMDPBuilder
import pycarl
 
class Shield:
    def __init__(self, transition_matrix, traps, goal, intervals):
        self.traps = traps
        self.goal = goal
        self.structure = transition_matrix
        self.intervals = intervals
        self.builder = IntervalMDPBuilder(transition_matrix, intervals, goal, traps)
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
        
    def calculateShieldPrism(self, prop):    
        start_total_time = time.time()
        for state in range(len(self.structure)):
            for action in range(len(self.structure[state])):
                # print(self.traps)
                # print(state)
                if state in self.traps:
                    self.shield.append([state, action, 1])
                elif state in self.goal:
                    self.shield.append([state, action, 0])
                else:
                    mdpprog = PrismEncoder.add_initial_state_to_prism_mdp(self.prismStr, -1, action, self.structure[state][action])
                    r1 = self.invokeStormPrism(mdpprog, prop)
                    # r2 = self.invokeStorm(mdpprog, prop2)
                    # r2 = 1
                    self.shield.append([state, action, r1])
                    
        end_total_time = time.time()
        
        self.shield = np.array(self.shield)
        # self.printShield()

        print("Total time needed to create the Shield:", end_total_time - start_total_time)
    
    def calculateShieldInterval(self, prop, model_function):
        start_total_time = time.time()
        for state in range(len(self.structure)):
            for action in range(len(self.structure[state])):
                if state in self.traps:
                    self.shield[state][action] = 1
                elif state in self.goal:
                    self.shield[state][action] = 0
                else:
                    next_states = self.structure[state][action]
                    model = model_function(state, action, next_states)
                    r1 = self.invokeStorm(model, prop)
                    self.shield[state][action] = 1-r1
                # time.sleep(100)
        end_total_time = time.time()
        

        print("Total time needed to create the Shield:", end_total_time - start_total_time)                                 
        
    def printShield(self):
        sorted_arr = self.shield2[self.shield2[:, 2].argsort()]
        np.set_printoptions(suppress=True, precision=6)
        print(sorted_arr)
        print(self.traps)
        
    def invokeStormPrism(self, mdpprog, prop):
        # write program to RAM
        temp_name = next(tempfile._get_candidate_names())
        file_name = "/dev/shm/prism-" + temp_name + ".nm"
        text_file = open(file_name, "w")
        text_file.write(mdpprog)
        text_file.close()

        # read program from RAM
        program = stormpy.parse_prism_program(file_name)
        properties = stormpy.parse_properties_for_prism_program(prop, program, None)

        start = time.time()
        model = stormpy.build_model(program, properties)
        initial_state = model.initial_states[0]
        end = time.time()
        # print(end-start)

        result = stormpy.model_checking(model, properties[0])
        # print(result.at(initial_state))

        os.remove(file_name)

        return result.at(initial_state)
                
    
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
    def calculateShield(self):
        # How likely are we to step into a trap
        prop = "Pmin=? [  F<4 \"trap\" ]"
        # prop1 = "Pmax=? [  F \"trap\" ]"
        
        # Is it possible to reach the goal
        prop2 = "Pmax=? [F \"goal\" & !F<4 \"trap\"]"
        # prop2 = "Pmax=? [F \"reach\" & !F \"trap\"]"
        prop3 = "Pmax=? [!\"trap\" U \"goal\"]"
        # prop3 = "Pmin=? [F<=5 \"reach\"]"
        return super().calculateShieldInterval(prop3, self.builder.build_MDP_model_with_init)
    
    def get_safe_actions_from_shield(self, state, threshold=0.2):
        probs = self.shield[state]
        safe_actions = []
        for i, prob in enumerate(probs):
            if prob >= 0 and prob <= threshold:
                safe_actions.append(i)

        if len(safe_actions) == 0:
            min_value = np.min(probs)
            safe_actions = np.where(probs == min_value)[0].tolist()
        return safe_actions

class ShieldWetChicken(Shield):
    def __init__(self, transition_matrix, width, length, goals, intervals):
        self.width = width
        self.length = length
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
        return super().calculateShieldInterval(prop, self.builder.build_wetChicken_model_with_init)
    
    def get_safe_actions_from_shield(self, state, threshold=0.2):
        probs = self.shield[state]
        safe_actions = []
        for i, prob in enumerate(probs):
            if prob >= 0 and prob <= threshold:
                safe_actions.append(i)

        if len(safe_actions) == 0:
            min_value = np.min(probs)
            safe_actions = np.where(probs == min_value)[0].tolist()
        return safe_actions
    
    
    
