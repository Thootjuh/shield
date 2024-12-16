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
        self.shield = []
    
        
    def createPrismStr(self):
        Tempstr = PrismEncoder.encodeMDP(self.structure)
        Tempstr = PrismEncoder.add_reach_label(Tempstr, self.goal)
        Tempstr = PrismEncoder.add_avoid_label(Tempstr, self.traps)
        with open("output.txt", 'w') as file:
            file.write(Tempstr)
        return Tempstr
        
    # note: this now only works for simple one variable states. Fix this if we want to work with more complex environments
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
                # print(self.traps)
                # print(state)
                if state in self.traps:
                    self.shield.append([state, action, 1])
                elif state in self.goal:
                    self.shield.append([state, action, 0])
                else:
                    next_states = self.structure[state][action]
                    model = model_function(state, action, next_states)
                    r1 = self.invokeStorm(model, prop)
                    self.shield.append([state, action, 1-r1])
        end_total_time = time.time()
        self.shield = np.array(self.shield)
        self.printShield()

        print("Total time needed to create the Shield:", end_total_time - start_total_time)                                 
        
    def printShield(self):
        sorted_arr = self.shield[self.shield[:, 2].argsort()]
        np.set_printoptions(suppress=True, precision=6)
        print(sorted_arr)
        print(self.traps)
        
    def invokeStormPrism(self, mdpprog, prop):
        # write program to RAM
        # print("writing prism program to RAM")
        temp_name = next(tempfile._get_candidate_names())
        file_name = "/dev/shm/prism-" + temp_name + ".nm"
        text_file = open(file_name, "w")
        text_file.write(mdpprog)
        text_file.close()

        # read program from RAM
        # print("parse prism program from RAM")
        program = stormpy.parse_prism_program(file_name)

        # print("parse properties")
        # prop = "Pmin=? [ F<=" + str((self.num_ghosts+1)*STEPS-1) +" \"crash\" ]"
        
        # probability of falling into a trap

        # prop = "Pmax=? [ (!F \"trap\" & F \"reach\") ]"
        
        # prop = "Pmax=? [F \"reach\"]"
        # prop = "Pmax=? [G !F<3 \"trap\ U \"reach\"]"
        # Find something for reach and combine them
        properties = stormpy.parse_properties_for_prism_program(prop, program, None)

        # print("Build Model")
        start = time.time()
        model = stormpy.build_model(program, properties)
        initial_state = model.initial_states[0]
        end = time.time()
        # print(end-start)

        result = stormpy.model_checking(model, properties[0])
        # print(result.at(initial_state))

        os.remove(file_name)

        return result.at(initial_state)
    
    def get_safe_actions_from_shield(self, state, threshold):
        print(self.shield)
    
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

class ShieldWetChicken(Shield):
    def __init__(self, transition_matrix, width, height, traps, intervals):
        self.traps = traps
        self.goal = []
        self.width = width
        self.height = height
        self.structure = transition_matrix
        self.intervals = intervals
        self.builder = IntervalMDPBuilder(transition_matrix, intervals, self.goal, self.traps)
        self.shield = []
        
    def createPrismStr(self):
        Tempstr = PrismEncoder.encodeWetChicken(self.structure, self.width, self.height)
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
        sorted_arr = self.shield[self.shield[:, 2].argsort()]
        np.set_printoptions(suppress=True, precision=6)
        for row in sorted_arr:
            state = row[0]
            action = actions[int(row[1])]
            prob = row[2]
            x = int(state / self.height)
            y = int(state % self.width)
            print(f"State: ({x}, {y}), action: {action}, with probability of falling: {prob}")
            
        print("with falling states being")
        for state in self.traps:
            x = int(state / self.height)
            y = int(state % self.width)
            print(f"({x},{y})")
            
    def calculateShield(self):
        # How likely are we to step into a trap
        prop = "Pmin=? [  F<4\"waterfall\" ]"
        return super().calculateShieldInterval(prop, self.builder.build_wetChicken_model_with_init)
    
    
    
