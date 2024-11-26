import stormpy
import PrismEncoder
import tempfile
import os
import time
import numpy as np

class Shield:
    def __init__(self, transition_matrix, traps, goal):
        self.traps = traps
        self.goal = goal
        self.structure = transition_matrix
        self.prismStr = self.createPrismStr()
        with open("output.txt", 'w') as file:
            file.write(self.prismStr)
        self.shield = []

        
    def createPrismStr(self):
        Tempstr = PrismEncoder.encodeMDP(self.structure)
        Tempstr = PrismEncoder.add_reach_label(Tempstr, self.goal)
        Tempstr = PrismEncoder.add_avoid_label(Tempstr, self.traps)
        return Tempstr
        
    # note: this now only works for simple one variable states. Fix this if we want to work with more complex environments
    def calculateShield(self):    
        start_total_time = time.time()
        for state in range(len(self.structure)):
            for action in range(len(self.structure[state])):
                # print(self.traps)
                # print(state)
                if state in self.traps:
                    self.shield.append([state, action, 1])
                if state in self.goal:
                    self.shield.append([state, action, 0])
                else:
                    # How likely are we to step into a trap in the worst case scenario
                    prop1 = "Pmin=? [  F<4 \"trap\" ]"
                    # prop1 = "Pmax=? [  F \"trap\" ]"
                    
                    # Is it possible to reach the goal
                    prop2 = "Pmax=? [F \"reach\" & !F<3 \"trap\"]"
                    
                    mdpprog = PrismEncoder.add_initial_state_to_prism_mdp(self.prismStr, -1, action, self.structure[state][action])
                    r1 = self.invokeStorm(mdpprog, prop1)
                    # r2 = self.invokeStorm(mdpprog, prop2)
                    r2 = 1
                    self.shield.append([state, action, r1 + 1-r2])
                    
        end_total_time = time.time()
        
        self.shield = np.array(self.shield)
        sorted_arr = self.shield[self.shield[:, 2].argsort()]
        np.set_printoptions(suppress=True, precision=6)
        print(sorted_arr)
        print(self.traps)
        print("Total time needed to create the Shield:", end_total_time - start_total_time)
        
    def invokeStorm(self, mdpprog, prop):
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



# get transition matrix as input
# Go through all state-action pairs and generate a corresponding MDP
# Check the MDP
# Store the probabilities in a lookup table
