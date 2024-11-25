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
                    mdpprog = PrismEncoder.add_initial_state_to_prism_mdp(self.prismStr, -1, action, self.structure[state][action])
                    result = self.invokeStorm(mdpprog)
                    self.shield.append([state, action, result])
                    
        end_total_time = time.time()
        print("Total time needed to create the Shield:", end_total_time - start_total_time)
        print(self.shield)
        
    def invokeStorm(self, mdpprog):
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
        # prop = "Pmin=? [ F \"trap\" ]"
        prop = "Pmax=? [ (!F \"trap\" & F \"reach\") ]"
        
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
