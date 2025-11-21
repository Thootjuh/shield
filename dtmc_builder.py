import stormpy
from collections import defaultdict

class dtmc_builder:
    def __init__(self, P, pi, n_states, n_actions, init, goal, traps):
        self.init = init
        self.n_states = n_states
        self.n_actions = n_actions
        self.transition_dynamics = P
        self.policy = pi
        self.goal = goal
        self.traps = traps
        self.reshape_P()

    def reshape_P(self):
        self.transitions_reshaped = defaultdict(list)
        for (s, a, next_s), p in self.transition_dynamics.items():
            self.transitions_reshaped[(s, a)].append((next_s, p)) 
            
    def build_model(self):
                # Use the known system dynamics and generated policy to construct a DTMC 
        dtmc_builder = stormpy.SparseMatrixBuilder(
            rows=self.n_states,
            columns=self.n_states,
            entries=0
        )

        for state in range(self.n_states):
            probs = defaultdict(float)
            for action in range(self.n_actions):
                action_prob = self.policy[state][action]
                if action_prob == 0:
                    continue
                for next_s, trans_prob in self.transitions_reshaped[(state, action)]:
                    probs[next_s] += action_prob * trans_prob

            for next_state, prob in probs.items():
                dtmc_builder.add_next_value(state, next_state, prob)

        self.dtmc_matrix = dtmc_builder.build()    

        components = stormpy.SparseModelComponents(transition_matrix=self.dtmc_matrix, state_labeling=self.set_state_labels(), reward_models=self.set_reward_model_MDP())
        return stormpy.storage.SparseDtmc(components)
    
    
    def set_reward_model_MDP(self):
        reward_models = {}
        reward_models["rewards"] = stormpy.SparseRewardModel()
        return reward_models    
    
class dtmcBuilderWetChicken(dtmc_builder):
    def set_state_labels(self):
        state_labeling = stormpy.storage.StateLabeling(self.n_states)
        labels = {"waterfall", "init", "goal"}
        for label in labels:
            state_labeling.add_label(label)
        for i in self.goal:
            state_labeling.add_label_to_state("goal", i)
        state_labeling.add_label_to_state("waterfall", self.n_states-1)
        
        state_labeling.add_label_to_state("init", 0)
        
        return state_labeling
    
class dtmcBuilderRandomMDPs(dtmc_builder):
    def set_state_labels(self):
        state_labeling = stormpy.storage.StateLabeling(self.n_states)
        labels = {"goal", "trap", "init"}
        for label in labels:
            state_labeling.add_label(label)
        for i in self.traps:
            state_labeling.add_label_to_state("trap", i)
        for i in self.goal:
            state_labeling.add_label_to_state("goal", i)
        state_labeling.add_label_to_state("init", 0)
        
        return state_labeling
    
class dtmcBuilderFrozenLake(dtmc_builder):
    def set_state_labels(self):
        state_labeling = stormpy.storage.StateLabeling(self.n_states)
        labels = {"goal", "hole", "init"}
        for label in labels:
            state_labeling.add_label(label)
        for state in range(self.n_states):
            if state in self.traps:
                state_labeling.add_label_to_state("hole", state)
            if state in self.goal:
                state_labeling.add_label_to_state("goal", state)
        state_labeling.add_label_to_state("init", 0)
        
        return state_labeling    
    
class dtmcBuilderPacMan(dtmc_builder):
    def set_state_labels(self):
        state_labeling = stormpy.storage.StateLabeling(self.n_states)
        labels = {"goal", "eaten", "maze"}
        for label in labels:
            state_labeling.add_label(label)
        for state in range(self.n_states):
            if state in self.traps:
                state_labeling.add_label_to_state("eaten", state)
            elif state in self.goal:
                state_labeling.add_label_to_state("goal", state)
            else:
                state_labeling.add_label_to_state("maze", state)
        return state_labeling
