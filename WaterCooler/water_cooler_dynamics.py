import numpy as np


INFLOW_PROB = [0.1, 0.6, 0.3]
OUTFLOW_PROB = [0.4,0.5,0.1]
class waterCooler:
    def __init__(self, capacity):
        self.capacity = capacity
        self.nb_states = self.capacity
        self.init = int(self.capacity/2)
        
    