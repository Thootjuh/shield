# This file will contain code related to learning an IMDP from a given partition and the data
# Modified from: https://github.com/LAVA-LAB/DynAbs

import numpy as np
import pandas as pd
from collections import defaultdict
import os
from pathlib import Path

from scipy.stats import beta as betaF

class imdp_builder:
    def __init__(self, trajectories, count_state_action_state, count_state_action, episodic, kstep, beta):
        self.data = trajectories
        self.episodic = episodic
        self.count_state_action_state = count_state_action_state
        self.count_state_action = count_state_action
        self.compute_transition_intervals(beta = beta, kstep = kstep)
    
    
            
    def createUniformSamples(self, N, low=-1, upp=1):
        
        if N > 1:
            rands = low + (upp-low)*np.random.rand(N)
        else:
            rands = low + (upp-low)*np.random.rand()
        
        return rands

    def computeBetaPPF(self, N, k, d, beta):
        
        epsilon = betaF.ppf(beta, k+d, N-(d+k)+1)
        
        return epsilon

    def computeBetaCDF(self, N, k, d, epsilon):
        
        cum_prob = betaF.cdf(epsilon, k+d, N-(d+k)+1)
        
        return cum_prob

    def validate_eps(self, trials, N, beta, d, krange, eps_low, eps_upp):
        
        correct_eps_sum_low     = np.zeros(len(krange))
        correct_eps_sum_upp     = np.zeros(len(krange))
        correct_eps_sum_both    = np.zeros(len(krange))
        correct_all             = 0
        
        for tr in range(self, trials):
        
            if tr % 1000 == 0:
                print('Trial number',tr)
            
            fac = 1e-6
            
            width = self.trial_SAD(N, beta, d, krange, fac)
        
            # Validate computed epsilon
            V_prob = np.zeros(len(width))
            for i,w in enumerate(width):
                V_prob[i] = 1 - w  
        
            # Check if bounds are correct
            correct_eps_sum_low += V_prob > eps_low
            correct_eps_sum_upp += V_prob < eps_upp
            correct_eps_sum_both += (V_prob > eps_low) * (V_prob < eps_upp)
            
            correct_all += all(V_prob > eps_low) and all(V_prob < eps_upp)
        
            beta_empirical_low      = correct_eps_sum_low / trials
            beta_empirical_upp      = correct_eps_sum_upp / trials
            beta_empirical_both     = correct_eps_sum_both / trials
            
            beta_overall            = correct_all / trials
            
        print('Beta empirical low:',    beta_empirical_low)
        print('Beta empirical upp:',    beta_empirical_upp)
        print('Beta empirical both:',   beta_empirical_both)
        
        print('Overall confidence level:', beta_overall,'(expected was: '+str(1-beta)+')')

    def trial_SAD(self, N, beta, d, krange, fac):
        
        # Create interval for N samples
        samples = self.createUniformSamples(N, 0,1)
        
        # Create interval for N samples
        samples_sort = np.sort(samples)
        
        # For every value of samples to discard
        width = np.array([ np.max(samples_sort[:int(N-k)]) for i,k in enumerate(krange)])
        
        return width
    
    def compute_interval_for_Nout(self, N, N_out, beta=1e-6, kstep=1):
        """
        Compute probability interval for a single value of N_out,
        instead of generating the full table.

        Parameters
        ----------
        N : int
            Total number of samples for (s,a).
        N_out : int
            Number of samples outside the transition of interest.
        beta : float
            Confidence parameter.
        kstep : int
            Granularity of discarded samples.

        Returns
        -------
        (float, float)
            Lower and upper bounds for the probability interval.
        """
        d = 1
        krange = np.arange(0, N, kstep)
        beta_bar = beta / (2 * len(krange))

        # Precompute epsilons for krange
        eps_low = [self.computeBetaPPF(N, k, d, beta_bar) for k in krange]
        eps_upp = [self.computeBetaPPF(N, k, d, 1 - beta_bar) for k in krange]

        # Case 1: N_out is larger than max(krange)
        if N_out > np.max(krange):
            lower_bound = 0
            upper_bound = 1 - eps_low[-1]
        else:
            id_in = (N_out - 1) // kstep + 1 if N_out > 0 else 0

            lower_bound = 1 - eps_upp[id_in]

            if N_out == 0:
                upper_bound = 1
            else:
                id_out = id_in - 1
                upper_bound = 1 - eps_low[id_out]

            # Sanity check for monotonicity
            if upper_bound < lower_bound:
                upper_bound = lower_bound

        return lower_bound, upper_bound

    def create_table(self, N, beta, kstep, trials, export=False):

        d = 1
        krange = np.arange(0, N, kstep)
                    
        eps_low                 = np.zeros(len(krange))
        eps_upp                 = np.zeros(len(krange))
        
        beta_bar = beta / (2*len(krange))
        
        # Compute violation level (epsilon)
        for i,k in enumerate(krange):
            # Compute violation levels for a specific level of k (nr of
            # discarded constraints)
            eps_low[i] = self.computeBetaPPF(N, k, d, beta_bar)
            eps_upp[i] = self.computeBetaPPF(N, k, d, 1 - beta_bar)

        P_low = np.zeros(N+1)
        P_upp = np.zeros(N+1)
        
        for k in range(0,N+1):
            
            # If k > N-kstep, we need to discard all samples to get a lower bound
            # probability, so the result is zero. The upper bound probability is
            # then given by removing exactly N-kstep samples.
            if k > np.max(krange):
                P_low[k] = 0
                P_upp[k] = 1 - eps_low[-1]
            
            else:  
                
                # Determine the index of the upper bound violation probability to
                # use for computing the lower bound probability.
                id_in = (k-1) // kstep + 1
                
                # Lower bound probability is 1 minus the upper bound violation
                P_low[k] = 1 - eps_upp[id_in]
                
                # If no samples are discarded, even for computing the lower bound
                # probability, we can only conclude an upper bound of one
                if k == 0:
                    P_upp[k] = 1
                else:
                    
                    # Else, the upper bound probability is given by the lower 
                    # bound probability, for discarding "kstep samples less" than
                    # for the lower bound probability
                    id_out = id_in - 1
                    P_upp[k] = 1 - eps_low[id_out]
                    
            # Sanity check to see if the upper bound is actually decreasing with
            # the number of discarded constraints
            if k > 0:
                if P_upp[k] > P_upp[k-1]:
                    print('-- Fix issue in P_upp['+str(k)+']')
                    P_upp[k] = P_upp[k-1]
                    
        # Due to numerical issues, P_low for N_out=0 can be incorrect. Check if 
        # this is the case, and change accordingly.
        if P_low[0] < P_low[1]:
            print('-- Fix numerical error in P_low[0]')
            P_low[0] = 1 - P_upp[N]
                    
        if trials > 0:
            self.validate_eps(trials, N, beta, d, krange, eps_low, eps_upp)

        if export:
            filename = 'input/SaD_probabilityTable_N='+str(N)+'_beta='+str(beta)+'.csv'

            cwd = os.path.dirname(os.path.abspath(__file__))
            root_dir = Path(cwd)

            filepath = Path(root_dir, filename)

            df = pd.DataFrame(np.column_stack((P_low, P_upp)),
                            index=np.arange(N+1),
                            columns=['P_low','P_upp']
                            )
            df.index.name = 'N_out'
            
            df.to_csv(filepath, sep=',')
            
            print('exported table to *.csv file')

        return P_low, P_upp
    
    def compute_transition_intervals(self, beta=1e-6, kstep=1, trials=0):
        """
        Computes transition probability intervals for each (s,a,s') triplet.

        Returns
        -------
        dict
            {(state, action, next_state): (lower_bound, upper_bound)}
        """

        interval_dict = {}

        for (state, action, next_state), N_sas in self.count_state_action_state.items():
            N_sa = self.count_state_action[(state, action)]
            N_out = N_sa - N_sas

            lb, ub = self.compute_interval_for_Nout(N_sa, N_out, beta=beta, kstep=kstep)
            interval_dict[(state, action, next_state)] = (lb, ub)

        self.intervals = interval_dict
    
    def get_intervals(self):
        """
        Returns all computed PAC intervals for (state, action, next_state) triplets.

        Returns
        -------
        dict
            A dictionary of all PAC intervals.
        """
        return self.intervals
    
    def get_specific_interval(self, state, action, next_state):
        """
        Retrieves the specific PAC interval for a given (state, action, next_state) triplet.

        Parameters
        ----------
        state : int
            The source state.
        action : int
            The action taken from the source state.
        next_state : int
            The resulting state after taking the action.

        Returns
        -------
        tuple[float, float]
            The PAC interval for the specified triplet.
        """
        return self.intervals[(state, action, next_state)]

