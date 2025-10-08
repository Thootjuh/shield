# This file contains code related to the partitioning of the continuous state space
# Code from: https://github.com/LAVA-LAB/DynAbs
import numpy as np
import itertools

def define_partition(dim, nrPerDim, regionWidth, origin):
    '''
    Define the partitions object `partitions` based on given settings.

    Parameters
    ----------
    dim : int
        Dimension of the state (of the LTI system).
    nrPerDim : list
        List of integers, where each value is the number of regions in that 
        dimension.
    regionWidth : list
        Width of the regions in every dimension.
    origin : list
        Coordinates of the origin of the continuous state space. 

    Returns
    -------
    partition : dict
        Dictionary containing the info regarding partisions.

    '''
    
    regionWidth = np.array(regionWidth)
    origin      = np.array(origin)
    
    elemVector   = dict()
    for i in range(dim):
        
        elemVector[i] = np.linspace(-(nrPerDim[i]-1)/2, 
                                     (nrPerDim[i]-1)/2,
                                     int(nrPerDim[i]))
        
    widthArrays = [[x*regionWidth[i] for x in elemVector[i]] 
                                              for i in range(dim)]
    
    idxArrays = [range(len(arr)) for arr in widthArrays]
    
    
    nr_regions = np.prod(nrPerDim)
    partition = {'center': np.zeros((nr_regions, dim), dtype=float), 
                 'idx': {},
                 'idx_inv': {},
                 'c_tuple': {},
                 'width': regionWidth,
                 'number': np.array(nrPerDim),
                 'origin': origin,
                 'nr_regions': nr_regions}
    
    partition['low'] = np.zeros((nr_regions, dim), dtype=float)
    partition['upp'] = np.zeros((nr_regions, dim), dtype=float)
    
    for i,(pos,idx) in enumerate(zip(itertools.product(*widthArrays),
                                     itertools.product(*idxArrays))):
        
        center = np.array(pos) + origin
        
        dec = 5
        
        partition['center'][i] = np.round(center, decimals=dec)
        partition['c_tuple'][tuple(np.round(center, decimals=dec))] = i
        partition['low'][i] = np.round(center - regionWidth/2, 
                                        decimals=dec)
        partition['upp'][i] = np.round(center + regionWidth/2, 
                                        decimals=dec)
        partition['idx'][tuple(idx)] = i
        partition['idx_inv'][i] = tuple(idx)
    
    return partition

def computeRegionCenters(points, partition):
    '''
    Function to compute to which region (center) a list of points belong

    Parameters
    ----------
    points : 2D Numpy array
        Array, with every row being a point to determine the center point for.
    partition : dict
        Dictionary of the partition.

    Returns
    -------
    2D Numpy array
        Array, with every row being the center coordinate of that row of the
        input array.

    '''
    
    # Check if 'points' is a vector or matrix
    if len(np.shape(points)) == 1:
        points = np.reshape(points, (1,len(points)))
    
    # Retreive partition parameters
    region_width = np.array(partition['width'])
    region_nrPerDim = partition['number']
    dim = len(region_width)
    
    # Boolean list per dimension if it has a region with the origin as center
    originCentered = [True if nr % 2 != 0 else False for nr in region_nrPerDim]

    # Initialize centers array
    centers = np.zeros(np.shape(points)) 
    
    # Shift the points to account for a non-zero origin
    originShift = np.array(partition['origin'] )
    pointsShift = points - originShift
    
    for q in range(dim):
        # Compute the center coordinates of every shifted point
        if originCentered[q]:
            
            centers[:,q] = ((pointsShift[:,q]+0.5*region_width[q]) //
                             region_width[q]) * region_width[q]
        
        else:
            
            centers[:,q] = (pointsShift[:,q] // region_width[q]) * \
                            region_width[q] + 0.5*region_width[q]
    
    # Add the origin again to obtain the absolute center coordinates
    return np.round(centers + originShift, decimals=5)

def successors_of_state_action_abstraction(
    s_idx, a_idx, partition, A_cl, B, K, U, U_prime_values, noise_sampler, N=1000
):
    """
    Compute possible successor abstract states for a given (s, a) pair
    using the abstraction method from Badings et al. (2024), but with
    physically meaningful d_a derived from linearized CartPole dynamics.
    """
    x_center = partition['center'][s_idx]
    # u_prime = np.array([U_prime_values[a_idx]])  # residual control for this abstract action
    u_prime = np.linalg.lstsq(B, d_a - A_cl @ x_center, rcond=None)[0]
    d_a = A_cl @ x_center + B @ u_prime  # nominal next state (paper-style)

    # Now simulate noise to estimate transition probabilities
    counts = np.zeros(partition['nr_regions'])
    for _ in range(N):
        eta = noise_sampler()
        x_next = d_a.flatten() + eta
        s_next = state2region(x_next, partition, partition['c_tuple'])
        if s_next is not None:
            counts[s_next] += 1

    N_eff = counts.sum()
    if N_eff == 0:
        return {}

    successors = {}
    for s_prime, count in enumerate(counts):
        if count > 0:
            low, upp = proportion_confint(count, N_eff, alpha=0.05, method="beta")
            successors[s_prime] = [low, upp]

    return successors

def successor_states(s_idx, a_idx, partition, A_cl, B, U_prime_values, noise_sampler, N=1000):
    x_center = partition['center'][s_idx]
    u_prime = np.array([U_prime_values[a_idx]])
    d_a = A_cl @ x_center + B @ u_prime
    counts = np.zeros(partition['nr_regions']+1)
    
    for i in range(N):
        eta = noise_sampler[i]
        x_next = d_a + eta
        s_next = state2region(x_next, partition, partition['c_tuple'])
        if s_next is not None:
            counts[s_next] += 1
        else:
            counts[-1] += 1
    successors = [i for i, count in enumerate(counts) if count > 0]
    return successors

def successors_of_state_action(s_idx, a_idx, partition, actions, A_cl, B, K, U, U_prime, noise_sampler, N=1000):
    """
    Compute possible successor abstract states for a given (s, a) pair.

    Parameters
    ----------
    s_idx : int
        Index of the abstract state (region).
    a_idx : int
        Index of the abstract action.
    partition : dict
        Partition object returned by define_partition.
    actions : np.ndarray
        Array of abstract actions (shape: [n_actions, state_dim]).
        Each action corresponds to a target point d_a in state space.
    A_cl, B, K : np.ndarray
        Closed-loop system matrices (discrete-time).
    U : tuple
        Continuous input set (min, max).
    U_prime : tuple
        Residual control bounds (min, max).
    noise_sampler : callable
        Function that returns a noise sample vector η (shape: [state_dim]).
    N : int
        Number of noise samples to draw for estimating successors.

    Returns
    -------
    successors : dict
        Dictionary mapping successor abstract state index s' → [lower_prob, upper_prob].
    """

    dim = partition['center'].shape[1]
    d_a = actions[a_idx]

    # Pick a representative x from the abstract state (e.g., its center)
    x_center = partition['center'][s_idx]

    # Solve for residual control u' at x_center: A_cl x + B u' ≈ d_a
    u_prime = np.linalg.lstsq(B, d_a - A_cl @ x_center, rcond=None)[0]
    u_actual = -K @ x_center + u_prime

    # Check feasibility (optional: skip if infeasible)
    # if not (U_prime[0] <= u_prime <= U_prime[1] and U[0] <= u_actual <= U[1]):
    #     return {}

    # Now simulate noisy successors
    counts = np.zeros(partition['nr_regions'])
    for _ in range(N):
        eta = noise_sampler()
        x_next = A_cl @ x_center + B @ u_prime + eta
        s_next = state2region(x_next, partition, partition['c_tuple'])
        if s_next is not None:
            counts[s_next] += 1

    # Empirical probabilities
    probs = counts / N

    # Convert to confidence intervals (here Clopper–Pearson style)
    successors = {}
    alpha = 0.05
    from statsmodels.stats.proportion import proportion_confint
    for s_prime, count in enumerate(counts):
        if count > 0:
            low, upp = proportion_confint(count, N, alpha=alpha, method="beta")
            successors[s_prime] = [low, upp]

    return successors

def state2region(state, partition, c_tuple):

    region_centers = computeRegionCenters(state, partition)

    try:
        region_idx = [c_tuple[tuple(c)] for c in region_centers]
        return region_idx
    except:
        # print('ERROR: state',state,'does not belong to any region')
        return None