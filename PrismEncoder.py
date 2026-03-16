import numpy as np
from collections import deque

def encodeMDP(transition_matrix):
    """
    Generates a PRISM MDP module string from a transition matrix.
    
    Parameters:
        transition_matrix (np.ndarray): A 3D matrix of shape 
                                         (num_states, num_actions, num_next_states).
                                         - Dimension 0: States
                                         - Dimension 1: Actions
                                         - Dimension 2: Next States indices (possible transitions)
    
    Returns:
        str: A PRISM MDP module as a string.
    """
    num_states, num_actions, num_next_states = transition_matrix.shape
    prism_lines = ["mdp", "", "module MDP", f"    s : [0..{num_states - 1}] init 0;  // States"]
    
    # Loop through states and actions
    for state in range(num_states):
        prism_lines.append(f"    // State {state}")
        for action in range(num_actions):
            # Extract possible next states
            next_states = transition_matrix[state, action, :]
            
            # Remove invalid transitions (e.g., -1 to indicate no transition)
            valid_next_states = [int(s) for s in next_states if s >= 0]
            
            if valid_next_states:
                # Uniform probability for each next state
                uniform_prob = 1 / len(valid_next_states)
                probabilities = [f"{uniform_prob:.3f}" for _ in valid_next_states]
                
                # Create PRISM transition command
                # TODO: edit this to use interval instead of uniform distribution                
                transitions = " + ".join(
                    f"{prob} : (s'={next_state})" for prob, next_state in zip(probabilities, valid_next_states)
                )
                action_label = f"action{action}"
                prism_lines.append(f"    [{action_label}] s={state} -> {transitions};")
    
    # Close module
    prism_lines.append("endmodule")
    
    # Return joined lines as a PRISM module string
    return "\n".join(prism_lines)


def encodeCartPole(transition_matrix, intervals, trap):
    actions = [
        "left",
        "right",
    ]
    
    num_states, num_actions = transition_matrix.shape

    # Initialize the PRISM MDP module 
    prism_lines = ["mdp", "", "module CartPole"]
    
    # Add statespace
    prism_lines.append(f"    s : [0..{num_states - 1}];  // state")
    
    # Track states that have at least one real successor
    valid_initial_states = set()
    
    # Add transitions
    for state in range(num_states):
        state_has_successor = False
        prism_lines.append(f"    // State {state}")
        for action in range(num_actions):
            # Extract possible next states
            next_states = transition_matrix[state, action]
            
            # Remove invalid transitions (e.g., -1 to indicate no transition)
            valid_next_states = [int(s) for s in next_states if s >= 0]
            
            # if len(valid_next_states):
            #     # Uniform probability for each next state
            #     # uniform_prob = 1 / len(valid_next_states)
            if len(valid_next_states) == 1:
                if valid_next_states[0] != state:
                    state_has_successor = True
                default_interval = (1.0, 1.0)
                probabilities = [
                    f"[{default_interval[0]}," 
                    f" {default_interval[1]}]"
                    for next_state in valid_next_states
                ]               
            elif len(valid_next_states) > 1:
                state_has_successor = True
                # print(f"state {state} is not like the other states")
                # print(valid_next_states, " ", len(valid_next_states), " >1")
                default_interval = (4.999999999999449e-05, 0.9999999)
                probabilities = [
                    f"[{intervals.get((state, action, next_state), default_interval)[0]}," 
                    f" {intervals.get((state, action, next_state), default_interval)[1]}]"
                    for next_state in valid_next_states
                ]
            else:
                # If there is no transition, loop 
                default_interval = (1.0, 1.0)
                probabilities = [
                    f"[{default_interval[0]}," 
                    f" {default_interval[1]}]"
                    for next_state in [state]
                ]  
                valid_next_states = [state]
            # Create PRISM transition command
            transitions = " + ".join(
                f"{prob}:(s'={next_state})" for prob, next_state in zip(probabilities, valid_next_states)
            )
            action_label = actions[action]
            prism_lines.append(f"    [{action_label}] s={state} -> {transitions};")
            
        if state_has_successor:
            valid_initial_states.add(state)
        
                
    # End the module
    
    prism_lines.append("endmodule")
    valid_initial_states = compute_initial_states(transition_matrix, valid_initial_states)
    if valid_initial_states:
        init_condition = " | ".join(f"(s={s})" for s in sorted(valid_initial_states))
        prism_lines.append(f"init {init_condition} endinit")
    prism_lines.append(f'label "trap" = s={trap[0]};')
    # Return the PRISM MDP as a string
    return "\n".join(prism_lines)

def encodeMaze(transition_matrix, intervals, trap, goal):
    actions = [
        "Up",
        "Down",
        "Right",
        "Left"
    ]
    
    num_states, num_actions = transition_matrix.shape

    # Initialize the PRISM MDP module 
    prism_lines = ["mdp", "", "module Maze"]
    
    # Add statespace
    prism_lines.append(f"    s : [0..{num_states - 1}];  // state")
    
    
    # Add transitions
    for state in range(num_states):
        prism_lines.append(f"    // State {state}")
        for action in range(num_actions):
            # Extract possible next states
            next_states = transition_matrix[state, action]
            
            # Remove invalid transitions (e.g., -1 to indicate no transition)
            valid_next_states = [int(s) for s in next_states if s >= 0]
            
            # if len(valid_next_states):
            #     # Uniform probability for each next state
            #     # uniform_prob = 1 / len(valid_next_states)
            if len(valid_next_states) == 1:
                # print(valid_next_states, " ", len(valid_next_states), " 1")
                default_interval = (1.0, 1.0)
                probabilities = [
                    f"[{default_interval[0]}," 
                    f" {default_interval[1]}]"
                    for next_state in valid_next_states
                ]               
            elif len(valid_next_states) > 1:
                # print(valid_next_states, " ", len(valid_next_states), " >1")
                default_interval = (4.999999999999449e-05, 0.9999999)
                probabilities = [
                    f"[{intervals.get((state, action, next_state), default_interval)[0]}," 
                    f" {intervals.get((state, action, next_state), default_interval)[1]}]"
                    for next_state in valid_next_states
                ]
            else:
                # If there is no transition, loop 
                default_interval = (1.0, 1.0)
                probabilities = [
                    f"[{default_interval[0]}," 
                    f" {default_interval[1]}]"
                    for next_state in [state]
                ]  
                valid_next_states = [state]
            # Create PRISM transition command
            transitions = " + ".join(
                f"{prob}:(s'={next_state})" for prob, next_state in zip(probabilities, valid_next_states)
            )
            action_label = actions[action]
            prism_lines.append(f"    [{action_label}] s={state} -> {transitions};")
        
                
    # End the module
    
    prism_lines.append("endmodule")
    prism_lines.append(f"init s<={num_states} endinit")
    prism_lines.append(f'label "trap" = s={trap[0]};')
    prism_lines.append(f'label "goal" = s={goal[0]};')
    # Return the PRISM MDP as a string
    return "\n".join(prism_lines)

def encodeFrozenLake(transition_matrix, intervals, trap, goal):
    actions = [
        "Left",
        "Down",
        "Right",
        "Up"
    ]
    
    num_states, num_actions = transition_matrix.shape

    # Initialize the PRISM MDP module 
    prism_lines = ["mdp", "", "module Maze"]
    
    # Add statespace
    prism_lines.append(f"    s : [0..{num_states - 1}];  // state")
    
    
    # Add transitions
    for state in range(num_states):
        prism_lines.append(f"    // State {state}")
        for action in range(num_actions):
            # Extract possible next states
            next_states = transition_matrix[state, action]
            
            # Remove invalid transitions (e.g., -1 to indicate no transition)
            valid_next_states = [int(s) for s in next_states if s >= 0]
            
            # if len(valid_next_states):
            #     # Uniform probability for each next state
            #     # uniform_prob = 1 / len(valid_next_states)
            if len(valid_next_states) == 1:
                # print(valid_next_states, " ", len(valid_next_states), " 1")
                default_interval = (1.0, 1.0)
                probabilities = [
                    f"[{default_interval[0]}," 
                    f" {default_interval[1]}]"
                    for next_state in valid_next_states
                ]               
            elif len(valid_next_states) > 1:
                # print(valid_next_states, " ", len(valid_next_states), " >1")
                default_interval = (4.999999999999449e-05, 0.9999999)
                probabilities = [
                    f"[{intervals.get((state, action, next_state), default_interval)[0]}," 
                    f" {intervals.get((state, action, next_state), default_interval)[1]}]"
                    for next_state in valid_next_states
                ]
            else:
                # If there is no transition, loop 
                default_interval = (1.0, 1.0)
                probabilities = [
                    f"[{default_interval[0]}," 
                    f" {default_interval[1]}]"
                    for next_state in [state]
                ]  
                valid_next_states = [state]
            # Create PRISM transition command
            transitions = " + ".join(
                f"{prob}:(s'={next_state})" for prob, next_state in zip(probabilities, valid_next_states)
            )
            action_label = actions[action]
            prism_lines.append(f"    [{action_label}] s={state} -> {transitions};")
        
                
    # End the module
    
    prism_lines.append("endmodule")
    prism_lines.append(f"init s<={num_states} endinit")
    prism_lines.append(f'label "hole" = s={trap[0]};')
    if goal:
        prism_lines.append(f'label "goal" = s={goal[0]};')
    # Return the PRISM MDP as a string
    return "\n".join(prism_lines)

def encodeCrashingMountainCar(transition_matrix, intervals, initial_state, trap, goal):
    actions = [
        "left",
        "stay",
        "right",
    ]
    
    num_states, num_actions = transition_matrix.shape

    # Initialize the PRISM MDP module 
    prism_lines = ["mdp", "", "module CrashingMountainCar"]
    
    # Add statespace
    prism_lines.append(f"    s : [0..{num_states - 1}];  // state")
    
    
    for state in range(num_states):
        prism_lines.append(f"    // State {state}")
        for action in range(num_actions):
            # Extract possible next states
            next_states = transition_matrix[state, action]
            
            # Remove invalid transitions (e.g., -1 to indicate no transition)
            valid_next_states = [int(s) for s in next_states if s >= 0]
            
            # if len(valid_next_states):
            #     # Uniform probability for each next state
            #     # uniform_prob = 1 / len(valid_next_states)
            if len(valid_next_states) == 1:
                # print(valid_next_states, " ", len(valid_next_states), " 1")
                default_interval = (1.0, 1.0)
                probabilities = [
                    f"[{default_interval[0]}," 
                    f" {default_interval[1]}]"
                    for next_state in valid_next_states
                ]               
            elif len(valid_next_states) > 1:
                # print(valid_next_states, " ", len(valid_next_states), " >1")
                default_interval = (4.999999999999449e-05, 0.9999999)
                probabilities = [
                    f"[{intervals.get((state, action, next_state), default_interval)[0]}," 
                    f" {intervals.get((state, action, next_state), default_interval)[1]}]"
                    for next_state in valid_next_states
                ]
            else:
                # If there is no transition, loop 
                default_interval = (1.0, 1.0)
                probabilities = [
                    f"[{default_interval[0]}," 
                    f" {default_interval[1]}]"
                    for next_state in [state]
                ]  
                valid_next_states = [state]
            # Create PRISM transition command
            transitions = " + ".join(
                f"{prob}:(s'={next_state})" for prob, next_state in zip(probabilities, valid_next_states)
            )
            action_label = actions[action]
            prism_lines.append(f"    [{action_label}] s={state} -> {transitions};")
                
    # End the module
    prism_lines.append("endmodule")
    prism_lines.append(f"init s<={num_states} endinit")
    trap_expr = " | ".join([f"s={t}" for t in trap])
    prism_lines.append(f'label "trap" = {trap_expr};')
    prism_lines.append(f'label "goal" = s={goal[0]};')

    # Return the PRISM MDP as a string
    return "\n".join(prism_lines)

def set_initial_state(prism_mdp, initial_state):
    """
    Modifies the initial state in a PRISM MDP module string.
    
    Parameters:
        prism_mdp (str): The PRISM MDP module as a string.
        initial_state (int): The desired initial state.
    
    Returns:
        str: The modified PRISM MDP module string.
    """
    lines = prism_mdp.splitlines()
    for i, line in enumerate(lines):
        # Look for the state variable definition line (e.g., "s : [0..N] init X;")
        if "init" in line and ":" in line:
            # Replace the current initial state with the new one
            lines[i] = line.split("init")[0] + f"init {initial_state};"
            break
    return "\n".join(lines)

def add_initial_state_to_prism_mdp(prism_mdp, new_state, action, next_states) :
    """
    Adds a new state with a specified action and transitions to a PRISM MDP module,
    sets the new state as the initial state, and adjusts the state variable range.
    
    Parameters:
        prism_mdp (str): The PRISM MDP module as a string.
        new_state (int): The value of the new state (e.g., -1).
        action (int): The integer representing the action.
        next_states (list[int]): The set of possible next states for the action.
    
    Returns:
        str: The modified PRISM MDP module string.
    """
    # Split the PRISM MDP module into lines for processing
    lines = prism_mdp.splitlines()
    
    # Find and update the state variable range
    for i, line in enumerate(lines):
        if "s : [" in line and "init" in line:
            # Extract the state range and adjust it to include the new state
            state_range = line.split("s : [")[1].split("]")[0]
            start, end = map(int, state_range.split(".."))
            new_start = min(new_state, start)
            new_end = max(new_state, end)
            lines[i] = f"    s : [{new_start}..{new_end}] init {new_state};  // States"
            break
    
    # Add the new state and action transitions
    uniform_prob = 1 / len(next_states)
    transitions = " + ".join(f"{uniform_prob:.3f} : (s'={s})" for s in next_states)
    new_command = f"    [action{action}] s={new_state} -> {transitions};"
    
    # Insert the new command into the module
    for i, line in enumerate(lines):
        if "endmodule" in line:
            lines.insert(i, new_command)
            break
    
    return "\n".join(lines)

def remove_state_transitions(prism_mdp, states):
    """
    Removes all transition functions associated with the specified states in the PRISM MDP string.
    
    Parameters:
        prism_mdp (str): The PRISM MDP module as a string.
        states (list[int]): The list of states whose transitions should be removed.
    
    Returns:
        str: The modified PRISM MDP module string.
    """
    # Generate a set of states to match
    state_set = set(states)
    
    # Split the MDP into lines
    lines = prism_mdp.splitlines()
    filtered_lines = []
    
    for line in lines:
        # Check if the line is a transition line (starts with an action and contains "s=")
        if any(f"s={state}" in line for state in state_set) and "[action" in line:
            # Skip lines containing transitions for states in `state_set`
            continue
        # Otherwise, keep the line
        filtered_lines.append(line)
    
    return "\n".join(filtered_lines)

def encodeLunarLander(transition_matrix, intervals, trap, goal, init):
    actions = [
        "Up",
        "Down",
        "Right",
        "Left"
    ]
    
    num_states, num_actions = transition_matrix.shape

    prism_lines = ["mdp", "", "module Maze"]
    
    # Add state space
    prism_lines.append(f"    s : [0..{num_states - 1}];  // state")

    # Track states that have at least one real successor
    valid_initial_states = set()

    # Add transitions
    for state in range(num_states):
        prism_lines.append(f"    // State {state}")
        state_has_successor = False

        for action in range(num_actions):

            next_states = transition_matrix[state, action]
            valid_next_states = [int(s) for s in next_states if s >= 0]

            if len(valid_next_states) == 1:
                if valid_next_states[0] != state:
                    state_has_successor = True
                default_interval = (1.0, 1.0)
                probabilities = [
                    f"[{default_interval[0]}, {default_interval[1]}]"
                    for _ in valid_next_states
                ]

            elif len(valid_next_states) > 1:
                state_has_successor = True
                default_interval = (4.999999999999449e-05, 0.9999999)
                probabilities = [
                    f"[{intervals.get((state, action, next_state), default_interval)[0]}, "
                    f"{intervals.get((state, action, next_state), default_interval)[1]}]"
                    for next_state in valid_next_states
                ]

            else:
                # No transition = self-loop
                default_interval = (1.0, 1.0)
                valid_next_states = [state]
                probabilities = [
                    f"[{default_interval[0]}, {default_interval[1]}]"
                ]

            transitions = " + ".join(
                f"{prob}:(s'={next_state})"
                for prob, next_state in zip(probabilities, valid_next_states)
            )

            action_label = actions[action]
            prism_lines.append(
                f"    [{action_label}] s={state} -> {transitions};"
            )

        if state_has_successor:
            valid_initial_states.add(state)

    prism_lines.append("endmodule")

    # Initial states = all states with at least one successor
    valid_initial_states = compute_initial_states(transition_matrix, valid_initial_states)
    if valid_initial_states:
        init_condition = " | ".join(f"(s={s})" for s in sorted(valid_initial_states))
        prism_lines.append(f"init {init_condition} endinit")

    # Trap label
    if trap:
        add_label(prism_lines, "crash", trap)

    # Goal label
    if goal:
        add_label(prism_lines, "goal", goal)
    else:
        prism_lines.append('label "goal" = false;')

    return "\n".join(prism_lines)

def compute_initial_states(transition_matrix, relevant_states):
    """
    Compute minimal set of initial states such that every relevant state
    is reachable from at least one initial state.

    Parameters
    ----------
    transition_matrix : np.ndarray (num_states, num_actions)
        Each entry contains a list/array of next states.

    relevant_states : set[int]
        States with outgoing transitions to a different state.

    Returns
    -------
    list[int]
        Selected initial states
    """

    num_actions = transition_matrix.shape[1]

    relevant_states = set(relevant_states)

    # Fast membership lookup
    is_relevant = [False] * transition_matrix.shape[0]
    for s in relevant_states:
        is_relevant[s] = True

    # Precompute adjacency only for relevant states
    adj = {}
    out_degree = {}

    for s in relevant_states:
        succ = []
        count = 0

        for a in range(num_actions):
            for ns in transition_matrix[s, a]:
                if ns >= 0 and ns != s and is_relevant[ns]:
                    succ.append(ns)
                    count += 1

        adj[s] = succ
        out_degree[s] = count

    # Sort states by descending outgoing transitions
    states_sorted = sorted(relevant_states, key=lambda s: out_degree[s], reverse=True)

    visited = set()
    initial_states = []

    for s in states_sorted:

        if s in visited:
            continue

        # select as initial state
        initial_states.append(s)

        # BFS/DFS reachability
        queue = deque([s])
        visited.add(s)

        while queue:
            u = queue.popleft()

            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)

    # select states without relevant predecessors
    print("total relevant states = ", len(relevant_states))
    print("The total number of initial states in therefore = ",len(initial_states))

    return sorted(initial_states)
    
def states_to_ranges(states):
    """
    Convert a list of integers into a list of (start, end) ranges.
    Example: [1,2,3,7,8,10] -> [(1,3), (7,8), (10,10)]
    """
    if not states:
        return []

    states = sorted(set(states))
    ranges = []

    start = prev = states[0]

    for s in states[1:]:
        if s == prev + 1:
            prev = s
        else:
            ranges.append((start, prev))
            start = prev = s
    ranges.append((start, prev))

    return ranges


def build_range_expression(ranges, chunk_size=500):
    """
    Build a PRISM-safe expression using ranges and chunking.
    """

    # Convert ranges into PRISM fragments
    fragments = []
    for start, end in ranges:
        if start == end:
            fragments.append(f"(s={start})")
        else:
            fragments.append(f"(s >= {start} & s <= {end})")

    if not fragments:
        return None

    # If small enough, just join directly
    if len(fragments) <= chunk_size:
        return " | ".join(fragments)

    # Otherwise, chunk into balanced subexpressions
    chunks = [
        " | ".join(fragments[i:i+chunk_size])
        for i in range(0, len(fragments), chunk_size)
    ]

    return " | ".join(f"({chunk})" for chunk in chunks)

def add_label(prism_lines, label_name, states):
    if not states:
        return

    ranges = states_to_ranges(states)
    expression = build_range_expression(ranges)

    prism_lines.append(f'label "{label_name}" = {expression};')
    
def add_reach_label(prism_mdp, states):
    """
    Adds a label "reach" to specified states in a PRISM MDP module string.
    
    Parameters:
        prism_mdp (str): The PRISM MDP module as a string.
        states (list[int]): A list of states to be labeled as "reach".
    
    Returns:
        str: The modified PRISM MDP module string with the added labels.
    """
    # Generate the label definition
    reach_states = " | ".join(f"s={state}" for state in states)
    reach_label = f'label "reach" = {reach_states};'
    prism_mdp = remove_state_transitions(prism_mdp, states)
    
    # Append the label definition to the PRISM module
    return prism_mdp.strip() + "\n\n" + reach_label + "\n"

def add_avoid_label(prism_mdp, states):
    """
    Adds a label "reach" to specified states in a PRISM MDP module string.
    
    Parameters:
        prism_mdp (str): The PRISM MDP module as a string.
        states (list[int]): A list of states to be labeled as "reach".
    
    Returns:
        str: The modified PRISM MDP module string with the added labels.
    """
    # Generate the label definition
    avoid_states = " | ".join([f"s={state}" for state in states])
    avoid_label = f'label "trap" = {avoid_states};'
    # Append the label definition to the PRISM module
    prism_mdp = remove_state_transitions(prism_mdp, states)
    
    return prism_mdp.strip() + "\n\n" + avoid_label + "\n"

