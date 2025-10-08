import numpy as np

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

def get_state_index(x, y, width):
    """
    Reconstructs the state index from the given (x, y) coordinates.

    Parameters:
        x (int): The x-coordinate (row index).
        y (int): The y-coordinate (column index).
        width (int): The width of the grid.

    Returns:
        int: The corresponding state index.
    """
    return x * width + y

def encodeWetChicken(transition_matrix, width, height):
    """
    Generates a PRISM MDP representation of the Wet Chicken environment.

    Parameters:
        transition_matrix (np.ndarray): A 3D matrix of shape (num_states, num_actions, num_next_states),
                                        where num_states = width * height.
        width (int): The width of the river.
        height (int): The height of the river.

    Returns:
        str: A PRISM MDP module as a string representing the Wet Chicken environment.
    """
    
    actions = [
        "drift",
        "hold",
        "paddle_back",
        "right",
        "left",
    ]
    num_states, num_actions = transition_matrix.shape

    # Initialize the PRISM MDP module 
    prism_lines = ["mdp", "", "module WetChicken"]

    # Define the state variables x and y
    prism_lines.append(f"    s : [0..{width*height - 1}] init 0;  // state-coordinate (position along river)")
    
    # Add transitions
    for state in range(num_states):
        prism_lines.append(f"    // state s = {state}")
        for action in range(num_actions):
            next_states = transition_matrix[state, action]
            valid_next_states = [int(ns) for ns in next_states if ns >= 0]

            if valid_next_states:
                # Uniform probability for each next state
                uniform_prob = 1 / len(valid_next_states)
                transitions = " + ".join(
                    f"{uniform_prob:.3f} : (s'={ns})"
                    for ns in valid_next_states
                )
                prism_lines.append(f"    [{actions[action]}] s={state} -> {transitions};")

    # End the module
    prism_lines.append("endmodule")

    # Return the PRISM MDP as a string
    return "\n".join(prism_lines)
def encodeWetChickenXY(transition_matrix, width, height):
    """
    Generates a PRISM MDP representation of the Wet Chicken environment.

    Parameters:
        transition_matrix (np.ndarray): A 3D matrix of shape (num_states, num_actions, num_next_states),
                                        where num_states = width * height.
        width (int): The width of the river.
        height (int): The height of the river.

    Returns:
        str: A PRISM MDP module as a string representing the Wet Chicken environment.
    """
    
    actions = [
        "drift",
        "hold",
        "paddle_back",
        "right",
        "left",
    ]
    num_states, num_actions = transition_matrix.shape

    # Initialize the PRISM MDP module 
    prism_lines = ["mdp", "", "module WetChicken"]

    # Define the state variables x and y
    prism_lines.append(f"    x : [0..{width - 1}] init 0;  // x-coordinate (position along river)")
    prism_lines.append(f"    y : [0..{height - 1}] init 0;  // y-coordinate (position across river)")
    
    # Add transitions
    for state in range(num_states):
        x = state // height
        y = state % width
        prism_lines.append(f"    // state x = {x}, y = {y}")
        for action in range(num_actions):
            next_states = transition_matrix[state, action]
            valid_next_states = [int(ns) for ns in next_states if ns >= 0]

            if valid_next_states:
                # Uniform probability for each next state
                uniform_prob = 1 / len(valid_next_states)
                transitions = " + ".join(
                    f"{uniform_prob:.3f} : (x'={ns // height}) & (y'={ns % width})"
                    for ns in valid_next_states
                )
                prism_lines.append(f"    [{actions[action]}] x={x} & y={y} -> {transitions};")

    # End the module
    prism_lines.append("endmodule")

    # Return the PRISM MDP as a string
    return "\n".join(prism_lines)

def encodeCartPole(transition_matrix, intervals, initial_state):
    actions = [
        "push left",
        "push right",
    ]
    
    num_states, num_actions = transition_matrix.shape

    # Initialize the PRISM MDP module 
    prism_lines = ["mdp", "", "module CartPole"]
    
    # Add statespace
    prism_lines.append(f"    s : [0..{num_states - 1}] init {initial_state};  // state")
    
    
    # Add transitions
    for state in range(num_states):
        prism_lines.append(f"    // State {state}")
        for action in range(num_actions):
            # Extract possible next states
            next_states = transition_matrix[state, action, :]
            
            # Remove invalid transitions (e.g., -1 to indicate no transition)
            valid_next_states = [int(s) for s in next_states if s >= 0]
            
            if valid_next_states:
                # Uniform probability for each next state
                # uniform_prob = 1 / len(valid_next_states)
                default_interval = (0.0, 1.0)
                probabilities = [
                    f"[{intervals.get((state, action, next_state), default_interval)[0], intervals.get((state, action, next_state), default_interval)[1]}]"
                    for next_state in valid_next_states
                ]
                
                # Create PRISM transition command
                transitions = " + ".join(
                    f"{prob}:(s'={next_state})" for prob, next_state in zip(probabilities, valid_next_states)
                )
                action_label = actions[action]
                prism_lines.append(f"    [{action_label}] s={state} -> {transitions};")
                
    # End the module
    prism_lines.append("endmodule")

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

