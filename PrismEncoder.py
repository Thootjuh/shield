def encodeWetChicken(transition_matrix, intervals, trap, goal, init):
    actions = [
        "drift", "hold", "paddle_back", "right", "left", "reset"
    ]
    
    num_states, num_actions = transition_matrix.shape

    # Initialize the PRISM MDP module 
    prism_lines = ["mdp", "", "module WetChicken"]
    
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
            
            if len(valid_next_states) == 1:
                default_interval = (1.0, 1.0)
                probabilities = [
                    f"[{default_interval[0]}," 
                    f" {default_interval[1]}]"
                    for next_state in valid_next_states
                ]               
            
            elif len(valid_next_states) > 1:
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
    prism_lines.append(f"init s = {init} endinit")
    prism_lines.append(f'label "waterfall" = s={trap[0]};')
    
    reach_states = " | ".join(f"s={state}" for state in goal)
    prism_lines.append(f'label "goal" = {reach_states};')
    # Return the PRISM MDP as a string
    return "\n".join(prism_lines)