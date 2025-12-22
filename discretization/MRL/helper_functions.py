import pandas as pd

# Function that transforms the dataset into the right format
# Trajectories to Dataframe

def trajToDF(trajectories, num_features):
    '''
    Docstring for trajToDF
    
    :param trajectories: Trajectories, a list of transitions [state, action, next_state, reward, is_done]
    
    returns:
    
    df: a dataframe containing the trajectories
    '''
    columns = (
        ["ID", "TIME"]
        + [f"FEATURE_{i}" for i in range(num_features)]
        + ["ACTION", "RISK"]
    )
    
    rows = []
    trajectory_count = 0
    for trajectory in trajectories:
        transition_count = 0
        for transition in trajectory:
            s = transition[0] # Is state a single number or a list? surely its a list right? its continuous, there aint enough number
            # print("s = ", s)
            a = transition[1]
            # print("a = ", a)
            n_s = transition[2]
            # print("ns = ", n_s)
            r = transition [3]
            # print("r = ", r)
            died = transition[5]
            transition_count+=1
            risk = r
            if died == True:
                risk = 0
                
            rows.append([trajectory_count, transition_count]
                        + [val for val in s] +
                        [a, risk])
        trajectory_count+=1
            
    data_df = pd.DataFrame(rows, columns=columns)
    return data_df

def state2region(predictor, state, num_features):
    columns = [f"FEATURE_{i}" for i in range(num_features)]
    rows = [[val for val in state]]
    new_df = pd.DataFrame(rows, columns=columns)
    region = predictor.predict(new_df)
    return region[0]