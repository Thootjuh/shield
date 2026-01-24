import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

class MRL_scratch:
    def __init__(self, data, max_k, num_features, standard_reward, threshold, verbose=True):
        self.data_df = self.trajToDF(data, num_features, standard_reward)
        self.max_k = max_k
        self.num_features = num_features
        self.verbose=verbose
        self.th = threshold
    
    def trajToDF(self, data, num_features, standard_reward):
        '''
        Docstring for trajToDF
        
        :param trajectories: Trajectories, a list of transitions [state, action, next_state, reward, terminated, truncated]
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
        for trajectory in data:
            transition_count = 0
            r = standard_reward
            for transition in trajectory:
                s = transition[0] 
                a = transition[1]
                n_s = transition[2]
                transition_count+=1
                risk = r         
                rows.append([trajectory_count, transition_count]
                                + [val for val in s] +
                                [a, risk])
                r = transition[3] # The reward associated with reaching the state is the reward earned in the previous transition
            
            if transition[4]: # If an end state is reached, add one more transition
                transition_count+=1
                rows.append([trajectory_count, transition_count]
                        + [val for val in n_s] +
                        ["None", r])
            trajectory_count+=1
                
        data_df = pd.DataFrame(rows, columns=columns)
        
        return data_df
    
    def initial_split(self):
        # Initially, the statespace is split based on the reward. Every reward value gets a cluster
        # Note that this assumes that the reward function is not continuous
        # Can alternatively implement a k-means version that can handle continuous reward functions
        nb_states = self.data_df["RISK"].nunique() 
        reward_state_index = {risk: i for i, risk in enumerate(self.data_df["RISK"].unique())}
        
        # create the initial training dataframe
        
        # Copy copyable (is that a word?) states from the data_df
        new_df = self.data_df[
            ["ID", "TIME"]
            + [f"FEATURE_{i}" for i in range(self.num_features)]
            + ["ACTION", "RISK"]
        ].copy()
        
        # Set initial clusters using the reward_state_index dictionary
        new_df["CLUSTER"] = new_df["RISK"].map(reward_state_index)
        
        # Set the next cluster by using reward_state_index on the next state, unless the next state is in a new trajectory
        next_time = new_df["TIME"].shift(-1)
        next_risk = new_df["RISK"].shift(-1)
        new_df["NEXT_CLUSTER"] = np.where(
            next_time > new_df["TIME"],
            next_risk.map(reward_state_index),
            -1
        ).astype(int)
        new_df
        
        if self.verbose:
            new_df.to_csv("initial_clusters.csv", index=False)
            print(f"Initial Split has {nb_states} states")
        
        return new_df
        
        
     
        
    def find_incoherence(self):   
        X = self.train_df.loc[:, ["CLUSTER", "NEXT_CLUSTER", "ACTION"]] 
        # keep only valid transitions
        X = X[X.NEXT_CLUSTER != -1] 
        
        # find  contradications, ie (cluster, action) pairs with multiple next states
        count = X.groupby(["CLUSTER", "ACTION"])["NEXT_CLUSTER"].nunique() 
        contradictions = list(count[list(count > 1)].index) 
        
        # If no contradictions: Doen
        if not len(contradictions): 
            return None 
        
        # transition counts per next cluster
        transition_counts = ( X.groupby(["CLUSTER", "ACTION", "NEXT_CLUSTER"]).size().unstack("NEXT_CLUSTER") ) 
        transition_counts = transition_counts.loc[contradictions] 
        
        # Find the contradiction with the second largest next state, and select that one for splitting
        def second_largest(series): 
            ssort = series.sort_values(ascending=False) 
            return ssort.iloc[1] 
        
        seclargest = transition_counts.apply(second_largest, axis=1) 
        c, a = seclargest.idxmax() 
        
        
        if seclargest.max() > self.th: 
            dominant_next_cluster = transition_counts.loc[(c, a)].idxmax()
            return (c, a, dominant_next_cluster) 
        
        return None

    
    def split(self, c, a, dominant_next_cluster):
        # create a fresh cluster id
        new_cluster = self.train_df["CLUSTER"].max() + 1

        df = self.train_df

        # iterate by position to allow previous-row updates
        for i in range(len(df)):
            print(i)
            row = df.iloc[i]

            # check matching (cluster, action)
            if row["CLUSTER"] == c and row["ACTION"] == a:

                # case 1: dominant transition → do nothing
                if row["NEXT_CLUSTER"] == dominant_next_cluster:
                    continue

                # case 2: non-dominant transition → split
                df.at[df.index[i], "CLUSTER"] = new_cluster

                # update previous row's NEXT_CLUSTER if valid
                if i > 0 and df.iloc[i - 1]["NEXT_CLUSTER"] != -1:
                    df.at[df.index[i - 1], "NEXT_CLUSTER"] = new_cluster

        return df
    
    def cluster_update(self, df, ids, k):
        df.loc[df.index.isin(ids), "CLUSTER"] = k
        previds = ids - 1
        df.loc[
            (df.index.isin(previds)) & (df["ID"] == df["ID"].shift(-1)), "NEXT_CLUSTER"
        ] = k
        return df
    
    def split_with_classifier(self, c, a, nc):
        """
        Resolves contradictions by splitting the data using a classifier.

        Parameters:
        df (DataFrame): Input dataframe.
        c (int): Initial cluster to split.
        a (int): Action to target.
        nc (int): Dominant next cluster.
        k (int): Indexer for next cluster (used for NEXT_CLUSTER updates).

        Returns:
        DataFrame: Updated dataframe with new cluster assignments.
        """
        k = self.train_df["CLUSTER"].max()

        df = self.train_df
        # select labeled rows for training
        labeled_parts = df[
            (df["CLUSTER"] == c) &
            (df["ACTION"] == a) &
            (df["NEXT_CLUSTER"] != -1)
        ]

        unlabeled_parts = df[df["CLUSTER"] == c]

        # assign labels: 1 if NEXT_CLUSTER == nc (dominant), 0 otherwise
        labeled_parts["LABEL"] = (labeled_parts["NEXT_CLUSTER"] == nc).astype(int)


        relabel_train = True
        m = LogisticRegression(random_state=0)
        tr_X = labeled_parts.iloc[:, 2 : 2 + self.num_features]
        tr_y = labeled_parts["LABEL"]
        
        m.fit(tr_X, tr_y.values.ravel())
            
        test_X = unlabeled_parts.iloc[:, 2 : 2 + self.num_features]
        
        if len(unlabeled_parts):
            Y = m.predict(test_X)
            unlabeled_parts["GROUP"] = Y.ravel()
                   
        for cluster_index in range(1, 2):
            ids = labeled_parts.loc[labeled_parts["LABEL"] == cluster_index].index.values
            if len(unlabeled_parts):
                id2 = unlabeled_parts.loc[
                    unlabeled_parts["GROUP"] == cluster_index
                ].index.values
                if relabel_train:
                    ids = id2
                else:
                    ids = np.concatenate((ids, id2))
            assert (
                df.loc[df.index.isin(ids), "CLUSTER"] == c
            ).all(), "trying to reassign cluster to points out of original cluster"
            self.cluster_update(df, ids, k + cluster_index - 1)
        return df

            

    def train_classifier(self):
        pass
    
    def fit(self):
        # get the initial split
        self.train_df = self.initial_split()
        
        # Split the dataset a maximum of max_k times
        for i in range(self.max_k):
            print(i)
            # Find the biggest incoherence
            incoherence = self.find_incoherence()
            if incoherence == None:
                break
            
            # split the incoherence
            self.train_df = self.split_with_classifier(incoherence[0], incoherence[1], incoherence[2])
        
        
        if self.verbose:
            self.train_df.to_csv("split_clusters.csv", index=False)
            nb_states = self.train_df["CLUSTER"].nunique() 
            print(f"Final Split has {nb_states} states")
        # Train a classifier on the split dataset
        # self.train_classifier()
        pass
    
    def predict(self, state):
        pass
    
    def get_reward_function(self):
        pass
    
    def get_transition_function(self):
        pass
    
    def get_next_state_list(self):
        pass