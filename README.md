This repository was adapted fro https://github.com/Philipp238/Safe-Policy-Improvement-Approaches-on-Discrete-Markov-Decision-Processes/tree/master, which contains the code accompanying the paper "Safe Policy Improvement Approaches on Discrete Markov Decision Processes" by Philipp Scholl, Felix Dietrich, Clemens Otte, and Steffen Udluft published at ICAART 2022, the chapter "Safe Policy Improvement Approaches and Their Limitations" by Philipp Scholl, Felix Dietrich, Clemens Otte, and Steffen Udluft of the Springer book series "Lecture notes in Artificial Intelligence" and the master's thesis "Evaluation of Safe Policy Improvement with Soft Baseline Bootstrapping" by Philipp Scholl [1], which 
investigates safe reinforcement learning
by building on the paper "Safe Policy Improvement with Soft Baseline Bootstrapping" by Nadjahi 
et al. [2] and their code https://github.com/RomainLaroche/SPIBB.

## Requirements
The code is implemented in Python version 3.10 and requires the packages specified in ``requirements.txt``. Additionally, Stormpy is required. https://github.com/moves-rwth/stormpy.
Some of the included bechmarks also require stormvogel, which can be found at https://moves-rwth.github.io/stormvogel/index.html

Before running the experiments, you
have to create a file called `paths.ini` (on the highest directory level) which contains the following:
````
[PATHS]
results_path = D:\results
````
Where results_path should be the absolute path pointing to the place where the results should be stored (I store the results outside of this repository as the results are often huge (>1GB))

## Structure
To run the experiment on the randomMDPs benchmark, one can use 

`python run_experiments.py random_mdps_shield_compare.ini experiment_results 1234 1 4`  
or
`python run_experiments.py wet_chicken_full.ini wet_chicken_results 1234 1 4` 

The `random_mdps_shield_full.ini` is the name of the config file used for this experiment. `experiment_results` is the folder name where the results are going to be stored, 1234 is the seed for the experiment, 1 is the number of threads and 4 is the number of iterations performed per thread per algorithm. The previous mentioned config file has to be stored in the folder experiments/ and contains parameters about:

1. the experiment itself (storage path, which benchmark, speedup function etc.),
2. the environment parameters,
3. the behavior/baseline policy parameters and
4. the algorithms and their hyper-parameters.

## References

[1] P. Scholl. *Evaluation of Safe Policy Improvement with Soft Baseline Bootstrapping*. Master's thesis. Technical University of Munich. Germany. 2021.

[2] K. Nadjahi, R. Laroche, R. Tachet des Combes. *Safe
			Policy Improvement with Soft Baseline Bootstrapping*. Proceedings of the 2019
		European Conference on Machine Learning and Principles and Practice of Knowledge
		Discovery in Databases (ECML-PKDD). 2019.
		