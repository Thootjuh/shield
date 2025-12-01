This repository was adapted from https://github.com/Philipp238/Safe-Policy-Improvement-Approaches-on-Discrete-Markov-Decision-Processes/tree/master, which contains the code accompanying the paper "Safe Policy Improvement Approaches and Their Limitations" by Philipp Scholl, Felix Dietrich, Clemens Otte, and Steffen Udluft.

## Requirements
The code is implemented in Python version 3.10 and requires the packages specified in ``requirements.txt``. Additionally [Storm](https://www.stormchecker.org/documentation/obtain-storm/build.html) and [Stormpy](https://github.com/moves-rwth/stormpy) are required. 
The experiments where performed using Storm and Stormpy version 1.9.0, which also requires you to manually install [Pycarl](https://moves-rwth.github.io/pycarl/index.html).
Some of the included bechmarks also require stormvogel, which can be found at https://moves-rwth.github.io/stormvogel/index.html

Before running the experiments, you have to create a file named `paths.ini`  which contains the following:
````
[PATHS]
results_path = D:\results
````
Where results_path should be the absolute path pointing to the place where the results should be stored.

## Structure
To run the experiment on the randomMDPs benchmark, one can use 
 
`python run_experiments.py wet_chicken_shield.ini wet_chicken_results 1234 1 1000` 

Where `wet_chicken_shield.ini` is the name of the config file used for this experiment. `experiment_results` is the folder name where the results are going to be stored, `1234` is the seed for the experiment, `1` is the number of threads and `1000` is the number of iterations performed per thread per algorithm. The previously mentioned config file has to be stored in the folder experiments/ and contains parameters about:

1. the experiment itself (storage path, which benchmark, speedup function etc.),
2. the environment parameters,
3. the behavior/baseline policy parameters and
4. the algorithms and their hyper-parameters.

To run the experiment on the Wet Chicken, Pacman or Frozen Lake benchmarks, you can use

`python run_experiments.py random_mdps_shield.ini experiment_results 1234 1 1000` 

or

`python run_experiments.py pacman_shield.ini wet_chicken_results 1234 1 1000` 

or

`python run_experiments.py frozen_lake_shield.ini wet_chicken_results 1234 1 1000` 

respectively.

## Plotting
The files in the `plot_functions` folder can be used to plot the data.
To plot the results of an experiment, you can run:

`python plot_functions/plot.py path/to/results`

This creates the following plots:

1. `results_all.png`, which plots the shielded and non-shielded data, as well as the corresponding CVaR data, for all methods, alongside the  baseline , optimal, and shielded baseline performance

Additionally, for all methods, it creates the plots:

2. `results_method.png`, which which plots the shielded and non-shielded performance of the method, alongside the alongside the  baseline , optimal, and shielded baseline performance
3. `results_method_cvar.png`, which plots the cvar data, alongside the  baseline , optimal, and shielded baseline performance
4. `results_method_interval.png`, which plots the 95% confidence interval of the data, alongside the  baseline , optimal, and shielded baseline performance

## References

[1] P. Scholl, F. Dietrich, C. Otte, and S. Udluft. Safe policy improvement approaches and their limitations. CoRR, abs/2208.00724, 2022.
		
