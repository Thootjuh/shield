This repository was adapted from https://github.com/Philipp238/Safe-Policy-Improvement-Approaches-on-Discrete-Markov-Decision-Processes/tree/master, which contains the code accompanying the paper "Safe Policy Improvement Approaches and Their Limitations" by Philipp Scholl, Felix Dietrich, Clemens Otte, and Steffen Udluft[1].

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
To run the experiment on the cart pole environment, one can use 
 
`python run_experiments.py cart_pole_shield.ini cart_pole_results 1234 1 100` 

Where `cart_pole_shield.ini` is the name of the config file used for this experiment. `cart_pole_results` is the folder name where the results are going to be stored, `1234` is the seed for the experiment, `1` is the number of threads and `100` is the number of iterations performed per thread per algorithm. The previously mentioned config file has to be stored in the folder experiments/ and contains parameters about:

1. the experiment itself (storage path, which benchmark, speedup function etc.),
2. the environment parameters,
3. the behavior/baseline policy parameters and
4. the algorithms and their hyper-parameters.

To run the experiment on the frozen lake, moving obstacles or lunar lander benchmarks, you can use

`python run_experiments.py frozen_lake_shield.ini frozen_lake_results 1234 1 100` 

or

`python run_experiments.py moving_obstacles_shield.ini moving_obstacles_results 1234 1 100` 

or

`python run_experiments.py lunar_lander_shield.ini lunar_lander_results 1234 1 100` 

respectively.

## References

[1] P. Scholl, F. Dietrich, C. Otte, and S. Udluft. Safe policy improvement approaches and their limitations. CoRR, abs/2208.00724, 2022.
		
