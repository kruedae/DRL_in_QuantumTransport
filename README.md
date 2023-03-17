# Extracting data from external sources
[![GitHub License](https://img.shields.io/github/license/jupyter-guide/ten-rules-jupyter.svg)](https://github.com/sbl-sdsc/mmtf-spark/blob/master/LICENSE)

This repository is a supplement to the understand how to set up in the simplest way a repository for use a Jupyter notebook

### Installation and dependencies

Download the GitHub repository and follow the next step befor go to run the notebooks. 

*Some dependencies:* 

Create a conda environment 

```conda create --name drl_env```

Activate the environment

```conda activate drl_env```

Install de dependencies. Just use the following command. Be sure you have Pip installed in your enviroment

```pip install -r requirements.txt```

## 
This notebook uses two implementations of a deep reinforcement algorithm in a coherent transport problem over a system of coupled quantum dots.
First one is the [Baseline](https://github.com/hill-a/stable-baselines) implentation, from an external library which is the one that performs the better(see notebooks folders). The second one is a proper implementation using the Keras library.

This work is based in a previous [paper](https://doi.org/10.1038/s42005-019-0169-x)

---

## Contact Us
If you encounter any problems with this repository, please report them [here](https://github.com/kruedae/DRL_in_QuantumTransport.git/issues).
