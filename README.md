# Neural-Stochastic-Control
This repository contains the code for the paper: Neural Stochastic Control
## Requirements
[Pytorch 1.8.1](https://pytorch.org/get-started/locally/)

[hessian (from mariogeiger)](https://github.com/mariogeiger/hessian)

[pylustrator](https://pylustrator.readthedocs.io/en/latest/)
## How it works
Each example consists of a learner of ES or/and AS and a generater. The learner minimizes the corresponding loss function to find optimal parameters in neural stochastic control function (and neural Lyapunov function). The generater sample trajectories under controlled SDEs and calculate corresponding indexes of control performance to check whether the trajectories are steered to targets. The data is provided in the [Google Drive](https://drive.google.com/file/d/1Reo_KysBPqjieAoyXEEgF3WTTtHVqr-S/view?usp=sharing).
## A typical procedure is as follows:
* Define the neural network with initial parameters for neural control function (and Lyapunov function) for AS(ES)
* Define the controlled dynamics for physical system 
* Compute loss function and train the parameters (Run `AS.py`, `ES_ICNN.py`, `ES_Quadratic.py`)
* Generate the trajectories for physical systems without and with control (Run `generate.py`) 
* Plot the corresponding results (Run `plot.py`)



