# The-EA-POD-NN-method
Code developed as part of my thesis on the EA-POD-NN method.

This repository contains the code developed for my thesis work on the EA-POD-NN method.

## Description

This project contains the code developped for the reduction of the txo dimensional shallow water equations. The methode develloped here is a non-intusive method, since it uses the high fidelity solver (Dassflow here) only to generate the snapshots. the the training and online phase are completly decoupled from the pdf and the solver. The code is divided into several parts:
### EA-POD-NN-REDUCTION-METHOD.py 

The main code which call all classes and modules from other codes, it contains all instructions to un all the steps of the EA-POD-NN method from the parameter generation to the results of comparison between the high fidelity solution and the reduced one. 

### parameter_generator.py 
Construct the parameter space and store them in hdf5 file to be used lated for the Dassflow solver, the code also generate the input parameters for EA-POD-NN reduction method.

### Parallel_code.py
Construct the high fidelity simulations in parallel using the finite volumes based solver (Dassflow2d).

### POD_Basis.py
Construct the POD reduced basis $\mathbf{B}\_{rb}$ using the generated high fidelity snapshots $\mathbf{S}$.
 
### POD_Basis_error.py

Construct the POD reduced basis $\mathbf{B}_{er}$ of the error matrix $\mathbf{E}$, computed from a given reduced basis $\mathbf{B}_{rb}$ and the high-fidelity snapshot matrix $\mathbf{S}$.

The error matrix is defined as:

$$
\mathbf{E} = \mathbf{S} - \mathbf{B}_{rb}\mathbf{B}_{rb}^{T}\mathbf{S}
$$

where $\mathbf{B}_{rb}\mathbf{B}_{rb}^{T}\mathbf{S}$ is the projection of $\mathbf{S}$ onto the reduced basis $\mathbf{B}_{rb}$.
### EA_POD-NN_offline.py
### EA_POD-NN_online.py
### Aude_test_case_online.py

## Requirements

List of required Python packages or software.

## Usage

Instructions to run the code.

## Author

Mustapha Allabou
