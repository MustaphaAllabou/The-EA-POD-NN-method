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
Construct the POD reduced basis $\mathbf{B}\_{rb}$ using the generated high fidelity snapshots **S**.
 
### POD_Basis_error.py

Construct the POD reduced basis **B_er** of the error matrix **E**, computed from a given reduced basis **B_rb** and the high-fidelity snapshot matrix **S**.

The error matrix is defined as:

````markdown
The error matrix is defined as:

```text
E = S - B_rb B_rb^T S
````

where **B_rb B_rb^T S** is the projection of **S** onto the reduced basis **B_rb**.

### EA_POD-NN_offline.py

Uses the snapshots and errors projected onto their corresponding reduced spaces, and trains the neural networks to learn the reduced coefficients of the snapshots and errors:

```text
S_rb = B_rb^T S
S_er = B_er^T E
```

where **S_rb** and **S_er** are the reduced representations of the snapshots and errors, respectively.

### EA_POD-NN_online.py

Once the offline phase is completed and the neural networks are optimized, this code predicts the high-fidelity solution **u_h(mu_new)** for a new parameter value **mu_new**.

### Aude_test_case_online.py

Uses the high-fidelity solver to compute the reference solution **u_h(mu_new)** and compares it with the reduced-order prediction **u_rb(mu_new)**. The code also plots the difference between both solutions, the error histograms, and the mean averaged errors as a function of time.

```
```

## Requirements

List of required Python packages or software.

## Usage

Instructions to run the code.

## Author

Mustapha Allabou
