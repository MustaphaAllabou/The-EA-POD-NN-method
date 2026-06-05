#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:11:57 2024

@author: allabou
"""
import numpy as np
import matplotlib.pyplot as plt
from parameter_generator import HydrographGenerator
from Parallel_code import DassFlowRunner
import shutil
import multiprocessing
from multiprocessing import Manager, Pool
import logging
import os
from POD_Basis import PODProcessor
from POD_Basis_error import PODProcessor_error
from EA_POD_NN_offline import POD_NN_METHOD
from EA_POD_NN_online import POD_NN_PREDICTION
from Aude_test_case_online_jupyter import DassflowSimulation

# #
# import tensorflow as tf
# import tensorflow.keras as k
# from tensorflow.keras.regularizers import l2 
# from tensorflow.keras.datasets import mnist
# import tensorflow.keras.preprocessing.image as kpi
# import tensorflow.keras.models as km
# import tensorflow.keras.layers as kl
# import tensorflow.keras.losses as kloss
# import tensorflow.keras.regularizers as kr
# import tensorflow.keras.backend as K
# import tensorflow.keras.utils as ku
# from tensorflow.keras import callbacks
# from tensorflow.keras.models import load_model

#%%
# Hydrograph Generator Parameters
hydrograph_params = {
    'constraint_values': ([31200., 1100., 30200., 300.], 0.035),
    'mu1_range': np.linspace(75000, 100000, 5),
    'mu2_range': np.linspace(600, 1200, 5),
    'mu3_range': np.linspace(50000, 80000, 5),
    'mu4_range': np.linspace(100, 400, 5),
    'tmontee1': 46800.,
    'tmontee2': 46800.,
    'param': np.array([31200., 1100., 30200., 300.]),
    'n':48,
    
}

pod_params_er = {
    'epsilon_er': 1e-1,
}       


pod_params = {
    'M': 10,
    'Nt': 500,
    'Nh': 7267,
    'epsilon_POD': 1e-3,
    'data_directory': '/home-local/allabou/Bureau/These/Code/Code_reduction/model_outputs/'
} 
  
# Definte the parameters for the Neural Network (NN)
my_nn_params = {
        'input_dim':None,
        'Nrb': None,
        'Ner': None,
        'Neurons': 20,
        'Neurons_er': None,
        'epochs': 10,
        'batch_size': 10,
        'lr': 0.001,
        'alpha': 0.2,
        'alpha1': 0.1
}

# DassFlowRunner Parameters
dassflow_params = {
    'run_type': "direct",
    'ts': 216000.,
    'dtw': 2160.,
    'Qin1': None,
    'Qin2': None,
    'dassflow_dir': "/home-local/allabou/Bureau/dasshydro-main",
    'save_dir': "/home-local/allabou/Bureau/These/Code/Code_reduction/model_outputs",
    'code_dir': "/home-local/allabou/Bureau/dasshydro-main/code/",
    'bin_dir': "/home-local/allabou/Bureau/dasshydro-main/code/bin_A/",
}
#%%
if __name__ == "__main__":
   
    ###########################################################################
    # Parameter space construction
    ###########################################################################
    
    # Construct the parameter space using HydrographGenerator
    hydrograph_generator = HydrographGenerator()
    
    # Check the parameter constraint (phase-shofted with 3.5%)
    hydrograph_generator.check_constraint(*hydrograph_params['constraint_values'])
    
    # Construct the parameter space
    parameter_space = hydrograph_generator.generate_parameter_space(
        hydrograph_params['mu1_range'],
        hydrograph_params['mu2_range'],
        hydrograph_params['mu3_range'],
        hydrograph_params['mu4_range'],
        hydrograph_params['constraint_values'][1]
    )
    echantillons = np.array(parameter_space)
    
    # Plot the parameter space
    hydrograph_generator.plot_parameter_space(
        echantillons,
        hydrograph_params['param'],
        hydrograph_params['tmontee1'],
        hydrograph_params['tmontee2']
    )

    # Define the hydrographs numpy vectors
    Qin1, Qin2 = hydrograph_generator.generate_hydrographs(echantillons, hydrograph_params['tmontee1'], hydrograph_params['tmontee2'])
   
    # Update the dictionary with the desired values
    dassflow_params['Qin1'] = Qin1
    dassflow_params['Qin2'] = Qin2
    
    # Construct the parameter set for the EA-POD-NN method
    POD_NN_Parameters = hydrograph_generator.generate_pod_nn_parameters(echantillons, hydrograph_params['tmontee1'], hydrograph_params['tmontee2'])

    # Update the dictionary with the desired values
    my_nn_params['input_dim'] = len(POD_NN_Parameters[0])
#%%    
    # ##########################################################################
    # # Snapshots computation
    # ##########################################################################
    
    # # Configure the logging system to output log messages with a severity level of DEBUG
    # logging.basicConfig(level=logging.DEBUG)
    # print(os.path.abspath("."))
    
    # # Define the DassFlow class with corresponding paths
    # dassflow_runner = DassFlowRunner(
    #     dassflow_params['dassflow_dir'],
    #     dassflow_params['code_dir'],
    #     dassflow_params['save_dir']
    # )
    
    # # Copy the case files
    # dassflow_runner.copy_case_files(os.path.join(dassflow_params['code_dir'], "bin_A"))
    
    # # Initialize the model
    # dassflow_runner.initialize_model(os.path.join(dassflow_params['code_dir'], "bin_A"))
    
    # # Set the hydrographs vectors
    # dassflow_runner.set_qin_data(dassflow_params['Qin1'], dassflow_params['Qin2'])
    
    # # Run the DassFlow simulation in parallel
    # dassflow_runner.run_simulations(16, dassflow_params['run_type'], dassflow_params['ts'], dassflow_params['dtw'])

#%%    
    ###########################################################################
    # Reduced basis Brb and Reduced basis Ber computation
    ###########################################################################
    
    # Compute the RB matrix Brb
    
    # Initialize the POD processor
    pod_processor = PODProcessor(**pod_params)
    
    # Read the hdf5 files and construct the snapshots matrix
    Snapshots = pod_processor.read_data_files()
    print('Snapshots shaps is',Snapshots.shape)
    
    # Comute the eigen values of the correlation matrix C
    pod_processor.compute_eigenvalues()
    
    # Compute and plot the eigenvalues and eigenvectors of C
    pod_processor.plot_eigenvalues()
    
    # Truncate the first Nrb modes
    Nrb = pod_processor.compute_number_of_modes(pod_params['epsilon_POD'])
    # Update the dictionary with the desired values
    my_nn_params['Nrb'] = Nrb
        
    
    # Verify the error estimtion of POD method
    # pod_processor.compute_error_estimation()  # Uncomment this line if needed
    
    # Get the RB matrix
    Brb = pod_processor.truncate_reduced_basis(Nrb)
    
    # Compute the RB matrix Ber
    
    # Initialize the POD  error processor
    pod_processor_er = PODProcessor_error(**pod_params_er)
    
    # Comute the eigen values of the correlation matrix C
    pod_processor_er.compute_eigenvalues_er(Snapshots,Nrb,Brb)
    
    # Compute and plot the eigenvalues and eigenvectors of C
    pod_processor_er.plot_eigenvalues_er() 
    
    # Truncate the first Nrb modes
    Ner = pod_processor_er.compute_number_of_modes_er(pod_params_er['epsilon_er'])

    # Update the dictionary with the desired values
    my_nn_params['Ner'] = Ner
    my_nn_params['Neurons_er'] = 2 * Ner
    
    # Get the RB matrix
    Ber = pod_processor_er.truncate_reduced_basis_er(Ner)

#%%    
    ###########################################################################
    # EA-POD-NN method: offline phase 
    ###########################################################################
    
    # Definte the parameters for the Neural Network (NN)
    
    # Initialize the NN parameters
    my_nn = POD_NN_METHOD(**my_nn_params)
    
    # Load the data
    my_nn.load_data()
    
    # Normalize the data
    my_nn.Urb_POD,Urb_POD_max,Urb_POD_min = my_nn.normalize_data(my_nn.Urb_POD)
    my_nn.Urb_POD_er,Urb_POD_er_max,Urb_POD_er_min = my_nn.normalize_data(my_nn.Urb_POD_er)
    my_nn.P,P_max,P_min = my_nn.normalize_data(my_nn.P)
    
    # Split data into train, test and validation parts
    my_nn.split_data()
    
    # Train the NN 
    history,history_er = my_nn.train_models()
    #history_er = my_nn.train_model_er()
    
    # Evaluate the NN model on the validation data
    my_nn.evaluate_model()
    
    # Plot the loss function of the NN 
    my_nn.plot_loss(history,history_er)
    
    # Save the optimized NN parameters
    my_nn.save_model('Neural-network-1','Neural-network-2')

#%%    
    ###########################################################################
    # EA-POD-NN method: online phase
    ###########################################################################
    
    # Computation of the hydrographs to simulate the reference solution
    hydrograph_generator.generate_hydrograph_online(np.zeros(hydrograph_params['n']), np.zeros(hydrograph_params['n']),
                                                    hydrograph_generator.Time, hydrograph_params['tmontee1'], hydrograph_params['param'])
    

    # Instantiate the class with required parameters
    pod_nn_prediction = POD_NN_PREDICTION(my_nn.model,my_nn.model_er, Brb, Ber, P_max,P_min,
                                          Urb_POD_max,Urb_POD_min,Urb_POD_er_max,Urb_POD_er_min)

    # Use the method to predict solution
    h_POD_NN, norm_vel_POD_NN = pod_nn_prediction.predict_solution(my_nn, hydrograph_params['param'],
                                                                   hydrograph_generator.Time_simu,len(Brb))
    
#%%
    
    ###########################################################################
    # Compute the reference solution for the online phase
    ###########################################################################
    
    # Initialise the simulation
    simulation = DassflowSimulation()
    
    # Load the inflows vectors
    simulation.load_inflow('Q_1.npy', 'Q_2.npy')
    
    # Setup the directories for the simulation
    simulation.setup_directories("/home-local/allabou/Bureau/dasshydro-main")
    
    # Initialize the dassFlow model
    simulation.initialize_model()
    
    # Get the boundary conditions
    simulation.get_boundary_conditions()
    
    # Run the simulation
    simulation.run_simulation()
    
    # Get the model outputs: h the water depth and nor_vel the velocity norm
    h, norm_vel = simulation.get_output('/home-local/allabou/Bureau/These/Code/Code_reduction/Dassflow_sol.npy')
    
    # Plot the bathymetry
    simulation.plot_bathy()
    
    # Compute the absoolute error between the reference solution and EA-POD-NN one
    Vel_error = simulation.error_computation(norm_vel, norm_vel_POD_NN)
    h_error = simulation.error_computation(h, h_POD_NN)
    
    # Plot the velocity norm outputs
    simulation.plot_velocity_outputs(norm_vel[:, -1], norm_vel_POD_NN[:, -1], Vel_error[:, -1])
    
    # Plot the water depth outputs
    simulation.plot_water_depth_outputs(h[:, -1], h_POD_NN[:, -1], h_error[:, -1])
    
    # Plot the histograme of the error on the water depth and velocity norm
    simulation.plot_error_histogram(h[:, -1], h_error[:, -1], norm_vel[:, -1], Vel_error[:, -1], np.linspace(0, 7, 7), np.linspace(0, 7, 6))

    # Compute and plot the Cummulative Distribution Function (CDF)
    simulation.plot_cdf(h_error[:, -1], Vel_error[:, -1])
    
    # Compute the means error in function of time
    simulation.plot_mean_errors(h_error, Vel_error)