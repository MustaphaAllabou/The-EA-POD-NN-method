#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:47:37 2024

@author: allabou
"""

import numpy as np
import os
import h5py
import numpy.linalg as npl
import scipy
import matplotlib.pyplot as plt
from math import atan
from numpy.linalg import eig
import pandas as pd
from scipy.linalg import eigh
import time
from datetime import datetime

class PODProcessor:
    def __init__(self, M=2, Nt=500, Nh=7267, epsilon_POD=None, data_directory=None):
        self.M = M
        self.Nt = Nt
        self.Nh = Nh
        self.step = 500
        self.Ns = Nt * M
        self.H = np.zeros(Nh)
        self.U = np.zeros(Nh)
        self.V = np.zeros(Nh)
        self.FROUD = np.zeros(Nh)
        self.epsilon_POD = epsilon_POD
        self.Snapshots = None
        self.Eigen_val = None
        self.Eigen_vec = None
        self.Brb = None
        self.s_eigen = None  # Nouvel attribut pour stocker s_eigen
        self.data_directory = data_directory

    def read_data_files(self):
        for i in range(self.M):
            # Read the hdf5 files
            f = h5py.File(os.path.join(self.data_directory, str(i), 'simu.hdf5'), 'r')

            # Access a specific group
            group = f['output']

            # Access a specific data set within the group
            dataset = group['result']

            # Reading data from a numpy table
            h = dataset['h']
            u = dataset['u']
            v = dataset['v']

            # Get the water depth
            h = np.array(h)
            h = h[:, 12:]
            #print('h shape is ',h.shape)
            # Get the x direction velocity
            u = np.array(u)
            u = u[:, 12:]

            # Get the y direction velocity
            v = np.array(v)
            v = v[:, 12:]

            # Compute the Froude number
            #g = 10.
            #Froud_nbr = np.sqrt(u**2 + v**2) / np.sqrt(g * h)

            # Concatenate the output variables into matrices
            self.U = np.c_[self.U, u]
            self.V = np.c_[self.V, v]
            self.H = np.c_[self.H, h]
            #self.FROUD = np.c_[self.FROUD, Froud_nbr]

        # Save the data in numpy files
        self.H = self.H[:, 1:]
        self.U = self.U[:, 1:]
        self.V = self.V[:, 1:]
        self.Snapshots = np.concatenate((self.H, self.U, self.V))
        self.Snapshots = self.Snapshots[:, 1:]
        np.save('Snapshots', self.Snapshots)
        #print('snapshots shape is ',self.Snapshots.shape)
        return self.Snapshots


    def compute_eigenvalues(self):
        C = np.dot(self.Snapshots.transpose(), self.Snapshots) # Il faut changer cela pour le cas test !!!!!!!
        print("cond of C=", npl.cond(C, 2))
        start_time = datetime.now()
        self.Eigen_val, self.Eigen_vec = np.linalg.eigh(C)
        Xi = np.zeros((len(self.Snapshots),len(self.Eigen_vec)))
        for i in range(len(self.Eigen_vec)):
            Xi[:,i] = (1/(np.sqrt(self.Eigen_val[i])))*np.dot(self.Snapshots,self.Eigen_vec[:,i])
        self.Eigen_vec = Xi
        self.normalize_eigenvalues()  # Appel à la méthode pour calculer s_eigen
        end_time = datetime.now()
        print('Duration: {}'.format(end_time - start_time))
        a = np.argsort(-self.Eigen_val)
        self.Eigen_val = self.Eigen_val[a]
        self.Eigen_vec = self.Eigen_vec[:, a]
        np.save('Eigen_values', self.Eigen_val)


    def plot_eigenvalues(self):
        a = np.argsort(-self.Eigen_val)
        self.Eigen_val = self.Eigen_val[a]
        Eigen_vec = self.Eigen_vec[:, a]
        
        Eigen_val_normalized = self.normalize_eigenvalues()
        
        Decay = np.arange(len(self.Eigen_val))
        fig = plt.figure(figsize=(12, 8))
        ax = fig.gca()
        ax.plot(Decay, Eigen_val_normalized, label='Normalized eigenvalues', color='r')
        ax.set_xticks(Decay)
        ax.set_xticklabels([r'$\lambda_{%i}$' % (i + 1) for i in Decay], fontsize=20)
        plt.xscale("log")
        plt.yscale("log")
        width = 0.5
        p = plt.bar(Decay, Eigen_val_normalized, width, color='b')
        plt.legend(fontsize=16)
        ax.legend(fontsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        plt.show()
        
    def normalize_eigenvalues(self):
        if self.s_eigen is None:
            self.s_eigen = np.sum(self.Eigen_val)
            print('s_eigen =', self.s_eigen)
        Eigen_val_normalized = (self.Eigen_val) / self.s_eigen
        return Eigen_val_normalized

    def compute_number_of_modes(self, epsilon_POD):
        Nrb = self.energy_number(epsilon_POD)
        print('This corresponds to Nrb = ', Nrb)
        return Nrb
    
    def energy_number(self, epsilon_POD):
        Eigen_val_normalized = self.normalize_eigenvalues()
        s1 = np.sum(Eigen_val_normalized)
        s = 0.
        index_min = 0
        for i in range(len(Eigen_val_normalized)):
            if s < s1 * (1 - epsilon_POD):
                s += Eigen_val_normalized[i]
                index_min += 1
        return index_min

    def truncate_reduced_basis(self, Nrb):
        if self.Brb is None:
            self.Brb = self.Eigen_vec[:, 0:Nrb]
            print("Brb shape is", self.Brb.shape)
            np.save('Brb', self.Brb)
        return self.Brb

    def compute_error_estimation(self):
        Uh = np.zeros(len(self.Snapshots))
        S = 0.

        for m in range(len(self.Snapshots[0])):
            Uh = self.Snapshots[:, m]
            Urb = np.dot(self.Brb, np.dot(self.Brb.transpose(), Uh))
            Diff = Uh - Urb
            S += np.dot(Diff.transpose(), Diff)

        Discarded_eigenvalues = np.sum(self.Eigen_val[len(self.Brb):])
        print("The error estimation by POD method is", abs(S - Discarded_eigenvalues) / abs(self.s_eigen))


# pod_params = {
#     'M': 5,
#     'Nt': 500,
#     'Nh': 7267,
#     'epsilon_POD': 1e-2,
#     'data_directory': '/home-local/allabou/Bureau/These/Code/Code_reduction/model_outputs/'
# }       
# # Compute the RB matrix Brb:
# pod_processor = PODProcessor(**pod_params)
   
# # Read the hdf5 files and construct the snapshots matrix
# pod_processor.read_data_files()
    
# # Comute the correlation matrix C
# pod_processor.compute_eigenvalues()
# # Compute and plot the eigenvalues and eigenvectors of C
# pod_processor.plot_eigenvalues() 
# # Truncate the first Nrb modes
# # Truncate the first Nrb modes
# Nrb = pod_processor.compute_number_of_modes(pod_params['epsilon_POD'])

# # Get the RB matrix
# Brb = pod_processor.truncate_reduced_basis(Nrb)
    
# Verify the error estimtion of POD method
# pod_processor.compute_error_estimation()  # Uncomment this line if needed