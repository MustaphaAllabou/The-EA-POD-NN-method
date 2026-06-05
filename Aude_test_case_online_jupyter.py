#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:50:40 2024

@author: allabou
"""

import dassflow2d as df2d
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import os
import h5py
import time
from scipy.stats import norm

class DassflowSimulation:
    def __init__(self):
        pass

    def load_inflow(self, file1, file2):
        self.Qin1 = np.load(file1)
        self.Qin2 = np.load(file2)
        self.n = len(self.Qin1)
        print(self.Qin1.shape)
        print(os.path.abspath("."))

    def setup_directories(self, dassflow_dir):
        self.dassflow_dir = dassflow_dir
        self.code_dir = f"{dassflow_dir}/code/"
        self.bin_dir = f"{self.code_dir}/bin_A/"
        print(f"Printing case files \n from {dassflow_dir}/cases/dev_case/Aude/built_bin  \n  to  {dassflow_dir}/code/bin_A ")
        os.system(f"rm -r {dassflow_dir}/code/bin_A/*")
        os.system(f"cp -r {dassflow_dir}/cases/dev_case/Aude/built_bin/* {dassflow_dir}/code/bin_A")
        os.chdir(f"{dassflow_dir}/code/")
        os.system("make cleanres cleanmin")
        if os.path.exists(f" {dassflow_dir}/code/bin_A/restart.bin"):
            os.system(f"rm {dassflow_dir}/code/bin_A/restart.bin")
        os.chdir(self.bin_dir)


    def initialize_model(self):
        self.my_model = df2d.dassflowmodel(bin_dir=self.bin_dir, hdf5_path=f"{self.dassflow_dir}/code/bin_A/res/simu.hdf5", run_type="direct", clean=True)
        print(self.my_model.config.get())
        self.my_model.init_all()

    def get_boundary_conditions(self):
        self.my_bc = df2d.wrapping.m_model.get_bc()
        print("The first hydrograph model bc.q",self.my_model.boundary.bc.hyd[0].q)
        print("The first hydrograph model bc.q",self.my_model.boundary.bc.hyd[1].q)
        print("The first hydrograph model bc.q",self.my_model.boundary.bc.hyd[2].q)
        print("The first hydrograph model bc.t",self.my_model.boundary.bc.hyd[0].t)
        print("The first hydrograph model bc.t",self.my_model.boundary.bc.hyd[1].t)
        print("The first hydrograph model bc.t",self.my_model.boundary.bc.hyd[2].t)
        c = self.my_model.boundary.bc.hyd[0].q
        print(c.shape)
        print("The first hydrograph model bc.t",self.my_model.boundary.bc.hyd[2].t)

    def run_simulation(self):
        debut = time.time()
        self.my_model.run()
        fin = time.time()
        duree = fin - debut
        print("Time of simulation by DassFlow method is:", duree, "secondes")
        self.my_model.save_all()
        df2d.wrapping.call_model.clean_model(self.my_model.kernel)

    def save_results(self, save_path):
        Dassflow_sol = np.concatenate((self.my_model.outputs.result.h, self.my_model.outputs.result.u, self.my_model.outputs.result.v))
        print(Dassflow_sol.shape)
        np.save(save_path, Dassflow_sol)
        
    def get_output(self, save_path):
        
        """
        Retrieves water depth (h), horizontal velocity component (u), and vertical velocity component (v) from the model output.
        
        Args:
        - model: The model object containing the output data.
        
        Returns:
        - h: Water depth.
        - u: Horizontal velocity component.
        - v: Vertical velocity component.
        """
        h = self.my_model.outputs.result.h
        u = self.my_model.outputs.result.u
        v = self.my_model.outputs.result.v
        
        # Slicing the arrays to remove unnecessary columns
        h = h[:, 12:]
        u = u[:, 12:]
        v = v[:, 12:]
        norm_vel = np.sqrt(u**2+v**2)
        
        Dassflow_sol = np.concatenate((h,u,v))
        print(Dassflow_sol.shape)
        np.save(save_path, Dassflow_sol)
        return h, norm_vel

    def load_pod_nn_results(self, file_path):
        os.chdir(file_path)
        self.Uh_POD_NN = np.load('Uh_POD_NN_All.npy')
        print('Uh_POD_NN shape', self.Uh_POD_NN.shape)
        self.h_POD_NN = self.Uh_POD_NN[:7267,:]
        self.u_POD_NN = self.Uh_POD_NN[7267:7267*2,:]
        self.v_POD_NN = self.Uh_POD_NN[7267*2:,:]
        self.norm_vel_POD_NN = np.sqrt(self.u_POD_NN**2 + self.v_POD_NN**2)
        
    # Function to plot the velocity norm outputs
    def plot_velocity_outputs(self, norm_vel, norm_vel_POD_NN, Vel_error):
        
        plotter = self.my_model.outputs.result.plot_field(my_mesh=self.my_model.meshing.mesh_pyvista,
                                                 my_scalar=norm_vel,
                                                 title_scale_bar="                    U [m/s]",
                                                 title_plot="",
                                                 notebook=True)
        plotter.update_scalar_bar_range([np.amin(norm_vel), np.amax(norm_vel)])
        self.adjust_plotter_camera(plotter)
        plotter.show(jupyter_backend='trame')

        plotter = self.my_model.outputs.result.plot_field(my_mesh=self.my_model.meshing.mesh_pyvista,
                                                 my_scalar=norm_vel_POD_NN,
                                                 title_scale_bar="                                         U_EA_POD-NN [m/s] ",
                                                 title_plot="",
                                                 notebook=True)
        plotter.update_scalar_bar_range([np.amin(norm_vel), np.amax(norm_vel)])
        self.adjust_plotter_camera(plotter)
        plotter.show(jupyter_backend='trame')
    
        plotter = self.my_model.outputs.result.plot_field(my_mesh=self.my_model.meshing.mesh_pyvista,
                                                 my_scalar=Vel_error,
                                                 title_scale_bar="                          Error [m/s] ",
                                                 title_plot="",
                                                 notebook=True)

        plotter.update_scalar_bar_range([np.amin(Vel_error), np.amax(Vel_error)])
        self.adjust_plotter_camera(plotter)
        plotter.show(jupyter_backend='trame')
        
    # Function to plot the water depth output
    def plot_water_depth_outputs(self, h, h_POD_NN, h_error):
        
        plotter = self.my_model.outputs.result.plot_field(my_mesh=self.my_model.meshing.mesh_pyvista,
                                                 my_scalar = h,
                                                 title_scale_bar="                    h [m/s]",
                                                 title_plot="",
                                                 notebook=True)
        plotter.update_scalar_bar_range([np.amin(h), np.amax(h)])
        self.adjust_plotter_camera(plotter)
        plotter.show(jupyter_backend='trame')

        plotter = self.my_model.outputs.result.plot_field(my_mesh=self.my_model.meshing.mesh_pyvista,
                                                 my_scalar=h_POD_NN,
                                                 title_scale_bar="                                         h_EA_POD-NN [m/s] ",
                                                 title_plot="",
                                                 notebook=True)
        plotter.update_scalar_bar_range([np.amin(h), np.amax(h)])
        self.adjust_plotter_camera(plotter)
        plotter.show(jupyter_backend='trame')
    
        plotter = self.my_model.outputs.result.plot_field(my_mesh=self.my_model.meshing.mesh_pyvista,
                                                 my_scalar=h_error,
                                                 title_scale_bar="                          Error [m/s] ",
                                                 title_plot="",
                                                 notebook=True)
        plotter.update_scalar_bar_range([np.amin(h_error), np.amax(h_error)])
        self.adjust_plotter_camera(plotter)
        plotter.show(jupyter_backend='trame')

    # Function to zoom on the figures
    def adjust_plotter_camera(self,plotter):
        
        # Adjust the camera position and focal point for zooming
        plotter.viewup = [0, 0, 1]  # Set the view from the top (you can adjust as needed)
        
        # Assuming plotter.camera_position is a single value or tuple
        if isinstance(plotter.camera_position, tuple) and len(plotter.camera_position) == 3:
            # Update the camera position, keeping the first two components unchanged and setting the third component to -1
            plotter.camera_position = (plotter.camera_position[0], plotter.camera_position[1], -1)
            
        # Manually adjust the camera distance for zooming
        plotter.camera.zoom(1.47)
     
    # Function to compute the absolute error
    def error_computation(self, A, B):
        
        error =  A - B
        return error
    
    # Function to plot the bathymety used in the simulation
    def plot_bathy(self):
        
        # plotter = self.my_model.meshing.mesh_pyvista.plot(show_edges=True, cpos= "xy", notebook =True)
        # plotter.show()
        
        plotter = self.my_model.outputs.result.plot_field(my_mesh = self.my_model.meshing.mesh_pyvista,
                                              what = "bathy",
                                              title_plot = "Bathymetry elevation",
                                              notebook =True)
        
        plotter.show(jupyter_backend='trame')
        

    def plot_error_histogram(self, h_vector, h_error_vector, u_vector, Vel_error_vector, interval_bounds_h, interval_bounds_u):
        # Initialiser les listes pour stocker les moyennes et écarts types pour h
        means_h = []
        std_devs_h = []
        
        # Boucle sur les intervalles pour h
        for i in range(len(interval_bounds_h) - 1):
            a, b = interval_bounds_h[i], interval_bounds_h[i + 1]
            
            # Trouver les indices où h est compris entre [a, b]
            indices_h = np.where((h_vector >= a) & (h_vector <= b))[0]
            
            # Sélectionner les valeurs correspondantes dans h_error
            selected_h_error = h_error_vector[indices_h]
            
            # Calculer la moyenne et l'écart type pour h
            mean_value_h = np.mean(selected_h_error)
            std_dev_value_h = np.std(selected_h_error)
            
            # Ajouter à la liste des moyennes et écarts types pour h
            means_h.append(mean_value_h)
            std_devs_h.append(std_dev_value_h)
            
        # Initialiser les listes pour stocker les moyennes et écarts types pour u
        means_u = []
        std_devs_u = []

        # Boucle sur les intervalles pour u
        for i in range(len(interval_bounds_u) - 1):
            a, b = interval_bounds_u[i], interval_bounds_u[i + 1]
            
            # Trouver les indices où u est compris entre [a, b]
            indices_u = np.where((u_vector >= a) & (u_vector <= b))[0]
            
            # Sélectionner les valeurs correspondantes dans u_error
            selected_u_error = Vel_error_vector[indices_u]
            
            # Calculer la moyenne et l'écart type pour u
            mean_value_u = np.mean(selected_u_error)
            std_dev_value_u = np.std(selected_u_error)

            # Ajouter à la liste des moyennes et écarts types pour u
            means_u.append(mean_value_u)
            std_devs_u.append(std_dev_value_u)

        # Convertir en tableaux NumPy
        means_h = np.array(means_h)
        std_devs_h = np.array(std_devs_h)
        
        means_u = np.array(means_u)
        std_devs_u = np.array(std_devs_u)
        
        # Set the size of the figure
        fig, ax = plt.subplots(figsize=(8, 6))  # Adjust width and height as needed
        
        # Plot
        ax.errorbar(interval_bounds_h[:-1] + 0.5, means_h, yerr=std_devs_h, fmt='none', ecolor='r', capsize=5, label='StD on $|\epsilon_h|$')
        ax.scatter(interval_bounds_h[:-1] + 0.5, means_h, alpha=0.7, color='blue', label='Mean on $|\epsilon_h|$')
        ax.errorbar(interval_bounds_u[:-1] + 0.5, means_u, yerr=std_devs_u, fmt='none', ecolor='green', capsize=5, label='StD on $|\epsilon_U|$')
        ax.scatter(interval_bounds_u[:-1] + 0.5, means_u, alpha=0.7, color='orange', label='Mean on $|\epsilon_U|$')
        ax.set_xlabel('$h$ ($m$) or $U$ ($m.s^{-1}$)', fontsize=18)
        ax.set_ylabel('$|\epsilon_h|$ ($m$) or $|\epsilon_U|$ ($m.s^{-1}$)', fontsize=18)
        ax.legend(loc='upper left', fontsize=10)
        
        # Save or show the plot
        plt.show()



    def plot_cdf(self, h_error, Vel_error):
        # Generate random data from a normal distribution for h
        data_h = np.abs(h_error)
        print('data_h=', data_h.shape)
        np.save('data1_h', data_h)
        
        # Fit the data to a normal distribution (optional, but useful for comparison)
        mu_h, std_h = norm.fit(data_h)
        print('mu_h=', mu_h)
        print('std_h=', std_h)
        
        # Compute the cumulative distribution function (CDF) for each value in data_h
        cdf_h = norm.cdf(data_h, mu_h, std_h)
        print('cdf_h=', cdf_h)
        
        # Sort data and x for h
        sorted_indices_h = np.argsort(data_h)
        sorted_data_h = data_h[sorted_indices_h]
        sorted_cdf_h = cdf_h[sorted_indices_h]
        np.save('sorted_data1_h', sorted_data_h)
        np.save('sorted_cdf1_h', sorted_cdf_h)
        
        # Plot the theoretical CDF for the fitted normal distribution for h (optional)
        plt.plot(sorted_data_h, norm.cdf(sorted_data_h, mu_h, std_h), label='Normal CDF for $|\epsilon_h|$', linestyle='--', color='red')
        
        # Generate random data from a normal distribution for u
        data_u = np.abs(Vel_error)
        print('data_u=', data_u.shape)
        np.save('data1_u', data_u)
        
        # Fit the data to a normal distribution for u (optional, but useful for comparison)
        mu_u, std_u = norm.fit(data_u)
        print('mu_u=', mu_u)
        print('std_u=', std_u)
        
        # Compute the cumulative distribution function (CDF) for each value in data_u
        cdf_u = norm.cdf(data_u, mu_u, std_u)
        print('cdf_u=', cdf_u)
        
        # Sort data and x for u
        sorted_indices_u = np.argsort(data_u)
        sorted_data_u = data_u[sorted_indices_u]
        sorted_cdf_u = cdf_u[sorted_indices_u]
        np.save('sorted_data1_u', sorted_data_u)
        np.save('sorted_cdf1_u', sorted_cdf_u)
        
        # Plot the theoretical CDF for the fitted normal distribution for u (optional)
        plt.plot(sorted_data_u, norm.cdf(sorted_data_u, mu_u, std_u), label='Normal CDF for $|\epsilon_U|$', linestyle='-', color='green')
        
        # Add labels and legend
        plt.xlabel('$|\epsilon_h|$ ($m$) or $|\epsilon_U|$ ($m.s^{-1}$)', fontsize=12)
        plt.ylabel('CDF for $|\epsilon_h|$ ($m$) or $|\epsilon_U|$ ($m.s^{-1}$)', fontsize=12)
        plt.legend()
        #plt.savefig('/home-local/allabou/Bureau/These/Code/2018Floods/CDF_ERROR_h_u_EA_POD_NN_triangle.png')
        # Show the plot
        plt.show()
        print('madiane of h is', np.median(sorted_data_h))
        print('madiane of u is', np.median(sorted_data_u))
        
    def plot_mean_errors(self, h_error, Vel_error):
        # Calculer les erreurs absolues
        Error_h = np.abs(h_error)
        Error_u = np.abs(Vel_error)
        
        # Calculer les moyennes des erreurs pour chaque pas de temps
        MeanErrors_EA_POD_NN_h = np.mean(Error_h, axis=0)  
        MeanErrors_EA_POD_NN_u = np.mean(Error_u, axis=0) 
        
        # Trouver les valeurs maximales d'erreur
        error_max_EA_POD_NN_h = np.max(MeanErrors_EA_POD_NN_h)
        error_max_EA_POD_NN_u = np.max(MeanErrors_EA_POD_NN_u)
        
        # Créer une figure
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Intervalle de temps
        time_interval = np.arange(12960/3600., 216000/3600., ((216000-12960)/len(Error_h[0]))/3600.)
        
        # Tracer les moyennes des erreurs en fonction du temps
        ax.plot(time_interval, MeanErrors_EA_POD_NN_h, label='$|\epsilon_h|$ (EA-POD-NN)', color='orange')
        ax.plot(time_interval, MeanErrors_EA_POD_NN_u, label='$|\epsilon_U|$ (EA-POD-NN)', color='b')
        ax.axhline(y=error_max_EA_POD_NN_h, color='orange', linestyle='--')#, label='Max Error POD-NN')
        ax.axhline(y=error_max_EA_POD_NN_u, color='b', linestyle='--')
        
        # Ajouter des étiquettes et une légende
        ax.set_xlabel('Time ($h$)')
        ax.set_ylabel('Mean on $|\epsilon_h|$ ($m$) or $|\epsilon_U|$ ($m.s^{-1}$)')
        ax.legend(loc='center right',fontsize=10)
        # Ajuster la mise en page pour éviter les chevauchements
        plt.tight_layout()
        # Set y-axis to log scale
        plt.yscale('log')
        # Afficher la figure
        plt.show()
    
        

#%%
# # Exemple d'utilisation :
# if __name__ == "__main__":
#     simulation = DassflowSimulation()
#     simulation.load_inflow('Q_1.npy', 'Q_2.npy')
#     simulation.setup_directories("/home-local/allabou/Bureau/dasshydro-main")
#     simulation.initialize_model()
#     simulation.get_boundary_conditions()
#     simulation.run_simulation()
#     h, norm_vel = simulation.get_output('/home-local/allabou/Bureau/These/Code/Code_reduction/Dassflow_sol.npy')
#     simulation.plot_bathy()
#     print(os.path.abspath("."))
#     os.chdir(f"/home-local/allabou/Bureau/These/Code/Code_reduction/")
#     print(os.path.abspath("."))
#     Uh_POD_NN_All = np.load('Uh_EA_POD_NN.npy')
#     h_POD_NN = Uh_POD_NN_All[:7267,:];u_POD_NN = Uh_POD_NN_All[7267:7267*2,:];v_POD_NN = Uh_POD_NN_All[7267*2:,:]
#     norm_vel_POD_NN = np.sqrt(u_POD_NN**2 + v_POD_NN**2)
#     Vel_error = simulation.error_computation(norm_vel, norm_vel_POD_NN)
#     h_error = simulation.error_computation(h, h_POD_NN)
#     simulation.plot_velocity_outputs(norm_vel[:, -1], norm_vel_POD_NN[:, -1], Vel_error[:, -1])
#     simulation.plot_water_depth_outputs(h[:, -1], h_POD_NN[:, -1], h_error[:, -1])
