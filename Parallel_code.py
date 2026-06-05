#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:47:37 2024

@author: allabou
"""
import os
import shutil
import multiprocessing
from multiprocessing import Manager, Pool
import logging
import numpy as np
import dassflow2d as df2d

class DassFlowRunner:
    def __init__(self, dassflow_dir, code_dir, save_dir):
        self.dassflow_dir = dassflow_dir
        self.code_dir = code_dir
        self.save_dir = save_dir
        self.Qin1 = None
        self.Qin2 = None

    def set_qin_data(self, Qin1, Qin2):
        self.Qin1 = Qin1
        self.Qin2 = Qin2

    def copy_case_files(self, bin_dir):
        os.system(f"cp -r {self.dassflow_dir}/cases/dev_case/Aude/built_bin/* {bin_dir}")

    def initialize_model(self, bin_dir):
        my_model = df2d.dassflowmodel(bin_dir=bin_dir, hdf5_path=f"{bin_dir}/res/simu.hdf5", run_type="direct", clean=True)
        my_model.init_all()
        return my_model

    def one_run(self, dassflow_dir, bin_dir, run_type, ts, dtw, qin1, qin2, id_experiment, save_dir, result_queue):
        os.system(f"cp -r {dassflow_dir}/cases/dev_case/Aude/built_bin/* {bin_dir}")
        my_model = df2d.dassflowmodel(bin_dir=bin_dir, hdf5_path=f"{bin_dir}/res/simu.hdf5", run_type=run_type, clean=True)
        my_model.init_all()
        my_bc = df2d.wrapping.m_model.get_bc()
        my_bc.hyd[1].q = qin1[:, id_experiment]
        my_bc.hyd[2].q = qin2[:, id_experiment]
        my_model.run()
        my_model.save_all()
        if not os.path.exists(f"{save_dir}/{id_experiment}"):
            os.mkdir(f"{save_dir}/{id_experiment}")
        os.system(f"cp {bin_dir}/res/simu.hdf5 {save_dir}/{id_experiment}")
        df2d.wrapping.call_model.clean_model(my_model.kernel)

    def parallel_run(self, start_index, end_index, dassflow_dir, bin_dir, run_type, ts, dtw, save_dir, result_queue):
        for id_experiment in range(start_index, end_index):
            self.one_run(dassflow_dir, bin_dir, run_type, ts, dtw, self.Qin1, self.Qin2, id_experiment, save_dir, result_queue)

    def run_simulations(self, num_processes, run_type, ts, dtw):
        if self.Qin1 is None or self.Qin2 is None:
            raise ValueError("Qin1 and Qin2 must be set before running simulations.")

        manager = Manager()
        result_queue = manager.Queue()
        experiments_per_process = len(self.Qin1[0, :]) // num_processes
        pool = Pool()
        processes = []

        for i in range(num_processes):
            bin_dir = os.path.join(self.code_dir, f"bin_dir_{i}")
            if not os.path.exists(bin_dir):
                os.makedirs(bin_dir)
            start_index = i * experiments_per_process
            end_index = start_index + experiments_per_process if i < num_processes - 1 else len(self.Qin1[0, :])
            process = pool.apply_async(self.parallel_run, args=(start_index, end_index, self.dassflow_dir, bin_dir, run_type, ts, dtw, self.save_dir, result_queue))
            processes.append(process)

        while not result_queue.empty():
            file_path = result_queue.get()
            shutil.copy(file_path, self.save_dir)

        for process in processes:
            process.get()

        pool.close()
        pool.join()

