# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""
import warnings
import tensorflow as tf
import numpy as np

warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import time
from Rota5.inference_utils import InputReader, get_ensemble_ouput, output_results

from Rota5.network.my_model import Model
from Rota5.network.pre_trained_embedding import Settings

def run_Rota5(preparation_config):

    #==================================Model===================================
    start_time = time.time()
    print ("Run OPUS-Rota5...")
    
    start_time = time.time()
    test_reader = InputReader(data_list=preparation_config["filenames"], 
                              preparation_config=preparation_config)

    #============================Parameters====================================
    params = {}
    params["n_1d_feat"] = 256
    params["n_2d_feat"] = 128
    params["max_relative_distance"] = 32
    params["evofomer_config"] = Settings.CONFIG
    params["n_cycle"] = 3
    params["save_path"] = "./Rota5/models"
    #============================Models====================================
    model_rota1 = Model(model_params=params)
    model_rota1.load_model(test_reader, name="rota5_1.h5")

    model_rota2 = Model(model_params=params)
    model_rota2.load_model(test_reader, name="rota5_2.h5")
    
    model_rota3 = Model(model_params=params)
    model_rota3.load_model(test_reader, name="rota5_3.h5")
    
    models = [model_rota1, model_rota2, model_rota3]
    
    for step, filenames_batch in enumerate(test_reader.dataset):

        filenames, x, x_trr, sc_masks, L = \
            test_reader.read_file_from_disk(filenames_batch)
        
        residue_index = np.array([range(L)])
        
        rota_predictions = []
        rmsd_predictions = []
        for model in models:
            if L > 512:
                n = L // 512 + 1
                rota_predictions_ = []
                rmsd_predictions_ = []
                for i in range(n):
                    residue_index_ = residue_index[:,i*512:(i+1)*512]
                    L_ = residue_index_.shape[1]
                    rota_prediction_, rmsd_prediction_ = model(x[:,i*512:(i+1)*512,:], x_trr[:,i*512:(i+1)*512,i*512:(i+1)*512,:], sc_masks[:,i*512:(i+1)*512,:], 
                                                               residue_index_, L_, training=False)  
                    rota_predictions_.append(rota_prediction_)
                    rmsd_predictions_.append(rmsd_prediction_)
                    
                rota_prediction = tf.concat(rota_predictions_, 0)
                rmsd_prediction = tf.concat(rmsd_predictions_, 0)
            else:
                rota_prediction, rmsd_prediction = model(x, x_trr, sc_masks,
                                                         residue_index, L, training=False)   
            
            rota_predictions.append(rota_prediction)
            rmsd_predictions.append(rmsd_prediction)

        x1_outputs, x2_outputs, x3_outputs, x4_outputs, rmsd_outputs, std_outputs = \
            get_ensemble_ouput(rota_predictions, rmsd_predictions, L)    
        
        output_results(filenames[0], x1_outputs, x2_outputs, x3_outputs, x4_outputs, rmsd_outputs, std_outputs, preparation_config)
        
    run_time = time.time() - start_time
    print('OPUS-Rota5 done..., time: %3.3f' % (run_time)) 
    #==================================Model===================================
    
    
    