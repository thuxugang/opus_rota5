# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""

import os
import warnings
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
import numpy as np
import time
import multiprocessing

from Rota5.inference_utils import mk_fasta_pp, mk_ss, mk_trr130, make_input
from Rota5.unet3d import utils, unet
from Rota5.inference import run_Rota5
from mkpdb.mk_pdb import toPDB

def preparation(multi_iter):
    
    file_path, filename, preparation_config = multi_iter
   
    fasta_filename = filename + '.fasta'
    pp_filename = filename + '.pp'
    if (not os.path.exists(os.path.join(preparation_config["tmp_files_path"], fasta_filename)) or
        not os.path.exists(os.path.join(preparation_config["tmp_files_path"], pp_filename))):
        mk_fasta_pp(file_path, filename, preparation_config)
        
    ss_filename = filename + '.ss'
    if not os.path.exists(os.path.join(preparation_config["tmp_files_path"], ss_filename)):
        mk_ss(file_path, filename, preparation_config)  

    trr130_filename = filename + '.trr130.npy'
    if not os.path.exists(os.path.join(preparation_config["tmp_files_path"], trr130_filename)):
        mk_trr130(file_path, filename, preparation_config)  

    make_input(filename, preparation_config)
        
if __name__ == '__main__':

    #============================Parameters====================================
    list_path = r"./bb_list_casp15"
    files_path = []
    f = open(list_path)
    for i in f.readlines():
        files_path.append(i.strip())
    f.close()
    
    preparation_config = {}
    preparation_config["batch_size"] = 1
    preparation_config["tmp_files_path"] = os.path.join(os.path.abspath('.'), "tmp_files")
    preparation_config["output_path"] = os.path.join(os.path.abspath('.'), "predictions")
    preparation_config["mkdssp_path"] = os.path.join(os.path.abspath('.'), "Rota5/mkdssp/mkdssp")
    
    num_cpu = 56
    
    #============================Parameters====================================
    
    
    #============================Preparation===================================
    print('Preparation start...')
    start_time = time.time()
    
    multi_iters = []
    filenames = []
    for file_path in files_path:
        filename = file_path.split('/')[-1].split('.')[0]
        multi_iters.append([file_path, filename, preparation_config])
        filenames.append(filename)
        
    pool = multiprocessing.Pool(num_cpu)
    pool.map(preparation, multi_iters)
    pool.close()
    pool.join()

    preparation_config["filenames"] = filenames
        
    run_time = time.time() - start_time
    print('Preparation done..., time: %3.3f' % (run_time))  
    #============================Preparation===================================

    #============================3D-Unet===================================
    print('Run 3D-Unet...')
    start_time = time.time()
    
    model1 = utils.U3DModel()
    model1.model(x=np.zeros((1,40,40,40,27)), label=np.zeros((1,20)))
    model1.load_model(weights = r"./Rota5/unet3d/models/model_1.h5")

    model2 = utils.U3DModel()
    model2.model(x=np.zeros((1,40,40,40,27)), label=np.zeros((1,20)))
    model2.load_model(weights = r"./Rota5/unet3d/models/model_2.h5")

    model3 = utils.U3DModel()
    model3.model(x=np.zeros((1,40,40,40,27)), label=np.zeros((1,20)))
    model3.load_model(weights = r"./Rota5/unet3d/models/model_3.h5")

    models = [model1, model2, model3]
    for file_path in files_path:
        filename = file_path.split('/')[-1].split('.')[0]
        u3d_filename = filename + '.3dcnn.npy'
        if not os.path.exists(os.path.join(preparation_config["tmp_files_path"], u3d_filename)):
            seq_len = np.loadtxt((os.path.join(preparation_config["tmp_files_path"], filename + ".1d_inputs"))).shape[0]
            u3d = unet.U3DEng(file_path, models=models)
            u3d.reconstruct_protein(seq_len=seq_len, 
                                    output_path=os.path.join(preparation_config["tmp_files_path"], filename + '.3dcnn'))  
            
    run_time = time.time() - start_time
    print('3D-Unet done..., time: %3.3f' % (run_time))  
    #============================3D-Unet===================================
        
    #============================OPUS-Rota5===============================
    run_Rota5(preparation_config)
    #============================OPUS-Rota5===============================
    
    #============================mkpdb===============================
    toPDB(files_path, preparation_config)
    #============================mkpdb===============================
    print('All done...')
    