# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 17:11:13 2016

@author: XuGang
"""

import os
from mkpdb.myclass import Residues, Myio
from mkpdb.buildprotein import RebuildStructure
import tensorflow as tf
import numpy as np

def run_script(multi_iter):
    
    bb_path, rota_init_path, output_pdb = multi_iter
    
    if not os.path.exists(output_pdb):
        init_rotamers, _ = Myio.readRotaNN(rota_init_path)
        num_rotamers = len(init_rotamers)
    
        # ############################## get main chain ##############################            
        atomsData_real = Myio.readPDB(bb_path)
        atomsData_mc = RebuildStructure.extractmc(atomsData_real)
        residuesData_mc = Residues.getResidueData(atomsData_mc) 
    
        assert num_rotamers == sum([i.num_dihedrals for i in residuesData_mc]) 
        num_atoms = sum([i.num_side_chain_atoms for i in residuesData_mc]) + 5*len(residuesData_mc)
        
        geosData = RebuildStructure.getGeosData(residuesData_mc)
        
        residuesData_mc = RebuildStructure.rebuild_cb(residuesData_mc, geosData)
        # ############################## get main chain ##############################  
    
        init_atoms_matrix = np.zeros((num_atoms, 3)).astype(np.float32) 
        init_atoms_matrix  = RebuildStructure.make_atoms_matrix(residuesData_mc, init_atoms_matrix)
    
        init_rotamers = [tf.Variable(i) for i in init_rotamers]
        
        atoms_matrix, atoms_matrix_name = RebuildStructure.rebuild(init_rotamers, residuesData_mc, geosData, init_atoms_matrix)
        Myio.outputPDB(residuesData_mc, atoms_matrix, atoms_matrix_name, output_pdb)
        
def toPDB(files_path, preparation_config):
    
    multi_iters = []
    for file_path in files_path:
        filename = file_path.split('/')[-1].split('.')[0]
        rota_init_path = os.path.join(preparation_config["output_path"], filename + ".rota5")
        output_pdb = os.path.join(preparation_config["output_path"], filename + ".pdb")
        multi_iters.append([file_path, rota_init_path, output_pdb])
    
    for multi_iter in multi_iters:
        run_script(multi_iter)
        
         