# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

from Rota5.network.my_layer import TrackableLayer
from Rota5.network.my_rnn import MyBilstm

import Rota5.network.pre_trained_embedding.model.EmbeddingModel as Pre_MSA_emb
import Rota5.network.pre_trained_embedding.model.EvoFormer as EvoFormer

class AAEmbedding(TrackableLayer):
    def __init__(self, config):
        super(AAEmbedding, self).__init__()
        self.config = config
        self.emb = Pre_MSA_emb.Embedding(self.config)

    def call(self, inp_1d, residue_index):
        return self.emb(inp_1d, residue_index)

class AFEvoformerEnsemble(TrackableLayer):
    def __init__(self, config, name_layer, iter_layer, name='evoformer_iteration', iters=None):
        super(AFEvoformerEnsemble, self).__init__(name=name_layer+"_"+str(iter_layer))
        self.config = config
        self.evo_iterations = []
        for i in range(len(iters)):
            global_config = {'iter': iters[i]}

            self.evo_iteration = EvoFormer.Evoformer(config, name=name, global_config=global_config)
            self.evo_iterations.append(self.evo_iteration)

    def call(self, msa, pair, training=True):
        for i in range(len(self.evo_iterations)):
            msa, pair = self.evo_iterations[i](msa, pair, training=training)
        return msa, pair

class Recycle(keras.layers.Layer):
    def __init__(self):
        super(Recycle, self).__init__()
        self.prev_msa_norm = keras.layers.LayerNormalization(name='prev_msa_first_row_norm')
        self.prev_pair_norm = keras.layers.LayerNormalization(name='prev_pair_norm')
        
        self.last_sc_encoding = keras.layers.Dense(256, name='last_sc_linear')

    def call(self, msa, pair, sc_last):
        # msa (1, L, 256), pair (L, L, 128), sc_last (1, L, 8)
        
        pair = tf.stop_gradient(pair)
        msa = tf.stop_gradient(msa)
        sc_last = tf.stop_gradient(sc_last)

        pair_last = self.prev_pair_norm(pair)
        msa_last = self.prev_msa_norm(msa) + self.last_sc_encoding(sc_last)
        
        return msa_last, pair_last

class ESMEmbedding(keras.layers.Layer):
    def __init__(self, n_feat):
        super(ESMEmbedding, self).__init__()
        self.esm_msa_proj = keras.layers.Dense(n_feat, name='esm_msa_linear')
        self.esm_msa_act_norm = keras.layers.LayerNormalization(name='esm_msa_act_norm')
        
    def call(self, feat):
        return self.esm_msa_act_norm(self.esm_msa_proj(feat))

class TRREmbedding(keras.layers.Layer):
    def __init__(self, name, n_feat):
        super(TRREmbedding, self).__init__()
        self.trr_emb = keras.layers.Conv2D(name=name, filters=n_feat, kernel_size=1, padding='SAME')
        self.norm = keras.layers.LayerNormalization()
        
    def call(self, feat):
        return self.norm(self.trr_emb(feat))

class CNN3dEmbedding(keras.layers.Layer):
    def __init__(self, num_layers, rate):
        super(CNN3dEmbedding, self).__init__()
        self.cov3d_1 = keras.layers.Conv3D(8, 3, strides = 1, padding = 'valid', activation = 'relu', name="3dcnn_1")
        self.cov3d_2 = keras.layers.Conv3D(2, 3, strides = 1, padding = 'valid', activation = 'relu', name="3dcnn_2")
        self.dnn = keras.layers.Dense(num_layers)
        self.dropout = keras.layers.Dropout(rate)
        
    def call(self, x, training):
        length = tf.shape(x)[1]
        x = tf.reshape(x, (1, length, 15, 15, 15, 5))
        x = tf.squeeze(x, 0)
        x = self.cov3d_1(x)
        x = self.cov3d_2(x)
        x = tf.reshape(x, (length, 11*11*11*2))
        x = tf.expand_dims(x, 0)
        x = self.dnn(x)
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        return x 

class Model(keras.Model):

    def __init__(self, model_params):
        super(Model, self).__init__()
        
        self.model_params = model_params
        #=========================== Embedding ===========================
        self.pre_msa_emb = AAEmbedding(self.model_params)

        self.evoformers = []
        for i in range(6):
            self.evoformers.append(AFEvoformerEnsemble(self.model_params["evofomer_config"]["evoformer"],
                                                       name_layer='evoformer_ensemble',
                                                       iter_layer=i,
                                                       iters=[4*i, 4*i+1, 4*i+2, 4*i+3]))

        self.esm_msa_proj = ESMEmbedding(n_feat=256)
        self.trr_emb = TRREmbedding(name="trr_feat", n_feat=self.model_params["n_2d_feat"])
        self.cnn3d_emb = CNN3dEmbedding(num_layers=512, rate=0.5)
        #=========================== Output ===========================
        self.bilstm = MyBilstm(num_layers=4,
                               units=512,
                               rate=0.25,
                               rota_output=8)
        
        self.recycle = Recycle()

        self.fc_rmsd = tf.keras.layers.Dense(20)

    def call(self, input_1d, input_2d, sc_mask, residue_index, L, training=False):
        
        assert sc_mask.shape == (1, L, 8)
        
        msa_last = tf.zeros((1, L, self.model_params["n_1d_feat"]))
        pair_last = tf.zeros((L, L, self.model_params["n_2d_feat"]))
        pred_sc_last = np.zeros((1, L, 8))

        msa_last, pair_last = self.recycle(msa_last, pair_last, pred_sc_last)  # msa_last (1, L, 256), pair_last (L, L, 128)

        f_seq = input_1d[:,:,:44]
        assert f_seq.shape == (1, L, 44)
        
        f_3d_cnn = input_1d[:,:,44:]
        assert f_3d_cnn.shape == (1, L, 16875)
        f_3d_cnn = self.cnn3d_emb(f_3d_cnn, training=training)
        assert f_3d_cnn.shape == (1, L, 512)
        
        f_esm = self.esm_msa_proj(np.zeros((1, L, 2560)))
        assert f_esm.shape == (1, L, 256)

        f_2d_trr = self.trr_emb(input_2d)[0]
        assert f_2d_trr.shape == (L, L, 128)
        
        CYCLE = self.model_params["n_cycle"]
        
        inp_1d = tf.concat([f_seq, f_3d_cnn], -1)
        assert inp_1d.shape == (1, L, 44 + 512)

        rota_outs = []
        rmsd_outs = []
        for c in range(CYCLE):
            f_1d, f_2d = self.pre_msa_emb(inp_1d, residue_index) # (1, L, 256) (L, L, 128)
            
            # Inject trr130 feature
            f_2d += f_2d_trr
            
            # Inject previous outputs for recycling
            f_2d += pair_last
            f_1d += msa_last

            for i in range(len(self.evoformers)):
                f_1d, f_2d = self.evoformers[i](f_1d, f_2d, training=training)
                
            rota_out = self.bilstm(f_1d, training=training) # (1, L, 8)
            
            rmsd_out = self.fc_rmsd(tf.concat([f_1d, rota_out], axis=-1)) # (1, L, 20)
            
            rota_outs.append(rota_out)
            rmsd_outs.append(rmsd_out)
            
            if c != CYCLE - 1:
                pred_sc_last = rota_out * sc_mask
                msa_last, pair_last = self.recycle(f_1d, f_2d, pred_sc_last)

        rmsd_out = tf.cast(tf.argmax(rmsd_out, -1), tf.float32)*0.05
        rmsd_out = tf.expand_dims(rmsd_out, -1)
        
        rota_out = tf.squeeze(rota_out*sc_mask, 0)
        rmsd_out = tf.squeeze(rmsd_out, 0)

        return rota_out, rmsd_out
 
    def load_model(self, reader, name):
        
        print ("load model:", os.path.join(self.model_params["save_path"], name))
        build_model_with_real_value(self, reader)
        self.load_weights(os.path.join(self.model_params["save_path"], name))
                
def build_model_with_real_value(model, reader):

    for step, filenames_batch in enumerate(reader.dataset):

        filenames, input_1d, input_2d, sc_mask, L = \
            reader.read_file_from_disk(filenames_batch)
            
        residue_index = np.array([range(L)])
        
        model(input_1d, input_2d, sc_mask, residue_index, L, training=False)
        break
    
    print ("build model")

