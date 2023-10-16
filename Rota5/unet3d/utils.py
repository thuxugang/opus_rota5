# This code is modifed from DLPacker. If you use it in your work, please cite the
# following paper as well:
# 
# @article {Misiura2021.05.23.445347,
#     author = {Misiura, Mikita and Shroff, Raghav and Thyer, Ross and Kolomeisky, Anatoly},
#     title = {DLPacker: Deep Learning for Prediction of Amino Acid Side Chain Conformations in Proteins},
#     elocation-id = {2021.05.23.445347},
#     year = {2021},
#     doi = {10.1101/2021.05.23.445347},
#     publisher = {Cold Spring Harbor Laboratory},
#     URL = {https://www.biorxiv.org/content/early/2021/05/25/2021.05.23.445347},
#     eprint = {https://www.biorxiv.org/content/early/2021/05/25/2021.05.23.445347.full.pdf},
#     journal = {bioRxiv}
# }

import re

import numpy as np

import tensorflow as tf
from tensorflow import keras

from collections import defaultdict

# do not change any of these
THE20 = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLN': 5,\
         'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11,\
         'MET': 12, 'PHE': 13, 'PRO': 14, 'SER': 15, 'THR': 16,\
         'TRP': 17, 'TYR': 18, 'VAL': 19}
SCH_ATOMS = {'ALA': 1, 'ARG': 7, 'ASN': 4, 'ASP': 4, 'CYS': 2, 'GLN': 5,\
             'GLU': 5, 'GLY': 0, 'HIS': 6, 'ILE': 4, 'LEU': 4, 'LYS': 5,\
             'MET': 4, 'PHE': 7, 'PRO': 3, 'SER': 2, 'THR': 3,\
             'TRP': 10, 'TYR': 8, 'VAL': 3}
BB_ATOMS = ['C', 'CA', 'N', 'O']
SIDE_CHAINS = {'MET': ['CB', 'CE', 'CG', 'SD'],
               'ILE': ['CB', 'CD1', 'CG1', 'CG2'],
               'LEU': ['CB', 'CD1', 'CD2', 'CG'],
               'VAL': ['CB', 'CG1', 'CG2'],
               'THR': ['CB', 'CG2', 'OG1'],
               'ALA': ['CB'],
               'ARG': ['CB', 'CD', 'CG', 'CZ', 'NE', 'NH1', 'NH2'],
               'SER': ['CB', 'OG'],
               'LYS': ['CB', 'CD', 'CE', 'CG', 'NZ'],
               'HIS': ['CB', 'CD2', 'CE1', 'CG', 'ND1', 'NE2'],
               'GLU': ['CB', 'CD', 'CG', 'OE1', 'OE2'],
               'ASP': ['CB', 'CG', 'OD1', 'OD2'],
               'PRO': ['CB', 'CD', 'CG'],
               'GLN': ['CB', 'CD', 'CG', 'NE2', 'OE1'],
               'TYR': ['CB', 'CD1', 'CD2', 'CE1', 'CE2', 'CG', 'CZ', 'OH'],
               'TRP': ['CB', 'CD1', 'CD2', 'CE2', 'CE3', 'CG', 'CH2', 'CZ2', 'CZ3', 'NE1'],
               'CYS': ['CB', 'SG'],
               'ASN': ['CB', 'CG', 'ND2', 'OD1'],
               'PHE': ['CB', 'CD1', 'CD2', 'CE1', 'CE2', 'CG', 'CZ']}
BOX_SIZE = 10
GRID_SIZE = 40
SIGMA = 0.65

class ResIdentitylBlock(keras.layers.Layer):
    
    def __init__(self, f1, f2):
        super(ResIdentitylBlock, self).__init__()

        self.conv1 = keras.layers.Conv3D(f1, 3, strides = 1, padding = 'same', activation = 'relu')
        self.conv2 = keras.layers.Conv3D(f1, 1, strides = 1, padding = 'same',  activation = 'relu')
        self.conv3 = keras.layers.Conv3D(f2, 3, strides = 1, padding = 'same',  activation = 'relu')
    
    def call(self, x, training=False):
        
        x_in = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + x_in
        return x

class UNETR_PP(tf.keras.Model):
    
    def __init__(self, depths=3, nres=8, width=128, grid_size=40):
        super(UNETR_PP, self).__init__()
        
        self.grid_size = grid_size
        
        self.fc = keras.layers.Dense(grid_size * grid_size * grid_size, activation = 'relu')

        self.conv_l1 = keras.layers.Conv3D(1 * width, 3, padding = 'same', strides = 2, activation = 'relu')
        self.blocks1 = []
        for _ in range(nres):
            self.blocks1.append(ResIdentitylBlock(1 * width, 1 * width))
            
        self.conv_l2 = keras.layers.Conv3D(2 * width, 3, padding = 'same', strides = 2, activation = 'relu')
        self.blocks2 = []
        for _ in range(nres):
            self.blocks2.append(ResIdentitylBlock(1 * width, 2 * width))
            
        self.conv_l3 = keras.layers.Conv3D(4 * width, 3, padding = 'same', strides = 2, activation = 'relu')
        self.blocks3 = []
        for _ in range(nres):
            self.blocks3.append(ResIdentitylBlock(2 * width, 4 * width))

        self.upsamp_1 = keras.layers.UpSampling3D(size = 2)
        self.conv_l4 = keras.layers.Conv3D(4 * width, 3, padding = 'same')

        self.upsamp_2 = keras.layers.UpSampling3D(size = 2)
        self.conv_l5 = keras.layers.Conv3D(2 * width, 3, padding = 'same')

        self.upsamp_3 = keras.layers.UpSampling3D(size = 2)
        self.conv_l6 = keras.layers.Conv3D(1 * width, 3, padding = 'same')
        
        self.blocks4 = []
        for _ in range(2):
            self.blocks4.append(ResIdentitylBlock(64, 1 * width))
        
        self.conv_l6_1 = keras.layers.Conv3D(4, 3, padding = 'same')
        self.conv_l6_2 = keras.layers.Conv3D(2, 3, padding = 'same')
        
    def call(self, x, label, training=False):
        
        fc = self.fc(label)
        fc = tf.reshape(fc, shape = (-1, self.grid_size, self.grid_size, self.grid_size, 1))
        
        l0 = tf.concat((x, fc), axis=-1)
        
        l1 = self.conv_l1(l0)
        for res_identity in self.blocks1:
            l1 = res_identity(l1, training=training)
            
        l2 = self.conv_l2(l1)
        for res_identity in self.blocks2:
            l2 = res_identity(l2, training=training)
            
        l3 = self.conv_l3(l2)
        for res_identity in self.blocks3:
            l3 = res_identity(l3, training=training)

        l = self.upsamp_1(l3)
        l = tf.concat((l, l2), axis=-1)
        l = self.conv_l4(l)
        l = tf.nn.leaky_relu(l, alpha=0.2)
        
        l = self.upsamp_2(l)
        l = tf.concat((l, l1), axis=-1)
        l = self.conv_l5(l)
        l = tf.nn.leaky_relu(l, alpha=0.2)
       
        l = self.upsamp_3(l)
        l = tf.concat((l, l0), axis=-1)
        l = self.conv_l6(l)
        l = tf.nn.leaky_relu(l, alpha=0.2)
        
        for res_identity in self.blocks4:
            l = res_identity(l, training=training)

        l1 = self.conv_l6_1(l)
        l2 = self.conv_l6_2(l)
        
        cnn3d_prediction = l1 + x[..., :4]
        
        return cnn3d_prediction, tf.nn.softmax(l2, -1)

class U3DModel():
    def __init__(self):
        self.model = UNETR_PP()
    
    def load_model(self, weights:str, history:str = ''):
        print (weights)
        self.model.load_weights(weights)
    
class InputBoxReader():
    def __init__(self, include_water:bool = False,\
                       charges_filename:str = './Rota5/unet3d/charges.rtp'):
        self.grid_size = GRID_SIZE
        self.grid_spacing = BOX_SIZE * 2 / GRID_SIZE # grid step
        self.offset = 10 * GRID_SIZE // 40 # to include atoms on the border
        self.total_size = GRID_SIZE + 2 * self.offset
        self.include_water = include_water

        # preparing the kernel and grid
        size = round(SIGMA * 4) # kernel size
        self.grid = np.mgrid[-size:size+self.grid_spacing:self.grid_spacing,\
                             -size:size+self.grid_spacing:self.grid_spacing,\
                             -size:size+self.grid_spacing:self.grid_spacing]

        # defining a kernel
        kernel = np.exp(-np.sum(self.grid * self.grid, axis = 0) / SIGMA**2 / 2) 
        kernel /= (np.sqrt(2 * np.pi) * SIGMA)
        self.kernel = kernel[1:-1, 1:-1, 1:-1]
        self.norm = np.sum(self.kernel)
        
        # read in the charges from special file
        self.charges = defaultdict(lambda: 0) # output 0 if the key is absent
        with open(charges_filename, 'r') as f:
            for line in f:
                if line[0] == '[' or line[0] == ' ':
                    if re.match('\A\[ .{1,3} \]\Z', line[:-1]):
                        key = re.match('\A\[ (.{1,3}) \]\Z', line[:-1])[1]
                        self.charges[key] = defaultdict(lambda: 0)
                    else:
                        l = re.split(r' +', line[:-1])
                        self.charges[key][l[1]] = float(l[3])
    
    def __call__(self, box:[str, dict]):

        amino_acids = set(box['resids'])
        amino_acids.remove(int(box['target']['id']))
                
        x  = np.zeros([self.total_size, self.total_size, self.total_size, 27])
        
        centers = (np.array(box['positions']) + BOX_SIZE) / self.grid_spacing
        centers += self.offset
        cr = np.round(centers).astype(np.int32)
        offsets = cr - centers
        offsets = offsets[:, :, None, None, None]
        
        i0 = self.kernel.shape[0] // 2
        i1 = self.kernel.shape[0] - i0
        
        for ind, a in enumerate(box['types']):
            if box['resnames'][ind] != 'HOH' or self.include_water:
                dist = self.grid + offsets[ind] * self.grid_spacing
                kernel = np.exp(-np.sum(dist * dist, axis = 0) / SIGMA**2 / 2)
                kernel = kernel[1:-1, 1:-1, 1:-1] * self.norm / np.sum(kernel)

                xa, xb = cr[ind][0]-i0, cr[ind][0]+i1
                ya, yb = cr[ind][1]-i0, cr[ind][1]+i1
                za, zb = cr[ind][2]-i0, cr[ind][2]+i1

                if a == 'C': ch = 0
                elif a == 'N': ch = 1
                elif a == 'O': ch = 2
                elif a == 'S': ch = 3
                else: ch = 4

                aa = box['resnames'][ind] # amino acid
                an = box['names'][ind]    # atom name

                if an in BB_ATOMS or box['resids'][ind] in amino_acids:
                    x[xa:xb, ya:yb, za:zb, ch] += kernel
                    if aa in self.charges:
                        charge = kernel * self.charges[aa][an]
                        x[xa:xb, ya:yb, za:zb, 5] += kernel * self.charges[aa][an]
                    else:
                        charge = kernel * self.charges['RST'][an[:1]]
                        x[xa:xb, ya:yb, za:zb, 5] += charge

                if an in BB_ATOMS:
                    if aa in THE20:
                        x[xa:xb, ya:yb, za:zb, 6 + THE20[aa]] += kernel
                    else:
                        x[xa:xb, ya:yb, za:zb, 6 + 20] += kernel
            b = self.offset
            
        return x[b:-b, b:-b, b:-b, :]