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

import numpy as np
import pickle5 as pickle

from Bio.PDB import PDBParser, Selection, Superimposer, PDBIO, Atom, Residue, Structure
from Rota5.unet3d.utils import U3DModel, InputBoxReader, THE20, SCH_ATOMS, BB_ATOMS, SIDE_CHAINS, BOX_SIZE

class U3DEng():
    def __init__(self, str_pdb:str, models = None):
        self.parser = PDBParser(PERMISSIVE = 1) # PDB files reader
        self.sup = Superimposer() # superimposer
        self.io = PDBIO() # biopython's IO lib
        
        self.box_size = BOX_SIZE  # do not change
        self.altloc = ['A', 'B']  # initial altloc selection order preference
        
        self.str_pdb = str_pdb
        self.ref_pdb = './Rota5/unet3d/reference.pdb' # reference atoms to align residues to
        self._read_structures()
        self.reconstructed = None
        
        self.lib_name = './Rota5/unet3d/library.pkl' # library of rotamers
        self._load_library()
        
        self.models = models

        self.input_reader = InputBoxReader()
    
    def _load_library(self):
        with open(self.lib_name, 'rb') as f:
            self.library = pickle.load(f)
        for k in self.library['grids']:
            self.library['grids'][k] = self.library['grids'][k].astype(np.float32)
    
    def _read_structures(self):
        self.structure = self.parser.get_structure('structure', self.str_pdb)
        self.reference = self.parser.get_structure('reference', self.ref_pdb)
        
        self._remove_hydrogens(self.structure) # we never use hydrogens
        self._convert_mse(self.structure)      # convers MSE to MET
        self._remove_water(self.structure)     # waters are not used anyway
        self._remove_altloc(self.structure)    # only leave one altloc
    
    def _remove_hydrogens(self, structure:Structure):
        for residue in Selection.unfold_entities(structure, 'R'):
            remove = []
            for atom in residue:
                if atom.element == 'H': remove.append(atom.get_id())
            for i in remove: residue.detach_child(i)
    
    def _convert_mse(self, structure:Structure):
        for residue in Selection.unfold_entities(structure, 'R'):
            if residue.get_resname() == 'MSE':
                residue.resname = 'MET'
                for atom in residue:
                    if atom.element == 'SE':
                        new_atom = Atom.Atom('SD',\
                                             atom.coord,\
                                             atom.bfactor,\
                                             atom.occupancy,\
                                             atom.altloc,\
                                             'SD  ',\
                                             atom.serial_number,\
                                             element='S')
                        residue.add(new_atom)
                        atom_to_remove = atom.get_id()
                        residue.detach_child(atom_to_remove)
    
    def _remove_water(self, structure:Structure):
        residues_to_remove = []
        for residue in Selection.unfold_entities(structure, 'R'):
            if residue.get_resname() == 'HOH':
                residues_to_remove.append(residue)
        for r in residues_to_remove:
            r.get_parent().detach_child(r.get_id())
    
    def _remove_altloc(self, structure:Structure):
        total_occupancy = {}
        for atom in Selection.unfold_entities(structure, 'A'):
            if atom.is_disordered():
                for alt_atom in atom:
                    occupancy = alt_atom.get_occupancy()
                    if alt_atom.get_altloc() in total_occupancy:
                        total_occupancy[alt_atom.get_altloc()] += occupancy
                    else:
                        total_occupancy[alt_atom.get_altloc()] = occupancy

        if 'A' in total_occupancy and 'B' in total_occupancy:
            if total_occupancy['B'] > total_occupancy['A']:
                self.altloc = ['B', 'A']
        
        disordered_list, selected_list = [], []
        for residue in Selection.unfold_entities(structure, 'R'):
            for atom in residue:
                if atom.is_disordered():
                    disordered_list.append(atom)
                    try:
                        selected_list.append(atom.disordered_get(self.altloc[0]))
                    except:
                        selected_list.append(atom.disordered_get(self.altloc[1]))
                    selected_list[-1].set_altloc(' ')
                    selected_list[-1].disordered_flag = 0
        
        for d, a in zip(disordered_list, selected_list):
            p = d.get_parent()
            p.detach_child(d.get_id())
            p.add(a)
    
    def _get_residue_tuple(self, residue:Residue):
        r = residue.get_id()[1]
        s = residue.get_full_id()[2]
        n = residue.get_resname()
        return (r, s, n)
    
    def _get_parent_structure(self, residue:Residue):
        return residue.get_parent().get_parent().get_parent()
    
    def _align_residue(self, residue:Residue):
        if not residue.has_id('N') or not residue.has_id('C') or not residue.has_id('CA'):
            print('Missing backbone atoms: residue', self._get_residue_tuple(residue))
            return False
        r = list(self.reference.get_atoms())
        s = [residue['N'], residue['CA'], residue['C']]
        self.sup.set_atoms(r, s)
        self.sup.apply(self._get_parent_structure(residue))
        return True
    
    def _get_box_atoms(self, residue:Residue):
        aligned = self._align_residue(residue)
        if not aligned: return []
        atoms = []
        b = self.box_size + 1 # one angstrom offset to include more atoms
        for a in self._get_parent_structure(residue).get_atoms():
            xyz = a.coord
            if xyz[0] < b and xyz[0] > -b and\
               xyz[1] < b and xyz[1] > -b and\
               xyz[2] < b and xyz[2] > -b:
                atoms.append(a)
        return atoms
    
    def _genetare_input_box(self, residue:Residue, allow_missing_atoms:bool = False):
        atoms = self._get_box_atoms(residue)
        if not atoms: return None
        
        r, s, n = self._get_residue_tuple(residue)
        
        exclude, types, resnames = [], [], []
        segids, positions, names = [], [], []
        resids = []
        
        for i, a in enumerate(atoms):
            p = a.get_parent()
            a_tuple = (p.get_id()[1], p.get_full_id()[2], p.get_resname())
            if a.get_name() not in BB_ATOMS and (r, s, n) == a_tuple:
                exclude.append(i)
            
            types.append(a.element)
            resnames.append(a.get_parent().get_resname())
            segids.append(a.get_parent().get_full_id()[2])
            positions.append(a.coord)
            names.append(a.get_name())
            resids.append(a.get_parent().get_id()[1])
            
        d = {'target': {'id': int(r), 'segid': s, 'name': n, 'atomids': exclude},\
             'types': np.array(types),\
             'resnames': np.array(resnames),\
             'segids': np.array(segids),\
             'positions': np.array(positions, dtype = np.float16),\
             'names': np.array(names),\
             'resids': np.array(resids)}
        
        if allow_missing_atoms or len(exclude) == SCH_ATOMS[n]:
            return d
        else:
            return None
    
    def _get_sorted_residues(self, structure:Structure):
        out = []
        for residue in Selection.unfold_entities(structure, 'R'):
            out.append(residue)
        return out
        
    def _remove_sidechains(self, structure:Structure):
        for residue in Selection.unfold_entities(structure, 'R'):
            if residue.get_resname() in THE20:
                self._remove_sidechain(residue)
    
    def _remove_sidechain(self, residue:Residue):
        l = []
        for atom in residue:
            if atom.get_name() not in BB_ATOMS:
                l.append(atom.get_id())
        for d in l: residue.detach_child(d)
    
    def _get_prediction(self, box:dict, label:str):

        i = self.input_reader(box)
        labels = np.zeros((1, 20), dtype = np.float32)
        labels[0, THE20[label]] = 1.0
        
        # run the model
        o11, o12 = self.models[0].model(i[None, ...], labels)
        o21, o22 = self.models[1].model(i[None, ...], labels)
        o31, o32 = self.models[2].model(i[None, ...], labels)
       
        o = np.mean([o11[0].numpy(), o21[0].numpy(), o31[0].numpy()], axis=0)
        o2 = np.mean([o12[0,:,:,:,1:].numpy(), o22[0,:,:,:,1:].numpy(), o32[0,:,:,:,1:].numpy()], axis=0)
        
        pred = o - i[..., :4]
        pred[pred < 0] = 0
        
        pred = pred[5:-5, 5:-5, 5:-5, :]
        
        dpred_ori = np.zeros((15, 15, 15, 4))
        for i in range(0, 30, 2):
            for j in range(0, 30, 2):
                for k in range(0, 30, 2):
                    v = np.mean(pred[i:i+2, j:j+2, k:k+2, :], axis = (0, 1, 2))
                    dpred_ori[i//2, j//2, k//2, :] = v
        
        if label not in ['ASN', 'GLN', 'HIS']: 
            dpred = np.sum(dpred_ori, axis = -1)
        else:
            dpred = dpred_ori

        o2 = o2[5:-5, 5:-5, 5:-5, :]
        
        o2_ori = np.zeros((15, 15, 15, 1))
        for i in range(0, 30, 2):
            for j in range(0, 30, 2):
                for k in range(0, 30, 2):
                    v = np.mean(o2[i:i+2, j:j+2, k:k+2, :], axis = (0, 1, 2))
                    o2_ori[i//2, j//2, k//2, :] = v
                    
        return dpred, dpred_ori, o2_ori
    
    def reconstruct_residue(self, residue:Residue):
        r, s, n = self._get_residue_tuple(residue)
        box = self._genetare_input_box(residue, True)
        
        if not box:
            print("Skipping residue:", (r, s, n), end = '\n')
            return
        
        pred, pred_ori, o2_ori = self._get_prediction(box, n)

        scores = np.abs(self.library['grids'][n] - pred)
        scores = np.mean(scores, axis = tuple(range(1, pred.ndim + 1)))
        best_ind = np.argmin(scores)
        best_score = np.min(scores)
        best_match = self.library['coords'][n][best_ind]
        
        self._remove_sidechain(residue)

        for i, name in enumerate(SIDE_CHAINS[n]):
            new_atom = Atom.Atom(name,\
                                 best_match[i],\
                                 0,\
                                 1,\
                                 ' ',\
                                 name,\
                                 2,\
                                 element = name[:1])
            residue.add(new_atom)
                
        return pred_ori, best_score, o2_ori
    
    def reconstruct_protein(self, seq_len:int, output_path:str = ''):
        if not self.reconstructed: self.reconstructed = self.structure.copy()
        else: print('Reconstructed structure already exists, something might be wrong!')
        self._remove_sidechains(self.reconstructed)
        
        feat_3dcnn = np.zeros((seq_len, 15*15*15*5+1))
        
        sorted_residues = self._get_sorted_residues(self.reconstructed)
        for i, residue in enumerate(sorted_residues):
            if residue.get_resname() in THE20 and residue.get_resname() != 'GLY':
                name = self._get_residue_tuple(residue)
                print("Working on residue:", i, name, end = '\r')
                pred_ori, best_score, o2_ori = self.reconstruct_residue(residue)
        for i, residue in enumerate(sorted_residues):
            if residue.get_resname() in THE20 and residue.get_resname() != 'GLY':
                name = self._get_residue_tuple(residue)
                print("Working on residue:", i, name, end = '\r')
                pred_ori, best_score, o2_ori = self.reconstruct_residue(residue)
        for i, residue in enumerate(sorted_residues):
            if residue.get_resname() in THE20 and residue.get_resname() != 'GLY':
                name = self._get_residue_tuple(residue)
                print("Working on residue:", i, name, end = '\r')
                pred_ori, best_score, o2_ori = self.reconstruct_residue(residue)
                out = np.concatenate([pred_ori, o2_ori], -1)
                feat = np.hstack((out.flatten(), best_score))
                assert feat.shape == (16875+1,)
                feat_3dcnn[i] = feat                
        assert seq_len == i + 1
        
        np.save(output_path, 
                feat_3dcnn.astype(np.float16)) 
        
