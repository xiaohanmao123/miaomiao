"""
Data Preprocessing Script for Drug-Target Interaction Prediction
This script processes drug and protein data for the Davis and KIBA datasets.
It performs the following steps:
1. Converts SMILES strings of compounds into graph representations.
2. Encodes protein sequences using tripeptide composition and Geary autocorrelation descriptors.
3. Prepares PyTorch datasets with multiple modalities (graph, tripeptide composition, Geary autocorrelation descriptors, predictions from APMs, etc.)
4. Saves the processed datasets in .pt format for downstream training.
"""

import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Lipinski, Descriptors
from rdkit.Chem.EState import EState
import networkx as nx
from utils import *
from itertools import product
from PyBioMed.PyProtein.Autocorrelation import *

def get_atom_logp(atom):
    crippen_contribs = rdMolDescriptors._CalcCrippenContribs(atom.GetOwningMol())
    atom_idx = atom.GetIdx()
    return crippen_contribs[atom_idx][0]

def get_atom_mr(atom):
    crippen_contribs = rdMolDescriptors._CalcCrippenContribs(atom.GetOwningMol())
    atom_idx = atom.GetIdx()
    return crippen_contribs[atom_idx][1]

def get_gasteiger_charge(atom):
    mol = atom.GetOwningMol()
    AllChem.ComputeGasteigerCharges(mol)
    return float(atom.GetProp('_GasteigerCharge'))

"""
Generates a feature vector for a single atom.
Features include:
- Atom type 
- Degree, hydrogen count, implicit valence
- Aromaticity
- Hybridization
- Crippen logP and molar refractivity
- Gasteiger charge
    
Parameters:
atom: RDKit Atom object
    
Returns:
numpy array of shape (feature_dim,)
"""
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +  # 44
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +  # 11
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +  # 11
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +  # 11
                    [atom.GetIsAromatic()] +
                    one_of_k_encoding_unk(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                                                   Chem.rdchem.HybridizationType.SP2,
                                                                   Chem.rdchem.HybridizationType.SP3,
                                                                   Chem.rdchem.HybridizationType.SP3D,
                                                                   Chem.rdchem.HybridizationType.SP3D2,
                                                                   'other']) +  # 6
                    [get_atom_logp(atom)] +
                    [get_atom_mr(atom)] +
                    [get_gasteiger_charge(atom)])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

"""
    Converts a SMILES string to a graph representation for GNN input.    
    Parameters:
        smile: string, SMILES of a compound    
    Returns:
        c_size: number of atoms
        features: numpy array of atom features (num_atoms x feature_dim)
        edge_index: list of edges [[source, target], ...]
"""
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature)

    # Normalize features
    features = np.array(features)
    features = features / np.sum(features, axis=1, keepdims=True)  # Normalize each row

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index

# -----------------------------
# Protein sequence encoding
# -----------------------------

seq_voc = "ACDEFGHIKLMNPQRSTVWXY"
max_seq_len = 999
L = len(seq_voc)
combinations = [''.join(comb) for comb in product(seq_voc, repeat=3)]
combination_dict = {comb: i for i, comb in enumerate(combinations)}

"""
Converts a protein sequence into tripeptide composition indices.
Each 3-residue segment (tripeptide) of the sequence is mapped to an index 
based on a predefined dictionary of all possible tripeptides.
Parameters:
sequence: string, protein sequence
comb_dict: dictionary mapping tripeptides to indices
Returns:
numpy array of indices representing the tripeptide composition
"""
def find_indices(sequence, comb_dict):
    units = [sequence[i:i + 3] for i in range(0, 999, 3)]
    indices = [comb_dict[unit] for unit in units if unit in comb_dict]
    return np.array(indices)

pro = []
for dt_name in ['dummy']:
    opts = ['train', 'test', 'val']
    for opt in opts:
        df = pd.read_csv('data/' + dt_name + '_' + opt + '.csv')
        pro += list(df['target_sequence'])
pro = set(pro)

# Encode protein sequences into tripeptide composition
pro_ct = {}
for k in pro:
    k_clean = k.replace('X', 'A')
    k1 = find_indices(k_clean, combination_dict)
    k1 = np.pad(k1, (0, 333 - len(k1)), mode='constant', constant_values=-1)
    k1 = (k1 + 1) / 9261
    #print(len(k1)) #333
    pro_ct[k] = k1

# Encode protein sequences into Geary autocorrelation descriptors
pro_AutoTotal = {}
for k in pro:
    k_clean = k.replace('X', 'A')
    k1 = CalculateGearyAutoTotal(k_clean)
    k1 = list(k1.values())
    k1 = np.array(k1)
    #print(len(k1)) #240
    pro_AutoTotal[k] = k1

compound_iso_smiles = []
for dt_name in ['dummy']:
    opts = ['train', 'test', 'val']
    for opt in opts:
        df = pd.read_csv('data/' + dt_name + '_' + opt + '.csv')
        compound_iso_smiles += list(df['compound_iso_smiles'])
compound_iso_smiles = set(compound_iso_smiles)

smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g


# -----------------------------
# Prepare PyTorch datasets
# -----------------------------

datasets = ['dummy']
for dataset in datasets:
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    processed_data_file_val = 'data/processed/' + dataset + '_val.pt'
    
# If processed data does not exist, create it
    if not os.path.isfile(processed_data_file_train) or not os.path.isfile(processed_data_file_test):
        df_train = pd.read_csv('data/' + dataset + '_train.csv')
        df_test = pd.read_csv('data/' + dataset + '_test.csv')
        df_val = pd.read_csv('data/' + dataset + '_val.csv')

        print(f"Train DataFrame: {df_train.shape[0]} samples")
        print(f"Test DataFrame: {df_test.shape[0]} samples")
        print(f"Valid DataFrame: {df_val.shape[0]} samples")

        # Extract columns for each modality
        train_drugs = list(df_train['compound_iso_smiles'])
        train_prots = list(df_train['target_sequence'])
        train_Y = list(df_train['affinity'])
        nox_train_prots = list(df_train['nox_target_sequence'])
        train_3dprotdta = list(df_train['dprotdta'])
        train_graphdta = list(df_train['graphdta'])
        train_kcdta = list(df_train['kcdta'])
        train_liggendta = list(df_train['liggendta'])
        train_tefdta = list(df_train['tefdta'])

        test_drugs = list(df_test['compound_iso_smiles'])
        test_prots = list(df_test['target_sequence'])
        test_Y = list(df_test['affinity'])
        nox_test_prots = list(df_test['nox_target_sequence'])
        test_3dprotdta = list(df_test['dprotdta'])
        test_graphdta = list(df_test['graphdta'])
        test_kcdta = list(df_test['kcdta'])
        test_liggendta = list(df_test['liggendta'])
        test_tefdta = list(df_test['tefdta'])

        val_drugs = list(df_val['compound_iso_smiles'])
        val_prots = list(df_val['target_sequence'])
        val_Y = list(df_val['affinity'])
        nox_val_prots = list(df_val['nox_target_sequence'])
        val_3dprotdta = list(df_val['dprotdta'])
        val_graphdta = list(df_val['graphdta'])
        val_kcdta = list(df_val['kcdta'])
        val_liggendta = list(df_val['liggendta'])
        val_tefdta = list(df_val['tefdta'])

        train_drugs = np.asarray(train_drugs)
        train_prots = np.asarray(train_prots)
        train_Y = np.asarray(train_Y)
        nox_train_prots = np.asarray(nox_train_prots)
        train_3dprotdta = np.asarray(train_3dprotdta)
        train_graphdta = np.asarray(train_graphdta)
        train_kcdta = np.asarray(train_kcdta)
        train_liggendta = np.asarray(train_liggendta)
        train_tefdta = np.asarray(train_tefdta)

        test_drugs = np.asarray(test_drugs)
        test_prots = np.asarray(test_prots)
        test_Y = np.asarray(test_Y)
        nox_test_prots = np.asarray(nox_test_prots)
        test_3dprotdta = np.asarray(test_3dprotdta)
        test_graphdta = np.asarray(test_graphdta)
        test_kcdta = np.asarray(test_kcdta)
        test_liggendta = np.asarray(test_liggendta)
        test_tefdta = np.asarray(test_tefdta)

        val_drugs = np.asarray(val_drugs)
        val_prots = np.asarray(val_prots)
        val_Y = np.asarray(val_Y)
        nox_val_prots = np.asarray(nox_val_prots)
        val_3dprotdta = np.asarray(val_3dprotdta)
        val_graphdta = np.asarray(val_graphdta)
        val_kcdta = np.asarray(val_kcdta)
        val_liggendta = np.asarray(val_liggendta)
        val_tefdta = np.asarray(val_tefdta)

        # Create PyTorch datasets
        print('preparing ', dataset + '_train.pt in pytorch format!')
        train_data = TestbedDataset(root='data', dataset=dataset + '_train', xd=train_drugs, xt=train_prots, y=train_Y, smile_graph=smile_graph, pro_dic=pro_ct, dpro_dic=pro_AutoTotal, nox_xt=train_prots, protdta=train_3dprotdta, graphdta=train_graphdta, kcdta=train_kcdta, liggendta=train_liggendta, tefdta=train_tefdta)
        print(f"{len(train_data)} samples in train dataset.")

        print('preparing ', dataset + '_val.pt in pytorch format!')
        val_data = TestbedDataset(root='data', dataset=dataset + '_val', xd=val_drugs, xt=val_prots, y=val_Y, smile_graph=smile_graph, pro_dic=pro_ct, dpro_dic=pro_AutoTotal, nox_xt=val_prots, protdta=val_3dprotdta, graphdta=val_graphdta, kcdta=val_kcdta, liggendta=val_liggendta, tefdta=val_tefdta)
        print(f"{len(val_data)} samples in val dataset.")
        
        print('preparing ', dataset + '_test.pt in pytorch format!')
        test_data = TestbedDataset(root='data', dataset=dataset + '_test', xd=test_drugs, xt=test_prots, y=test_Y, smile_graph=smile_graph, pro_dic=pro_ct, dpro_dic=pro_AutoTotal, nox_xt=test_prots, protdta=test_3dprotdta, graphdta=test_graphdta, kcdta=test_kcdta, liggendta=test_liggendta, tefdta=test_tefdta)
        print(f"{len(test_data)} samples in test dataset.")
        
        print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')
    else:
        print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')