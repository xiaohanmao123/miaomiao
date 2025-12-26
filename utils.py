import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch

"""
    PyTorch Geometric Dataset for drug-target affinity (DTA) tasks.
    Stores graph representations for molecules and features for targets.
    Convert SMILES, protein features, and labels to PyG Data objects.

        Args:
            xd: list of drug SMILES strings
            xt: list of target identifiers
            y: list of affinity labels
            smile_graph: dict mapping SMILES to (num_atoms, features, edge_index)
            pro_dic: dict mapping target to tripeptide composition
            dpro_dic: dict mapping target to Geary autocorrelation descriptors
            protdta, graphdta, kcdta, liggendta, tefdta: APMs' predictions

        Returns:
            Saves preprocessed PyG Data objects to disk
"""
class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None, pro_dic=None ,dpro_dic=None, nox_xt=None, protdta=None, graphdta=None, kcdta=None, liggendta=None, tefdta=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, y, smile_graph, pro_dic, dpro_dic, nox_xt, protdta, graphdta, kcdta, liggendta, tefdta)
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, xt, y, smile_graph, pro_dic, dpro_dic, nox_xt, protdta, graphdta, kcdta, liggendta, tefdta):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xt)

        for i in range(data_len):
            #print('Converting data to format: {}/{}'.format(i+1, data_len))

            smiles = xd[i]
            target = xt[i]
            nox_target = nox_xt[i]
            c_size, features, edge_index = smile_graph[smiles]
            pro_dic1=pro_dic[target]
            DCpro_dic1=dpro_dic[nox_target]
            labels = y[i]

            protdta_ = protdta[i]
            graphdta_ = graphdta[i]
            kcdta_ = kcdta[i]
            liggendta_ = liggendta[i]
            tefdta_ = tefdta[i]

            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))
            #print('DCpro_dic1', np.array(DCpro_dic1).shape)
            #print('pro_dic1', np.array(pro_dic1).shape)
            GCNData.dcpro = torch.FloatTensor([DCpro_dic1])
            GCNData.target = torch.FloatTensor([pro_dic1])
            #print('GCNData.dcpro', GCNData.dcpro.shape) #GCNData.dcpro torch.Size([1, 240])
            
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))

            GCNData.prot = torch.FloatTensor([protdta_])
            GCNData.graph = torch.FloatTensor([graphdta_])
            GCNData.kc = torch.FloatTensor([kcdta_])
            GCNData.liggen = torch.FloatTensor([liggendta_])
            GCNData.tef = torch.FloatTensor([tefdta_])

            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        #for key in data.keys():
            #print(key, data[key].shape)
        torch.save((data, slices), self.processed_paths[0])
# -------------------------------
# Evaluation metrics
# -------------------------------
def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci