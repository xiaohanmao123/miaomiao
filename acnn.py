import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_max_pool as gmp
from torch_geometric.nn import GATv2Conv
import numpy as np
from odconv import ODConv2d
import yaml

with open("./config.yaml", "r") as f:
    config = yaml.safe_load(f)

size1 = config["size1"]
size2 = config["size2"]
size3 = config["size3"]
size4 = config["size4"]
size5 = config["size5"]
size6 = config["size6"]
size7 = config["size7"]
layers = config["layers"]
num = config["num"]

def odconv3x3(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
    return ODConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                    reduction=reduction, kernel_num=kernel_num)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=0.0625, kernel_num=1):
        super(BasicBlock, self).__init__()
        self.conv1 = odconv3x3(inplanes, planes, stride, reduction=reduction, kernel_num=kernel_num)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = odconv3x3(planes, planes, reduction=reduction, kernel_num=kernel_num)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class DoubleAttention(nn.Module):

    def __init__(self, in_channels,c_m=size1,c_n=size1,reconstruct = True):
        super().__init__()
        self.in_channels=in_channels
        self.reconstruct = reconstruct
        self.c_m=c_m
        self.c_n=c_n
        self.convA = nn.Conv1d(in_channels, out_channels=size1, kernel_size=1, stride=1, padding=1)
        self.convB = nn.Conv1d(in_channels, out_channels=size1, kernel_size=1, stride=1, padding=1)
        self.convV = nn.Conv1d(in_channels, out_channels=size1, kernel_size=1, stride=1, padding=1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv1d(in_channels=size1, out_channels=333, kernel_size = 1)
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h=x.shape
        assert c==self.in_channels
        A=self.convA(x)
        B=self.convB(x) #b,c_n,h,w
        V=self.convV(x) #b,c_n,h,w
        tmpA=A.view(b,self.c_m,-1)
        attention_maps=F.softmax(B.view(b,self.c_n,-1))
        attention_vectors=F.softmax(V.view(b,self.c_n,-1))
        global_descriptors=torch.bmm(tmpA,attention_maps.permute(0,2,1)) #b.c_m,c_n
        tmpZ = global_descriptors.matmul(attention_vectors) #b,c_m,h*w
        if self.reconstruct:
            tmpZ=self.conv_reconstruct(tmpZ)

        return tmpZ 

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = DoubleAttention(333)
        self.norm1 = nn.LayerNorm(258)
        self.norm2 = nn.LayerNorm(258)

        self.feed_forward = nn.Sequential(
            nn.Linear(258, 4*258),
            nn.ReLU(),
            nn.Linear(4*258, 258)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value)
        extra_dim = torch.zeros_like(attention[:, :, :2])
        query = torch.cat((query, extra_dim), dim=-1)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward+x))
        return out

class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
        trg_vocab_size,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
            for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(258, trg_vocab_size)
    
    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to('cuda:0')
        out = self.dropout(x.unsqueeze(-1) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        out = self.fc_out(out)
        return out

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx=0,
        trg_pad_idx=0,
        embed_size = 256,
        num_layers = layers,
        forward_expansion = 4,
        heads = 8,
        dropout = 0.1,
        device = "cuda",
        max_length = 333,
    ):
        super(Transformer, self).__init__()
    
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
            trg_vocab_size
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
            src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
            return src_mask.to(self.device)

    def make_trg_mask(self, trg):
            N, trg_len = trg.shape
            trg_mask = torch.tril(torch.ones((trg_len,trg_len))).expand(
                N, 1, trg_len, trg_len
            )
            return trg_mask.to(self.device)
        
    def forward(self, src):
            src_mask = self.make_src_mask(src)
            out = self.encoder(src, src_mask)
            return out

class targetmodel(nn.Module):
    def __init__(self, char_set_len=2):
        super().__init__()
        self.transform = Transformer(char_set_len, char_set_len)

    def forward(self, smiles):
        v = self.transform(smiles)
        v, _  = torch.max(v, -1) 
        return v

class NonLocal(nn.Module):
    def __init__(self, channel):
        super(NonLocal, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv1d(channel, self.inter_channel, kernel_size=1, stride=1, padding=0)
        self.conv_theta = nn.Conv1d(channel, self.inter_channel, kernel_size=1, stride=1, padding=0)
        self.conv_g = nn.Conv1d(channel, self.inter_channel, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv1d(self.inter_channel, channel, kernel_size=1, stride=1, padding=0)
  
    def forward(self, x):
        b, c, h = x.size()
        x_phi = self.conv_phi(x)
        x_theta = self.conv_theta(x).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).permute(0, 2, 1).contiguous()
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous()
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x 
        return out

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert(self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, values, keys, queries):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        values = values.reshape(N, value_len, self.heads, self.head_dim) 
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) 

        attention = torch.softmax(energy/ (self.embed_size ** (1/2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim) 
        out = self.fc_out(out)
        return out

class AFF(nn.Module):
    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.conv_1 = nn.Conv1d(in_channels=size3, out_channels=32, kernel_size=3, padding=1,
                            stride=1, groups=1)
        self.conv_2 = nn.Conv1d(in_channels=size3, out_channels=32, kernel_size=5, padding=2,
                            stride=1, groups=4)
        self.conv_3 = nn.Conv1d(in_channels=size3, out_channels=32, kernel_size=7, padding=3,
                            stride=1, groups=8)
        self.conv_4 = nn.Conv1d(in_channels=size3, out_channels=32, kernel_size=9, padding=4,
                            stride=1, groups=16)

        self.convs1 = nn.Conv1d(in_channels=1, out_channels=size3, kernel_size=1, stride=1, padding=0)
        self.nonlocal_block = NonLocal(size3)
        self.multihead = SelfAttention(333,3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        x = self.convs1(x)
        residual = self.convs1(residual)
        xa = x + residual
        x1 = self.conv_1(xa)
        x2 = self.conv_2(xa)
        x3 = self.conv_3(xa)
        x4 = self.conv_4(xa) 
        xl = torch.cat((x1, x2, x3, x4), dim=1) 

        xg = self.nonlocal_block(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

class cnn(nn.Module):
    def __init__(self, n_output=1, num_features_xd=87, output_dim=128, dropout=0.1):
        super(cnn, self).__init__()

        self.targetmodel = targetmodel()

        self.aff = AFF()

        self.basicblock = nn.ModuleList([BasicBlock(1,4) for _ in range(layers)])
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.n_output = n_output

        self.conv7 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=1)
        self.conv8 = nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3)
        self.conv9 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5)
        self.pool1d = nn.MaxPool1d(kernel_size=2)
        self.ppool2d = nn.MaxPool2d(kernel_size=(2, 2))

        self.ppfc1 = nn.Linear(3744, size4*num)
        self.ppfc2 = nn.Linear(size4*num, size4)
        self.ppfc3 = nn.Linear(size4, 333)

        self.pool3d = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.pfc1 = nn.Linear(1344, size5*num)
        self.pfc2 = nn.Linear(size5*num, size5)
        self.pfc3 = nn.Linear(size5, output_dim)

        self.conv1 = GATv2Conv(num_features_xd, num_features_xd)
        self.conv2 = GATv2Conv(num_features_xd, num_features_xd*2)
        self.conv3 = GATv2Conv(num_features_xd*2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd*4, 128)
        self.fc_g2 = torch.nn.Linear(128, output_dim)

        self.ddfc1 = nn.Linear(42624, size6*num)
        self.ddfc2 = nn.Linear(size6*num, size6)
        self.ddfc3 = nn.Linear(size6, size6)

        self.fc1 = nn.Linear(1044, size7)
        self.fc2 = nn.Linear(size7, size7//num)
        self.out = nn.Linear(size7//num, self.n_output)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target
        prot = data.prot
        graph = data.graph
        kc = data.kc
        liggen = data.liggen
        tef = data.tef
        prot = prot.unsqueeze(1)
        graph = graph.unsqueeze(1)
        kc = kc.unsqueeze(1)
        liggen = liggen.unsqueeze(1)
        tef = tef.unsqueeze(1)
        
        #AFEM
        tt = self.targetmodel(target)
        
        dcpro = data.dcpro        
        dcpro = dcpro.unsqueeze(1)
        
        #MSFEM
        xd = self.conv7(dcpro)
        xd = self.relu(xd)
        xd = self.conv8(xd)
        xd = self.relu(xd)
        xd = self.conv9(xd)
        xd = self.relu(xd)
        xd = self.pool1d(xd)
        xd = xd.view(-1, 3744)
        xd = self.ppfc1(xd)
        xd = self.relu(xd)
        xd = self.dropout(xd)
        xd = self.ppfc2(xd)
        xd = self.relu(xd)
        xd = self.dropout(xd)
        xd = self.ppfc3(xd)
        xd = self.relu(xd)
        xd = self.dropout(xd)

        #DFEM
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)       
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        #FFM
        xd = xd.unsqueeze(1)
        tt = tt.unsqueeze(1)
        xq = self.aff(xd,tt)
        xq = xq.view(-1, 42624)
        xq = self.ddfc1(xq)
        xq = self.relu(xq)
        xq = self.dropout(xq)
        xq = self.ddfc2(xq)
        xq = self.relu(xq)
        xq = self.dropout(xq)
        xq = self.ddfc3(xq)
        xq = self.relu(xq)
        xq = self.dropout(xq)
        xc = torch.cat((x, xq, prot, graph, kc, liggen, tef), 1)
        xc = xc.view(-1,3,87)
        xc = xc.unsqueeze(1)
        for block in self.basicblock:
            xc = block(xc)
        xc = xc.view(-1,1044)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        
        return out
