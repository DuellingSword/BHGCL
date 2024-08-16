import numpy as np
import torch
import torch as th
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GATConv,GraphConv
import scipy.sparse as sp
def consis_loss(logps, temp, lam):
    ps = [th.exp(p) for p in logps]
    ps = th.stack(ps, dim=2)

    avg_p = th.mean(ps, dim=2)
    sharp_p = (th.pow(avg_p, 1. / temp) / th.sum(th.pow(avg_p, 1. / temp), dim=1, keepdim=True)).detach()

    sharp_p = sharp_p.unsqueeze(2)
    loss = th.mean(th.sum(th.pow(ps - sharp_p, 2), dim=1, keepdim=True))

    loss = lam * loss
    return loss

def drop_node(feats, drop_rate, training):
    n = feats.shape[0]
    drop_rates = th.FloatTensor(np.ones(n) * drop_rate)

    if training:

        masks = th.bernoulli(1. - drop_rates).unsqueeze(1)
        feats = masks.to(feats.device) * feats

    else:
        feats = feats * (1. - drop_rate)

    return feats


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, input_droprate, hidden_droprate, use_bn=False):
        super(MLP, self).__init__()

        self.layer = nn.Linear(nfeat, nhid, bias=True)
        self.input_dropout = nn.Dropout(input_droprate)
        self.gat = GATConv(128, 128, 1, hidden_droprate, hidden_droprate, activation=F.relu_,
                        allow_zero_in_degree=True)
        # self.GCN = GCN()
        # self.hidden_dropout = nn.Dropout(hidden_droprate)
        # self.bn1 = nn.BatchNorm1d(nfeat)
        # self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn

    # def reset_parameters(self):
        # self.layer1.reset_parameters()
        # self.layer2.reset_parameters()

    def forward(self, g, x):
        x = self.input_dropout(x)
        x = self.gat(g,x)
        # x = self.GCN(g,x)
        x = x.view(x.shape[0],x.shape[2])
        # if self.use_bn:
        #     x = self.bn1(x)
        # x = F.leaky_relu(x)

        return x


class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()

    def forward(self, graph, feature):
        with graph.local_scope():
            node_f = feature
            # D^-1/2
            degs = graph.out_degrees().to(node_f.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)

            node_f = node_f * norm

            graph.ndata['n_f'] = node_f
            graph.update_all(message_func=fn.copy_u('n_f', 'm'), reduce_func=fn.sum('m', 'n_f'))

            rst = graph.ndata['n_f']

            degs = graph.in_degrees().to(node_f.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)
            rst = rst * norm

            return rst

def GRANDConv(graph, feats, order):
    '''
    Parameters
    -----------
    graph: dgl.Graph
        The input graph
    feats: Tensor (n_nodes * feat_dim)
        Node features
    order: int
        Propagation Steps
    '''
    with graph.local_scope():
        ''' Calculate Symmetric normalized adjacency matrix   \hat{A} '''
        degs1 = graph.in_degrees().float().clamp(min=1)
        norm1 = th.pow(degs1, -0.5).to(feats.device).unsqueeze(1)
        degs2 = graph.out_degrees().float().clamp(min=1)
        norm2 = th.pow(degs2, -0.5).to(feats.device).unsqueeze(1)
        graph.ndata['norm1'] = norm1
        graph.ndata['norm2'] = norm2
        node_f = feats * norm2

        graph.ndata['n_f'] = node_f
        graph.update_all(message_func=fn.copy_u('n_f', 'm'), reduce_func=fn.sum('m', 'n_f'))

        rst = graph.ndata['n_f']
        rst = rst * norm1
        graph.apply_edges(fn.u_mul_v('norm1', 'norm2', 'weight'))

        ''' Graph Conv '''
        x = rst
        y = 0 + feats

        for i in range(order):
            graph.ndata['h'] = x
            # graph.ndata['h'] = rst
            graph.update_all(fn.u_mul_e('h', 'weight', 'm'), fn.sum('m', 'h'))
            x = graph.ndata.pop('h')
            y.add_(x)

    return y / (order + 1)


class GRAND(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 S=1,
                 K=3,
                 node_dropout=0.0,
                 input_droprate=0.0,
                 hidden_droprate=0.1,
                 batchnorm=False):

        super(GRAND, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.S = S
        self.K = K

        self.mlp = MLP(in_dim, hid_dim, input_droprate, hidden_droprate, batchnorm)

        self.dropout = node_dropout
        self.node_dropout = nn.Dropout(node_dropout)
        self.consis_loss = consis_loss

    def forward(self, graph, feats):

        X = feats
        S = self.S

        output_list = []
        for s in range(S):
            drop_feat = drop_node(X, self.dropout, True)  # Drop node
            feat = GRANDConv(graph, drop_feat, self.K)  # Graph Convolution
            output_list.append(th.log_softmax(self.mlp(graph,feat), dim=-1))  # Prediction
            output = th.stack(output_list)
        return torch.mean(output,dim=0)
