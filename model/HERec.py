import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl
from dgl.nn.pytorch import DeepWalk
from dgl.nn.pytorch import GATConv,HGTConv,GraphConv
import numpy as np
from torch.optim import Adam


class deepwalk(nn.Module):
    def __init__(self, meta_path_patterns, device, userkey, itemkey):
        super(deepwalk,self).__init__()
        self.meta_path_patterns = list(tuple(meta_path_pattern) for meta_path_pattern in meta_path_patterns)
        self._cached_graph = None
        self._cached_coalesced_graph = {}
        self.device = device
        self.userkey = userkey
        self.itemkey = itemkey

    def forward(self, g):
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path_pattern in self.meta_path_patterns:
                self._cached_coalesced_graph[meta_path_pattern] = dgl.metapath_reachable_graph(
                    g, meta_path_pattern)
        X = []
        for i, meta_path_pattern in enumerate(self.meta_path_patterns):
            new_g = self._cached_coalesced_graph[meta_path_pattern]
            # self.model = DeepWalk(new_g,walk_length=5,window_size=3,sparse=False).to(self.device)
            self.model = DeepWalk(new_g, walk_length=20, window_size=3, sparse=False).to(self.device)
            self.opt = Adam(self.model.parameters(), lr=0.01)
            self.dataloader = DataLoader(torch.arange(new_g.num_nodes()), batch_size=128,
                                    shuffle=True, collate_fn=self.model.sample)
            for epoch in range(15):
                for batch_walk in self.dataloader:
                    replace_greater_than(batch_walk, 0)
                    loss = self.model(batch_walk)
                    # loss += loss
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
            X.append(self.model.node_embed.weight.detach())

        return X

def replace_greater_than(tensor, k):
    mask = tensor < k  # 创建一个布尔掩码，标记大于 k 的元素
    tensor[mask] = 0  # 使用掩码将大于 k 的元素置为 0


class HERec(nn.Module):
    def __init__(self, g, meta_path_patterns,userkey, itemkey, in_size, out_size, num_heads, dropout, device):
        super(HERec, self).__init__()
        # embeddings for different types of nodes, h0
        self.initializer = nn.init.xavier_uniform_
        self.g = g
        self.userkey = userkey
        self.itemkey = itemkey
        self.unum = self.g.num_nodes(userkey)
        self.inum = self.g.num_nodes(itemkey)
        self.device = device
        self.feature_dict = nn.ParameterDict({
            ntype: nn.Parameter(self.initializer(torch.empty(g.num_nodes(ntype), in_size))) for ntype in g.ntypes
        })
        self.meta_path_patterns = meta_path_patterns
        self.hans = nn.ModuleDict({
            key: deepwalk(value, device, userkey, itemkey) for key, value in self.meta_path_patterns.items()
        })
        self.deepwalk_emb = {}
        for key in self.meta_path_patterns.keys():
            self.deepwalk_emb[key] = self.hans[key](self.g)

        self.user_metapathnum = len(self.deepwalk_emb[userkey])
        self.item_metapathnum = len(self.deepwalk_emb[itemkey])
        self.X = torch.stack(self.deepwalk_emb[userkey],dim=1)
        self.Y = torch.stack(self.deepwalk_emb[itemkey], dim=1)
        self.user_metapathdims = [128, 128, 128, 128]
        self.item_metapathdims = [128, 128, 128, 128]

        self.E = torch.nn.Embedding(self.unum, in_size)
        self.initializer(self.E.weight, gain=1)
        self.H = torch.nn.Embedding(self.inum, in_size)
        self.initializer(self.H.weight, gain=1)
        self.U = torch.nn.Embedding(self.unum, in_size)
        self.initializer(self.U.weight, gain=1)
        self.V = torch.nn.Embedding(self.inum, in_size)
        self.initializer(self.V.weight, gain=1)

        self.pu = nn.Parameter(torch.ones((self.unum, self.user_metapathnum)) * 1.0 / self.user_metapathnum)
        self.pv = nn.Parameter(torch.ones((self.inum, self.item_metapathnum)) * 1.0 / self.item_metapathnum)
        Wu = []
        bu = []
        for k in range(self.user_metapathnum):
            Wu.append(torch.randn(size=(out_size, self.user_metapathdims[k])) * 0.1)
            bu.append(torch.randn(out_size) * 0.1)
        self.Wu = nn.Parameter(torch.stack(Wu, dim=0))
        self.bu = nn.Parameter(torch.stack(bu, dim=0))

        Wv = []
        bv = []
        for k in range(self.item_metapathnum):
            Wv.append(torch.randn(size=(out_size, self.item_metapathdims[k]), requires_grad=True) * 0.1)
            bv.append(torch.randn(out_size, requires_grad=True) * 0.1)
        self.Wv = nn.Parameter(torch.stack(Wv, dim=0))
        self.bv = nn.Parameter(torch.stack(bv, dim=0))
        self.reg_u = 0.5
        self.reg_v = 0.5


    def bpr_loss(self, users, pos, neg):
        pos_scores = self.get_rating(users, pos)
        # pos_scores = torch.diagonal(pos_scores)
        neg_scores = self.get_rating(users, neg)
        # neg_scores = torch.diagonal(neg_scores)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss

    def getUsersRating(self, users):
        items = torch.Tensor(np.arange(self.inum)).long().to(self.device)
        ui = self.cal_u(users)
        vj = self.cal_v(items)
        a = torch.matmul(self.U.weight[users], self.V.weight[items].T)
        b = self.reg_u * torch.matmul(ui, self.H.weight[items].T)
        c = self.reg_v * torch.matmul(self.E.weight[users], vj.T)
        return a + b + c

    def cal_u(self, i):
        # unsqueeze
        unsqueeze1 = torch.unsqueeze(self.pu[i], dim=1)
        unsqueeze2 = torch.unsqueeze(self.X[i], dim=3)
        # matmul
        matmul1 = torch.matmul(self.Wu, unsqueeze2)
        # squeeze
        squeeze1 = torch.squeeze(matmul1,dim=3)

        # sigmoid
        sigmoid1 = torch.sigmoid(squeeze1 + self.bu)

        matmul2 = torch.matmul(unsqueeze1, sigmoid1)
        squeeze2 = torch.squeeze(matmul2)
        final = torch.sigmoid(squeeze2)


        return final

    def cal_u_all(self):
        return torch.sigmoid(torch.squeeze(torch.matmul(torch.unsqueeze(self.pu, dim=1), torch.sigmoid(
            torch.squeeze(torch.matmul(self.Wu, torch.unsqueeze(self.X, dim=3)),dim=3) + self.bu))))

    def cal_v(self, i):
        return torch.sigmoid(torch.squeeze(torch.matmul(torch.unsqueeze(self.pv[i], dim=1), torch.sigmoid(
            torch.squeeze(torch.matmul(self.Wv, torch.unsqueeze(self.Y[i], dim=3)),dim=3) + self.bv))))

    def get_rating(self, i, j):
        ui = self.cal_u(i)
        vj = self.cal_v(j)
        a = torch.mul(self.U.weight[i], self.V.weight[j]).sum(dim=1)
        b = self.reg_u * torch.mul(ui, self.H.weight[j]).sum(dim=1)
        c = self.reg_v * torch.mul(self.E.weight[i], vj).sum(dim=1)
        # return self.U.weight[i, :].dot(self.V.weight[j, :]) + self.reg_u * ui.dot(self.H.weight[j, :]) + self.reg_v * self.E.weight[i, :].dot(vj)
        return a+b+c

    def get_rating_2(self, i, j, ui, vj):
        return self.U[i, :] * (self.V[j, :]) + self.reg_u * ui.dot(self.H[j, :]) + self.reg_v * self.E[i, :] * (vj)