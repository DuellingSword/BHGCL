import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GATConv,GraphConv
import numpy as np
import networkx as nx
from model.randomGCN import GRAND

# Semantic attention in the metapath-based aggregation (the same as that in the HAN)
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        '''
        Shape of z: (N, M , D*K)
        N: number of nodes
        M: number of metapath patterns
        D: hidden_size
        K: number of heads
        '''
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        return (beta * z).sum(1)  # (N, D * K)

# Metapath-based aggregation (the same as the HANLayer)
class HANLayer(nn.Module):
    def __init__(self, meta_path_patterns, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_path_patterns)):
            self.gat_layers.append(
                GRAND(in_size, out_size, 1, 7, 0.7, 0.3)
                                   )
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_path_patterns = list(tuple(meta_path_pattern) for meta_path_pattern in meta_path_patterns)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []
        # obtain metapath reachable graph
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path_pattern in self.meta_path_patterns:
                self._cached_coalesced_graph[meta_path_pattern] = dgl.metapath_reachable_graph(
                    g, meta_path_pattern)

        for i, meta_path_pattern in enumerate(self.meta_path_patterns):
            new_g = self._cached_coalesced_graph[meta_path_pattern]
            # new_g = dgl.remove_self_loop(new_g)
            # if meta_path_pattern == ("mg", "gm"):
            #     new_g = dgl.to_homogeneous(new_g)
            #     coo = new_g.adj(scipy_fmt='coo', etype='_E')
            #     csr_matrix = coo.tocsr()
            #     sp.save_npz("./data/movielensmgm.npz", csr_matrix)
            semantic_embeddings.append(self.gat_layers[i](new_g, h))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


# Relational neighbor aggregation
class RelationalAGG(nn.Module):
    def __init__(self, g, in_size, out_size, dropout=0.3):
        super(RelationalAGG, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        # Transform weights for different types of edges
        self.W_T = nn.ModuleDict({
            name: nn.Linear(in_size, out_size, bias=False) for name in g.etypes
        })

        # Attention weights for different types of edges
        self.W_A = nn.ModuleDict({
            name: nn.Linear(out_size, 1, bias=False) for name in g.etypes
        })

        # layernorm
        self.layernorm = nn.LayerNorm(out_size)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, feat_dict):
        funcs = {}
        for srctype, etype, dsttype in g.canonical_etypes:
            g.nodes[dsttype].data['h'] = feat_dict[dsttype]  # nodes' original feature
            g.nodes[srctype].data['h'] = feat_dict[srctype]
            g.nodes[srctype].data['t_h'] = self.W_T[etype](feat_dict[srctype])  # src nodes' transformed feature

            # compute the attention numerator (exp)
            # Update the features of the specified edges by the provided function.
            g.apply_edges(fn.u_mul_v('t_h', 'h', 'x'), etype=etype)
            g.edges[etype].data['x'] = torch.exp(self.W_A[etype](g.edges[etype].data['x']))

            # first update to compute the attention denominator (\sum exp)
            funcs[etype] = (fn.copy_e('x', 'm'), fn.sum('m', 'att'))
        g.multi_update_all(funcs, 'sum')

        funcs = {}
        for srctype, etype, dsttype in g.canonical_etypes:
            g.apply_edges(fn.e_div_v('x', 'att', 'att'),
                          etype=etype)  # compute attention weights (numerator/denominator)
            funcs[etype] = (fn.u_mul_e('h', 'att', 'm'), fn.sum('m', 'h'))  # \sum(h0*att) -> h1
        # second update to obtain h1
        g.multi_update_all(funcs, 'sum')

        # apply activation, layernorm, and dropout
        feat_dict = {}
        for ntype in g.ntypes:
            feat_dict[ntype] = self.dropout(
                self.layernorm(F.relu_(g.nodes[ntype].data['h'])))  # apply activation, layernorm, and dropout

        return feat_dict


class TAHIN(nn.Module):
    def __init__(self, g, meta_path_patterns,userkey, itemkey, in_size, out_size, num_heads, dropout, device):
        super(TAHIN, self).__init__()
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

        # relational neighbor aggregation, this produces h1
        # self.RelationalAGG = RelationalAGG(g,userkey,itemkey,in_size,out_size,1,0.1)
        self.RelationalAGG = RelationalAGG(g, in_size, out_size)
        # metapath-based aggregation modules for user and item, this produces h2
        self.meta_path_patterns = meta_path_patterns
        # one HANLayer for user, one HANLayer for item
        self.hans = nn.ModuleDict({
            key: HANLayer(value, in_size, out_size, num_heads, dropout) for key, value in
            self.meta_path_patterns.items()
        })
        # layers to combine h0, h1, and h2
        if itemkey == 'movie':
            pos_uu = sp.load_npz("./data/pos_u.npz")
            pos_ii = sp.load_npz("./data/pos_i.npz")
            num_u,num_i = 944, 1683
        elif itemkey == 'item':
            pos_uu = sp.load_npz("./data/amazon_pos_u.npz")
            pos_ii = sp.load_npz("./data/amazon_pos_i.npz")
            num_u, num_i = 6170, 2753
        else:
            pos_uu = sp.load_npz("./data/yelp_pos_u.npz")
            pos_ii = sp.load_npz("./data/yelp_pos_i.npz")
            num_u, num_i = 16240, 14285
        pos_uu = (pos_uu.row, pos_uu.col)
        pos_ii = (pos_ii.row, pos_ii.col)
        pos_u = torch.zeros(num_u, num_u, dtype=torch.int, device='cuda:0')
        pos_u[pos_uu] = 1
        pos_i = torch.zeros(num_i, num_i, dtype=torch.int, device='cuda:0')
        pos_i[pos_ii] = 1
        self.pos_u = pos_u.to(device)
        self.pos_i = pos_i.to(device)
        self.contrast = Contrast(128, 0.8, 0.5)
        # used to update node embeddings
        # self.user_layer1 = nn.Linear((num_heads + 1) * out_size, out_size, bias=True)
        self.user_layer1 = nn.Linear(out_size, out_size, bias=True)

        # self.item_layer1 = nn.Linear((num_heads + 1) * out_size, out_size, bias=True)
        self.item_layer1 = nn.Linear(out_size, out_size, bias=True)


        self.semantic_attention = SemanticAttention(in_size=128)
        # layernorm
        self.layernorm = nn.LayerNorm(out_size)

    def forward(self, user_idx, item_idx, neg_item_idx):
        user_key = self.userkey
        item_key = self.itemkey
        # relational neighbor aggregation, h1
        h1 = self.RelationalAGG(self.g, self.feature_dict)

        # metapath-based aggregation, h2
        h2 = {}
        for key in self.meta_path_patterns.keys():
            h2[key] = self.hans[key](self.g, self.feature_dict[key])

        loss_u = self.contrast(h1[user_key], h2[user_key], self.pos_u)
        loss_i = self.contrast(h1[item_key], h2[item_key], self.pos_i)
        # loss_u = 0
        # loss_i = 0

        user_soft = torch.cat((h1[user_key].unsqueeze(1), h2[user_key].unsqueeze(1)), 1)
        user_emb = self.semantic_attention(user_soft)
        item_soft = torch.cat((h1[item_key].unsqueeze(1), h2[item_key].unsqueeze(1)), 1)
        item_emb = self.semantic_attention(item_soft)
        # update node embeddings
        # user_emb = torch.cat((user, h2[user_key]), 1)
        # item_emb = torch.cat((item, h2[item_key]), 1)

        # user_emb = torch.cat((h1[user_key], h2[user_key]), 1)
        # item_emb = torch.cat((h1[item_key], h2[item_key]), 1)
        user_emb = self.user_layer1(user_emb)
        item_emb = self.item_layer1(item_emb)
        # user_emb = self.user_layer2(torch.cat((user_emb, self.feature_dict[user_key]), 1))
        # item_emb = self.item_layer2(torch.cat((item_emb, self.feature_dict[item_key]), 1))
        # Relu
        # user_emb = F.relu_(user_emb)
        # item_emb = F.relu_(item_emb)
        user_emb = F.leaky_relu_(user_emb)
        item_emb = F.leaky_relu_(item_emb)
        # layer norm
        user_emb = self.layernorm(user_emb)
        item_emb = self.layernorm(item_emb)

        # obtain users/items embeddings and their interactions
        user_feat = user_emb[user_idx]
        item_feat = item_emb[item_idx]
        neg_item_feat = item_emb[neg_item_idx]

        # return user_feat, item_feat, neg_item_feat
        return user_feat, item_feat, neg_item_feat, loss_u, loss_i

    def bpr_loss(self, users, pos, neg):
        # users_emb, pos_emb, neg_emb = self.forward(users, pos, neg)
        users_emb, pos_emb, neg_emb, loss_u, loss_i = self.forward(users, pos, neg)

        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        # return loss, reg_loss
        return loss, reg_loss, loss_u, loss_i

    def getUsersRating(self, user_idx):
        item_idx = torch.Tensor(np.arange(self.inum)).long().to(self.device)
        # users_emb, all_items, _ = self.forward(user_idx, item_idx, None)
        users_emb, all_items, _, loss_u, loss_i = self.forward(user_idx, item_idx, None)
        rating = torch.matmul(users_emb, all_items.t())
        return rating

class Contrast(nn.Module):

    def __init__(self, hidden_dim, tau, lambda_):
        """对比损失模块

        :param hidden_dim: int 隐含特征维数
        :param tau: float 温度参数
        :param lambda_: float 0~1之间，网络结构视图损失的系数（元路径视图损失的系数为1-λ）
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lambda_ = lambda_
        # self.reset_parameters()

    # def reset_parameters(self):
    #     gain = nn.init.calculate_gain('relu')
    #     for model in self.proj:
    #         if isinstance(model, nn.Linear):
    #             nn.init.xavier_normal_(model.weight, gain)

    def sim(self, x, y):
        """计算相似度矩阵

        :param x: tensor(N, d)
        :param y: tensor(N, d)
        :return: tensor(N, N) S[i, j] = exp(cos(x[i], y[j]))
        """
        x_norm = torch.norm(x, dim=1, keepdim=True)
        y_norm = torch.norm(y, dim=1, keepdim=True)
        numerator = torch.mm(x, y.t())
        denominator = torch.mm(x_norm, y_norm.t())
        return torch.exp(numerator / denominator / self.tau)

    def forward(self, z_sc, z_mp, pos):
        """
        :param z_sc: tensor(N, d) 目标顶点在网络结构视图下的嵌入
        :param z_mp: tensor(N, d) 目标顶点在元路径视图下的嵌入
        :param pos: tensor(N, N) 0-1张量，每个顶点的正样本
        :return: float 对比损失
        """
        z_sc_proj = self.proj(z_sc)
        z_mp_proj = self.proj(z_mp)
        sim_sc2mp = self.sim(z_sc_proj, z_mp_proj)
        sim_mp2sc = sim_sc2mp.t()

        sim_sc2mp = sim_sc2mp / (sim_sc2mp.sum(dim=1, keepdim=True) + 1e-8)  # 不能改成/=
        loss_sc = -torch.log(torch.sum(sim_sc2mp * pos, dim=1).mean())

        sim_mp2sc = sim_mp2sc / (sim_mp2sc.sum(dim=1, keepdim=True) + 1e-8)
        loss_mp = -torch.log(torch.sum(sim_mp2sc * pos, dim=1).mean())
        return self.lambda_ * loss_sc + (1 - self.lambda_) * loss_mp