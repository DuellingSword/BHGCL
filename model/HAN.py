import torch
import torch.nn as nn
import torch.nn.functional as F
from model.randomGCN import GRAND
import dgl
import dgl.function as fn
from model.sgat import GAT
from dgl.nn.pytorch import GATConv,HGTConv,GraphConv
import numpy as np

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
                GATConv(in_size, out_size, layer_num_heads,
                        0.3, 0.3, activation=F.elu,
                        allow_zero_in_degree=True)
                # GRAND(in_size, out_size, 1, 1, 0.0, 0.0)
                # GraphConv(in_size, out_size, norm='both', weight=True, bias=True,
                #         allow_zero_in_degree=True)
                # GAT(1, in_size, 64, 128, [1, 1], F.elu, 0.1,
                    # 0.1, 0.2, 1e-6, False, 1)
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
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)



class TAHIN(nn.Module):
    def __init__(self,dataset, g, meta_path_patterns,userkey, itemkey, in_size, out_size, num_heads, dropout, device):
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


        # metapath-based aggregation modules for user and item, this produces h2
        self.meta_path_patterns = meta_path_patterns
        # one HANLayer for user, one HANLayer for item
        self.hans = nn.ModuleDict({
            # key: HANLayer(value, in_size, out_size, num_heads, dropout) for key, value in
            # self.meta_path_patterns.items()
            key: HANLayer(value, in_size, out_size, num_heads, dropout) for key, value in
            self.meta_path_patterns.items()
        })

        # layers to combine h0, h1, and h2
        # used to update node embeddings
        self.user_layer1 = nn.Linear(out_size, out_size, bias=True)
        self.item_layer1 = nn.Linear(out_size, out_size, bias=True)


        # layernorm
        self.layernorm = nn.LayerNorm(out_size)

        # network to score the node pairs
        self.pred = nn.Linear(out_size, out_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_size, 1)


    def forward(self, user_idx, item_idx, neg_item_idx):
        user_key = self.userkey
        item_key = self.itemkey

        # metapath-based aggregation, h2
        h2 = {}
        # for key in self.meta_path_patterns.keys():
        #     h2[key] = self.hans[key](self.g, self.feature_dict[key])
        h2[user_key] = self.hans[user_key](self.g, self.feature_dict[user_key])
        # update node embeddings
        user_emb = h2[user_key]
        # item_emb = h2[item_key]
        item_emb = self.feature_dict[item_key]
        user_emb = self.user_layer1(user_emb)
        item_emb = self.item_layer1(item_emb)

        # layer norm
        user_emb = self.layernorm(user_emb)
        item_emb = self.layernorm(item_emb)

        # obtain users/items embeddings and their interactions
        user_feat = user_emb[user_idx]
        item_feat = item_emb[item_idx]
        neg_item_feat = item_emb[neg_item_idx]

        return user_feat, item_feat, neg_item_feat

    def bpr_loss(self, users, pos, neg):
        users_emb, pos_emb, neg_emb = self.forward(users, pos, neg)

        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def getUsersRating(self, user_idx):
        item_idx = torch.Tensor(np.arange(self.inum)).long().to(self.device)
        users_emb, all_items, _ = self.forward(user_idx, item_idx, None)
        rating = torch.matmul(users_emb, all_items.t())
        return rating