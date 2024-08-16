import torch
import torch.nn as nn
import torch.nn.functional as F
from model.sgat import GAT
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import GraphConv,HGTConv,GCN2Conv
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
                GAT(1, in_size, 256, 128, [1, 1], F.elu, 0.1,
                    0.1, 0.2, 1e-6, False, 1)
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
        self.decay = 1e-5
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
        self.RelationalAGG = RelationalAGG(g, in_size, out_size)
        self.semantic_attention = SemanticAttention(in_size=128)

        # metapath-based aggregation modules for user and item, this produces h2
        self.meta_path_patterns = meta_path_patterns
        # one HANLayer for user, one HANLayer for item
        self.hans = nn.ModuleDict({
            key: HANLayer(value, in_size, out_size, num_heads, dropout) for key, value in
            self.meta_path_patterns.items()
        })

        # layers to combine h0, h1, and h2
        # used to update node embeddings
        # self.user_layer1 = nn.Linear((num_heads + 1) * out_size, out_size, bias=True)
        self.user_layer1 = nn.Linear(out_size, out_size, bias=True)
        self.user_layer2 = nn.Linear(2 * out_size, out_size, bias=True)
        # self.item_layer1 = nn.Linear((num_heads + 1) * out_size, out_size, bias=True)
        self.item_layer1 = nn.Linear(out_size, out_size, bias=True)
        self.item_layer2 = nn.Linear(2 * out_size, out_size, bias=True)


        # layernorm
        self.layernorm = nn.LayerNorm(out_size)


    def forward(self, user_idx, item_idx, neg_item_idx):
        user_key = self.userkey
        item_key = self.itemkey
        # relational neighbor aggregation, h1
        h1 = self.RelationalAGG(self.g, self.feature_dict)
        # h1 = self.feature_dict
        # h1['user'] = self.user_emb
        # h1['movie'] = self.item_emb
        # metapath-based aggregation, h2
        h2 = {}
        for key in self.meta_path_patterns.keys():
            h2[key] = self.hans[key](self.g, self.feature_dict[key])
        user_soft = torch.cat((h1[user_key].unsqueeze(1), h2[user_key].unsqueeze(1)), 1)
        user_emb = self.semantic_attention(user_soft)
        item_soft = torch.cat((h1[item_key].unsqueeze(1), h2[item_key].unsqueeze(1)), 1)
        item_emb = self.semantic_attention(item_soft)
        # update node embeddings
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
        # rating = self.activation(rating)
        return rating