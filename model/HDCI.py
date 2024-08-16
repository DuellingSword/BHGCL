import logging
import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import GATConv, HGTConv, GraphConv
from model.Contrast_User import Contrast_User
from model.Contrast_Item import Contrast_Item
from model.Contrast_IB import Contrast_IB


def print_hyperparameters(model_name, hyperparams, use_logging=False):
    header = f"{model_name} Hyperparameters"
    separator = "-" * len(header)
    formatted_params = "\n".join([f"{k.rjust(20, ' ')}: {v}" for k, v in hyperparams.items()])
    output = f"\n{separator}\n{header}\n{separator}\n{formatted_params}\n"
    if use_logging:
        logging.info(output)
    else:
        print(output)


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
    def __init__(self, g, meta_path_patterns, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()
        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_path_patterns)):
            self.gat_layers.append(
                # GCN_layer()
                GraphConv(in_size, out_size, norm='both', weight=None, bias=None,
                          activation=None, allow_zero_in_degree=True)
            )
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_path_patterns = list(tuple(meta_path_pattern) for meta_path_pattern in meta_path_patterns)
        self._cached_coalesced_graph = {}
        for meta_path_pattern in self.meta_path_patterns:
            self._cached_coalesced_graph[meta_path_pattern] = dgl.metapath_reachable_graph(
                g, meta_path_pattern)

    def forward(self, h):
        semantic_embeddings = []
        for i, meta_path_pattern in enumerate(self.meta_path_patterns):
            new_g = self._cached_coalesced_graph[meta_path_pattern]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


# Relational neighbor aggregation
class RelationalAGG(nn.Module):
    def __init__(self, g, args):
        super(RelationalAGG, self).__init__()
        self.in_size = in_size = args.in_size
        self.dropout = dropout = args.dropout
        self.out_size = args.out_size  # 暂时没用到
        self.device = args.device

        # Transform weights for different types of edges
        # 把in_size的src节点维度 转成 in_size维度
        self.W_T = nn.ModuleDict(
            {
                name: nn.Linear(in_size, in_size, bias=False)
                for name in g.etypes
            }
        )

        # Attention weights for different types of edges
        self.W_A = nn.ModuleDict(
            {name: nn.Linear(in_size, 1, bias=False) for name in g.etypes}
        )

        # layernorm
        self.layernorm = nn.LayerNorm(in_size)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

    # feat_dict为节点特征字典
    def forward(self, g, feat_dict):
        funcs = {}
        for srctype, etype, dsttype in g.canonical_etypes:
            g.nodes[dsttype].data["h"] = feat_dict[dsttype]
            g.nodes[srctype].data["h"] = feat_dict[srctype]
            g.nodes[srctype].data["t_h"] = self.W_T[etype](
                feat_dict[srctype]
            )  # src nodes' transformed feature

            # compute the attention numerator (exp)
            # etype=etype 表示只对指定类型的边执行这个操作
            g.apply_edges(fn.u_mul_v("t_h", "h", "x"), etype=etype)  # 对每条边上的 t_h 和 h 执行逐元素乘法，结果存储在目标张量 x 中
            g.edges[etype].data["x"] = torch.exp(
                # 注意力权重系数向量和g.edges[etype].data["x"]做点乘
                self.W_A[etype](g.edges[etype].data["x"])
            )

            # first update to compute the attention denominator (\sum exp)
            funcs[etype] = (fn.copy_e("x", "m"), fn.sum("m", "att"))  # 按边类型聚合消息
        g.multi_update_all(funcs, "sum")  # 首先按照边类型进行聚合funcs[etype]，然后在不同类型之间进行聚合"sum"，最后更新所有节点的特征

        # u --e--> v
        funcs = {}
        for srctype, etype, dsttype in g.canonical_etypes:
            g.apply_edges(
                fn.e_div_v("x", "att", "att"), etype=etype
            )  # compute attention weights (numerator/denominator)
            funcs[etype] = (
                fn.u_mul_e("h", "att", "m"),
                fn.sum("m", "h"),
            )  # \sum(h0*att) -> h1
        # second update to obtain h1
        g.multi_update_all(funcs, "sum")

        # apply activation, layernorm, and dropout
        feat_dict = {}
        for ntype in g.ntypes:
            feat_dict[ntype] = self.dropout(
                self.layernorm(F.relu_(g.nodes[ntype].data["h"]))
            )  # apply activation, layernorm, and dropout

        return feat_dict


class HDCI(nn.Module):
    def __init__(self, g, args):
        super(HDCI, self).__init__()
        self.g = g
        self.in_size = in_size = args.in_size
        self.out_size = out_size = args.out_size
        self.num_heads = num_heads = args.num_heads
        self.dropout = dropout = args.dropout
        self.user_key = userkey = args.user_key
        self.item_key = itemkey = args.item_key
        self.unum = self.g.num_nodes(userkey)
        self.inum = self.g.num_nodes(itemkey)
        self.meta_path_patterns = args.meta_path_patterns
        self.device = args.device

        # embeddings for different types of nodes, h0
        self.initializer = nn.init.xavier_uniform_
        self.feature_dict = nn.ParameterDict({
            ntype: nn.Parameter(self.initializer(torch.empty(g.num_nodes(ntype), in_size))) for ntype in g.ntypes
        })

        # relational neighbor aggregation, this produces h1
        self.RelationalAGG = RelationalAGG(g, args)

        # DisenHAN aggregation, this produces h2
        self.hans = nn.ModuleDict({
            key: HANLayer(g, value, in_size, out_size, num_heads, dropout) for key, value in
            self.meta_path_patterns.items()
        })

        # 跨域-对比学习器
        self.cl_rate = args.cl_rate
        self.Contrast_User = Contrast_User(g, args)
        self.Contrast_Item = Contrast_Item(g, args)
        contrast_params = {
            "temperature": args.temperature,
            "lam": args.lam,
            "cl_rate": args.cl_rate,
            "hidden_dim": args.cl_hidden_dim,
            "pos_num": args.pos_num
        }
        print_hyperparameters("Contrast", contrast_params)
        print_hyperparameters("Contrast", contrast_params, use_logging=True)

        # 信息瓶颈-对比学习器
        self.IB_rate = args.IB_rate
        self.mask_reg = args.mask_reg
        self.Contrast_IB = Contrast_IB(g, args)
        IB_params = {
            "IB_rate": args.IB_rate,
            "mask_reg": args.mask_reg,
            "choosing_tmp": args.choosing_tmp,
            "ssl_temp": args.ssl_temp,
            "walk_length": args.walk_length,
            "LCN_layer": args.LCN_layer
        }
        print_hyperparameters("IB", IB_params)
        print_hyperparameters("IB", IB_params, use_logging=True)

        # layers to combine h0, h1, and h2
        # used to update node embeddings
        self.user_layer1 = nn.Linear(
            num_heads * out_size + in_size, out_size, bias=True
        )
        self.user_layer2 = nn.Linear(out_size + in_size, out_size, bias=True)
        self.item_layer1 = nn.Linear(
            num_heads * out_size + in_size, out_size, bias=True
        )
        self.item_layer2 = nn.Linear(out_size + in_size, out_size, bias=True)

        # layernorm
        self.layernorm = nn.LayerNorm(out_size)

    def get_h1_h2(self):
        # relational neighbor aggregation, h1
        h1 = self.RelationalAGG(self.g, self.feature_dict)
        # metapath-based aggregation, h2
        h2 = {}
        for key in self.meta_path_patterns.keys():
            h2[key] = self.hans[key](self.feature_dict[key])
        return h1, h2

    def predict(self, user_idx, item_idx):
        user_key = self.user_key
        item_key = self.item_key
        h1, h2 = self.get_h1_h2()
        user_emb = torch.cat((h1[user_key], h2[user_key]), 1)
        item_emb = torch.cat((h1[item_key], h2[item_key]), 1)
        user_emb = self.user_layer1(user_emb)
        item_emb = self.item_layer1(item_emb)
        user_emb = self.user_layer2(torch.cat((user_emb, self.feature_dict[user_key]), 1))
        item_emb = self.item_layer2(torch.cat((item_emb, self.feature_dict[item_key]), 1))
        user_embeddings_edge_drop, item_embeddings_edge_drop, user_embeddings_node_drop, item_embeddings_node_drop \
            = self.Contrast_IB.predict(user_emb, item_emb, self.feature_dict[self.user_key],self.feature_dict[self.item_key], user_idx, item_idx)
        catUser_emb = torch.cat((user_emb[user_idx], user_embeddings_edge_drop, user_embeddings_node_drop), dim=-1)
        catItem_emb = torch.cat((item_emb[item_idx], item_embeddings_edge_drop, item_embeddings_node_drop), dim=-1)

        return catUser_emb, catItem_emb

    def getUsersRating(self, user_idx):
        item_idx = torch.Tensor(np.arange(self.inum)).long().to(self.device)
        users_emb, all_items = self.predict(user_idx, item_idx)
        rating = torch.matmul(users_emb, all_items.t())
        return rating

    def forward(self, user_idx, item_idx, neg_item_idx):
        user_key = self.user_key
        item_key = self.item_key
        h1, h2 = self.get_h1_h2()
        user_emb = torch.cat((h1[user_key], h2[user_key]), 1)
        item_emb = torch.cat((h1[item_key], h2[item_key]), 1)
        user_emb = self.user_layer1(user_emb)
        item_emb = self.item_layer1(item_emb)
        user_emb = self.user_layer2(torch.cat((user_emb, self.feature_dict[user_key]), 1))
        item_emb = self.item_layer2(torch.cat((item_emb, self.feature_dict[item_key]), 1))
        # Relu
        user_emb = F.relu_(user_emb)
        item_emb = F.relu_(item_emb)
        # layer norm
        user_emb = self.layernorm(user_emb)
        item_emb = self.layernorm(item_emb)
        # obtain users/items embeddings and their interactions
        user_feat = user_emb[user_idx]
        item_feat = item_emb[item_idx]
        neg_item_feat = item_emb[neg_item_idx]

        # 计算对比损失
        cl_user_loss = self.Contrast_User(h1[user_key], h2[user_key])
        cl_item_loss = self.Contrast_Item(h1[item_key], h2[item_key])
        cl_loss = cl_user_loss[user_idx].mean() + cl_item_loss[item_idx].mean()

        # 计算 IB 损失
        bpr_loss_edge_drop, bpr_loss_node_drop, \
            score_unique_user_edge, score_unique_item_edge, \
            score_unique_user_node, score_unique_item_node, edge_reg, node_reg \
            = self.Contrast_IB(user_emb, item_emb,
                               self.feature_dict[self.user_key],
                               self.feature_dict[self.item_key],
                               user_idx, item_idx, neg_item_idx)

        return user_feat, item_feat, neg_item_feat, \
            cl_loss, \
            bpr_loss_edge_drop, bpr_loss_node_drop, \
            score_unique_user_edge, score_unique_item_edge, \
            score_unique_user_node, score_unique_item_node, edge_reg, node_reg

    def bpr_loss(self, users, pos, neg):
        users_emb, pos_emb, neg_emb, \
            cl_loss, \
            bpr_loss_edge_drop, bpr_loss_node_drop, \
            score_unique_user_edge, score_unique_item_edge, \
            score_unique_user_node, score_unique_item_node, edge_reg, node_reg = self.forward(users, pos, neg)

        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        bpr_loss = bpr_loss_edge_drop + bpr_loss_node_drop
        IB_loss = score_unique_user_edge + score_unique_item_edge + score_unique_user_node + score_unique_item_node
        mask_loss = node_reg + edge_reg
        return loss, reg_loss, \
            cl_loss, \
            bpr_loss, IB_loss, mask_loss
