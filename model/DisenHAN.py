from copy import deepcopy
import numpy as np
from line_profiler import profile
from typing import List, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from collections import defaultdict
from model.aggregator import HomoAggregate_attention, HeteAttention


class DisenHAN(nn.Module):
    def __init__(self, g, args):
        super(DisenHAN, self).__init__()
        self.in_size = args.in_size
        self.out_size = args.out_size
        self.dropout = args.atten_dropout
        self.n_iter = args.n_iter
        self.n_face = args.n_face
        self.n_layer = args.n_layer
        self.hidden_dim = args.disen_hidden_dim  # 暂时没用到
        self.n_neigh = args.n_neigh  # umber of neighbor to sample
        self.device = args.device
        self.user_key = args.user_key
        self.item_key = args.item_key

        # neighs_type[dst_idx] = [src_idx1, src_idx2,...]
        self.neighs_type = [[] for _ in g.ntypes]
        self.ntype_to_idx = {ntype: idx for idx, ntype in enumerate(g.ntypes)}  # ntype_to_idx[ntype] = ntype_idx
        self.idx_to_ntype = {idx: ntype for idx, ntype in enumerate(g.ntypes)}  # ntype_to_idx[ntype_idx] = ntype
        self.n_nodes_list = [g.num_nodes(ntype) for ntype in g.ntypes]  # n_nodes_list[ntype_idx] = num_ntype
        for srctype, etype, dsttype in g.canonical_etypes:
            dst_idx = self.ntype_to_idx[dsttype]  # 获取目标类型的索引
            src_idx = self.ntype_to_idx[srctype]  # 获取源类型的索引
            self.neighs_type[dst_idx].append(src_idx)

        # we project its feature vector 𝑥𝑡 ∈ R 𝑑𝑖𝑛 into 𝐾 different subspaces, For 𝑡 of type 𝜙 (𝑡) ∈ A，还有 s
        # 也就是根据节点类型进行投影
        # pre_encoders[l][ntype_idx] 第 l+1 层聚合给第 l 层时 ntype 类型节点的投影
        self.pre_encoders = nn.ModuleList([nn.ModuleList(
            [
                nn.Linear(self.in_size, self.out_size, bias=False)
                for _ in range(len(self.n_nodes_list))
            ]
        ) for _ in range(self.n_layer)])

        # homo_encoders[l][dstType_idx][srcType_idx] 第 l+1 层聚合给第 l 层时, 每个 srcType 聚合给 dstType 的同质编码器
        self.homo_encoders = nn.ModuleList([nn.ModuleList(
            [
                nn.ModuleList([
                    HomoAggregate_attention(self.out_size, self.n_face[l], self.dropout)
                    for _ in range(len(self.neighs_type[dsttype]))
                ]) for dsttype in range(len(self.n_nodes_list))
            ]
        ) for l in range(self.n_layer)])

        # hete_encoders[l][dstType_idx] 第 l+1 层聚合给第 l 层时, dstType 处理 srcType集群 的异质编码器
        self.hete_encoders = nn.ModuleList([nn.ModuleList(
            [
                HeteAttention(self.out_size, self.n_face[l], len(self.neighs_type[dsttype]))
                for dsttype in range(len(self.n_nodes_list))
            ]
        ) for l in range(self.n_layer)])

    def recur_aggregate(self, hidden, next_hidden, hidden_nodes, hidden_type, current_layer):
        if isinstance(hidden, list):
            updated_neighs_list = []
            weight_p_neighs_list = []
            for type_index in range(len(hidden)):
                next_type = self.neighs_type[hidden_type][type_index]
                neigh = hidden[type_index]
                next_neigh = next_hidden[type_index]
                next_hidden_nodes = hidden_nodes[type_index]
                updated_hidden, weight_p = self.recur_aggregate(neigh, next_neigh, next_hidden_nodes, next_type,
                                                                current_layer)
                updated_neighs_list.append(updated_hidden)
                weight_p_neighs_list.append(weight_p)
        else:
            # 1层 -> 0层 的聚合情况
            # hidden: np.array([user], dtype=np.int)的延伸嵌入
            # next_hidden:[userType_邻居type0, userType_邻居type1,...]的延伸嵌入
            batch_size = hidden.shape[0]
            hidden_ = hidden.view(batch_size, -1, self.out_size)
            support_size = hidden_.shape[1]

            # 该参数(不参与到反向传播的参数更新)表示 元关系 𝜓(e) 对 方面k 的影响程度 目标节点是ntype类型
            # weight_p 是 dstType_idx对应的srcType_idx集群
            # weight_p[srcType_idx] = torch.ones(batch_size, support_size, self.n_face[current_layer])
            weight_p = [torch.ones(batch_size, support_size, self.n_face[current_layer]).to(self.device)
                        for _ in range(len(self.neighs_type[hidden_type]))]

            if current_layer > 0:
                hidden_ = self.pre_encoders[current_layer][hidden_type](hidden_)
            neigh_hiddens = []
            for type_index in range(len(next_hidden)):
                next_type = self.neighs_type[hidden_type][type_index]
                neigh_hidden_ = next_hidden[type_index].view(batch_size, support_size, self.n_neigh[current_layer],
                                                             self.out_size)
                neigh_hiddens.append(self.pre_encoders[current_layer][next_type](neigh_hidden_))
            hidden_tmp = hidden_  # hidden_tmp就是z_t_k

            for clus_iter in range(self.n_iter):
                neigh_encodes = []
                for type_index in range(len(next_hidden)):
                    neigh_homo = self.homo_encoders[current_layer][hidden_type][type_index](weight_p[type_index],
                                                                                            hidden_tmp,
                                                                                            neigh_hiddens[type_index])
                    neigh_encodes.append(neigh_homo)
                hidden_tmp, weight_p = self.hete_encoders[current_layer][hidden_type](hidden_,
                                                                                      torch.stack(neigh_encodes, dim=2))
            if current_layer > 0:
                hidden_tmp = torch.relu(hidden_tmp)
            updated_neighs_list = hidden_tmp.view_as(hidden)

            weight_p = torch.stack(weight_p, dim=2)
            hidden_shape = list(hidden.shape[:-1])
            hidden_shape.append(len(self.neighs_type[hidden_type]))
            hidden_shape.append(self.n_face[current_layer])
            weight_p_neighs_list = weight_p.view(hidden_shape)

        return updated_neighs_list, weight_p_neighs_list

    def recur_emb(self, neighs_layer, current_type, feature_dict):
        if isinstance(neighs_layer, list):
            neighs_layer_emb = []
            for type_index in range(len(neighs_layer)):
                next_type = self.neighs_type[current_type][type_index]
                neigh_emb = self.recur_emb(neighs_layer[type_index], next_type, feature_dict)
                neighs_layer_emb.append(neigh_emb)
            return neighs_layer_emb
        else:
            # neighs_layer = torch.from_numpy(neighs_layer).to(torch.long).to(self.device)
            return feature_dict[self.idx_to_ntype[current_type]](neighs_layer)

    def forward(self, feature_dict, user_key, item_key, user_neighs_layers, item_neighs_layers):
        # user_neighs_layers 和 user_hidden 相互配合, 其实就是在user_neighs_layers后面加多一维嵌入维度(大小是self.in_size)
        user_hidden = []
        item_hidden = []
        for l in range(self.n_layer + 1):  # 0 -> 1 -> 2  给每层需要参与运算的节点加上新的一维嵌入维度(大小是self.in_size)
            user_hidden.append(self.recur_emb(user_neighs_layers[l], self.ntype_to_idx[user_key], feature_dict))
            item_hidden.append(self.recur_emb(item_neighs_layers[l], self.ntype_to_idx[item_key], feature_dict))
        # print(user_hidden[0].shape)  torch.Size([943, 100]) batchSize_user x dim
        # print(user_hidden[1][0].shape)  torch.Size([943, 20, 100])  batchSize_user x n_neigh[0] x dim
        # print(user_hidden[2][0][0].shape)  torch.Size([943, 20, 10, 100]) batchSize_user x n_neigh[0] x n_neigh[1] x dim
        user_h = user_hidden[self.n_layer]
        item_h = item_hidden[self.n_layer]
        self.user_weight_p = []
        self.item_weight_p = []
        for l in range(self.n_layer - 1, -1, -1):  # l: 1 -> 0  user_hidden[1]  user_h=user_hidden[2]
            user_h, user_weight_p = self.recur_aggregate(user_hidden[l], user_h, user_neighs_layers[l],
                                                         self.ntype_to_idx[user_key], l)
            item_h, item_weight_p = self.recur_aggregate(item_hidden[l], item_h, item_neighs_layers[l],
                                                         self.ntype_to_idx[item_key], l)
            self.user_weight_p.append(user_weight_p)
            self.item_weight_p.append(item_weight_p)
        return user_h, item_h
