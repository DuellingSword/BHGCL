import dgl
import torch
import numpy as np
import torch_sparse
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv, HGTConv, GraphConv
from torch_cluster import random_walk
from model.Contrast_IB import Contrast_IB
from model.Contrast_User import Contrast_User
from model.Contrast_Item import Contrast_Item


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
                # GCN_layer()
                GraphConv(in_size, out_size, norm='both', weight=None, bias=None,
                          activation=None, allow_zero_in_degree=True)
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
            # new_g = dgl.to_homogeneous(new_g)
            # coo = new_g.adj(scipy_fmt='coo', etype='_E')
            # csr_matrix = coo.tocsr()
            # semantic_embeddings.append(self.gat_layers[i](h, csr_matrix).flatten(1))
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class DCCF(nn.Module):
    def __init__(self, g, args):
        super(DCCF, self).__init__()
        self.g = g
        self.userkey = userkey = args.user_key
        self.itemkey = itemkey = args.item_key
        self.n_users = self.g.num_nodes(userkey)
        self.n_items = self.g.num_nodes(itemkey)
        self.n_nodes = self.n_users + self.n_items
        self.emb_dim = args.in_size
        self.n_layers = 2
        self.n_intents = 128

        _user_intent = torch.empty(self.emb_dim, self.n_intents)
        nn.init.xavier_normal_(_user_intent)
        self.user_intent = torch.nn.Parameter(_user_intent, requires_grad=True)
        _item_intent = torch.empty(self.emb_dim, self.n_intents)
        nn.init.xavier_normal_(_item_intent)
        self.item_intent = torch.nn.Parameter(_item_intent, requires_grad=True)

    def forward(self, G_indices, G_values, feature_dict):
        self.G_indices = G_indices
        self.G_values = G_values
        self.feature_dict = feature_dict
        all_embeddings = [torch.concat([self.feature_dict[self.userkey], self.feature_dict[self.itemkey]], dim=0)]

        for i in range(0, self.n_layers):
            # Graph-based Message Passing
            gnn_layer_embeddings = torch_sparse.spmm(self.G_indices, self.G_values, self.n_nodes, self.n_nodes,
                                                     all_embeddings[i])
            # Intent-aware Information Aggregation
            u_embeddings, i_embeddings = torch.split(all_embeddings[i], [self.n_users, self.n_items], 0)
            u_int_embeddings = torch.softmax(u_embeddings @ self.user_intent, dim=1) @ self.user_intent.T
            i_int_embeddings = torch.softmax(i_embeddings @ self.item_intent, dim=1) @ self.item_intent.T
            int_layer_embeddings = torch.concat([u_int_embeddings, i_int_embeddings], dim=0)
            all_embeddings.append(gnn_layer_embeddings + int_layer_embeddings + all_embeddings[i])

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.sum(all_embeddings, dim=1, keepdim=False)
        self.ua_embedding, self.ia_embedding = torch.split(all_embeddings, [self.n_users, self.n_items], 0)
        return self.ua_embedding, self.ia_embedding


class TAHIN(nn.Module):
    def __init__(self, g, args):
        super(TAHIN, self).__init__()
        self.g = g
        self.user_key = user_key = args.user_key
        self.item_key = item_key = args.item_key
        self.ui_relation = args.ui_relation
        self.in_size = args.in_size
        self.out_size = args.out_size
        self.n_users = self.g.num_nodes(user_key)
        self.n_items = self.g.num_nodes(item_key)
        self.n_nodes = self.n_users + self.n_items
        self.device = args.device
        self.initializer = nn.init.xavier_uniform_
        self.feature_dict = nn.ParameterDict({
            ntype: nn.Parameter(self.initializer(torch.empty(g.num_nodes(ntype), args.in_size))) for ntype in g.ntypes
        })

        self.IB_rate = args.IB_rate
        self.walk_length = args.walk_length
        self.choosing_tmp = args.choosing_tmp
        self.ssl_temp = args.ssl_temp
        self.LCN_layer = args.LCN_layer
        self.ui_relation = args.ui_relation
        self.latent_size = args.in_size
        self.node_mask_learner = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.in_size, self.in_size),
                nn.ReLU(),
                nn.Linear(self.in_size, 1)
            )
            for _ in range(self.LCN_layer)])
        self.edge_mask_learner = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * self.in_size, self.in_size),
                nn.ReLU(),
                nn.Linear(self.in_size, 1)
            )
            for _ in range(self.LCN_layer)])

        self.create_sparse_adjaceny()
        self.DCCFs = nn.ModuleList([DCCF(g, args) for _ in range(3)])
        # metapath-based aggregation modules for user and item, this produces h2
        self.meta_path_patterns = args.meta_path_patterns
        self.han_layers = 1
        # one HANLayer for user, one HANLayer for item
        self.hans = nn.ModuleDict({
            key: HANLayer(value, args.in_size, args.out_size, args.num_heads, args.dropout) for key, value in
            self.meta_path_patterns.items()
        })

    def create_sparse_adjaceny(self):
        row_idx = []
        col_idx = []
        adj = self.g.adj_external(ctx='cpu', scipy_fmt='csr', etype=self.ui_relation)
        for i in range(adj.shape[0]):  # 每一行代表 与目标类型id=i相连的srcType的节点ID
            # 从 CSC 矩阵中获取第 i 列的所有非零行索引，即所有与列索引 i 有连接的节点
            start = adj.indptr[i]
            end = adj.indptr[i + 1]
            cols = adj.indices[start:end]
            # 记录行索引和列索引
            for col in cols:
                row_idx.append(i)
                col_idx.append(col)
        # 将列索引转换成物品索引，确保它们在用户索引之后
        col_idx = [idx + self.n_users for idx in col_idx]
        # 转换为 NumPy 数组
        row_np = np.array(row_idx, dtype=np.int32)
        col_np = np.array(col_idx, dtype=np.int32)
        # 创建一个与 user_np 相同长度的全 1 数组
        ratings = np.ones_like(row_np, dtype=np.float32)
        # 构建新的稀疏矩阵
        tmp_adj = sp.csr_matrix((ratings, (row_np, col_np)), shape=(self.n_nodes, self.n_nodes), dtype=np.float32)
        self.ui_adj = tmp_adj + tmp_adj.T
        self.plain_adj = self.ui_adj
        rows, cols = self.ui_adj.nonzero()
        self.all_h_list = rows
        self.all_t_list = cols
        self.A_in_shape = self.plain_adj.tocoo().shape
        self.A_indices = torch.tensor([self.all_h_list, self.all_t_list], dtype=torch.long).cuda()
        self.D_indices = torch.tensor(
            [list(range(self.n_users + self.n_items)), list(range(self.n_users + self.n_items))],
            dtype=torch.long).cuda()
        self.all_h_list = torch.LongTensor(self.all_h_list).cuda()
        self.all_t_list = torch.LongTensor(self.all_t_list).cuda()
        A_values = torch.ones(size=(len(self.all_h_list), 1)).view(-1).cuda()
        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=A_values,
                                             sparse_sizes=self.A_in_shape).cuda()
        D_values = A_tensor.sum(dim=1).pow(-0.5)
        G_indices, G_values = torch_sparse.spspmm(self.D_indices, D_values, self.A_indices, A_values,
                                                  self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
        self.G_indices, self.G_values = torch_sparse.spspmm(G_indices, G_values, self.D_indices, D_values,
                                                            self.A_in_shape[0],
                                                            self.A_in_shape[1], self.A_in_shape[1])

        """ 构建归一化邻接矩阵, 大小为 (self.user_num + self.item_num, self.user_num + self.item_num) """
        self.joint_adjaceny_matrix_normal = torch.sparse_coo_tensor(self.G_indices, self.G_values,
                                                                    [self.A_in_shape[0],
                                                                     self.A_in_shape[1]]).coalesce().to(self.device)
        joint_indices = self.joint_adjaceny_matrix_normal.indices()
        self.row = joint_indices[0].to('cpu')
        self.col = joint_indices[1].to('cpu')
        """ 构建随机游走矩阵 """
        start = torch.arange(self.n_nodes)
        walk = random_walk(self.row, self.col, start, walk_length=self.walk_length)
        self.rw_adj = torch.zeros((self.n_nodes, self.n_nodes))
        self.rw_adj = torch.scatter(self.rw_adj, 1, walk, 1).to_sparse()
        degree = torch.sparse.sum(self.rw_adj, dim=1).to_dense()
        degree = torch.pow(degree + 1e-8, -1)
        degree[torch.isinf(degree)] = 0
        D_inverse = torch.diag(degree, diagonal=0).to_sparse()
        # 左乘 self.rw_adj 来进行归一化，确保了从每个节点出发的转移概率加起来等于1
        # size=(n_user+n_item, n_user+n_item)
        self.rw_adj = torch.sparse.mm(D_inverse, self.rw_adj).coalesce().to(self.device)

    def forward(self, user_idx, item_idx, neg_item_idx):
        # DCCF嵌入
        ua_embedding, ia_embedding = self.DCCFs[0](self.G_indices, self.G_values, self.feature_dict)
        # metapath-based aggregation, h2
        h2 = {}
        for key in self.meta_path_patterns.keys():
            for i in range(self.han_layers):
                if i == 0:
                    h2[key] = self.hans[key](self.g, self.feature_dict[key])
                else:
                    h2[key] = self.hans[key](self.g, h2[key])
        user_emb = 0.5 * ua_embedding + 0.5 * h2[self.user_key]
        item_emb = 0.5 * ia_embedding + 0.5 * h2[self.item_key]

        """ 重参数化 """
        cur_user_emb = user_emb
        cur_item_emb = item_emb
        original_user_emb = self.feature_dict[self.user_key]
        original_item_emb = self.feature_dict[self.item_key]
        edge_mask_list = []
        node_mask_list = []
        cur_embedding = torch.cat([cur_user_emb, cur_item_emb], dim=0)
        for l in range(self.LCN_layer):
            # edge_num * 2emebdding_size
            edge_cat_embedding = torch.cat([cur_embedding[self.row], cur_embedding[self.col]], dim=-1)
            # edge_num
            edge_mask = self.edge_mask_learner[l](edge_cat_embedding)
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(edge_mask.size()) + (1 - bias)  # 范围 ∈ (0.0002, 0.9999]
            edge_gate_inputs = torch.log(eps) - torch.log(1 - eps)
            edge_gate_inputs = edge_gate_inputs.to(self.device)
            edge_gate_inputs = (edge_gate_inputs + edge_mask) / self.choosing_tmp
            edge_mask = torch.sigmoid(edge_gate_inputs).squeeze(1)
            edge_mask_list.append(edge_mask)

            # user_num + Item_num
            node_mask = self.node_mask_learner[l](cur_embedding)
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(node_mask.size()) + (1 - bias)
            node_gate_inputs = torch.log(eps) - torch.log(1 - eps)
            node_gate_inputs = node_gate_inputs.to(self.device)
            node_gate_inputs = (node_gate_inputs + node_mask) / self.choosing_tmp
            node_mask = torch.sigmoid(node_gate_inputs)
            node_mask_list.append(node_mask)

        """ edge_dropout_view """
        cur_embedding_edge_drop = torch.cat([original_user_emb, original_item_emb], dim=0)
        all_embeddings_edge_drop = [cur_embedding_edge_drop]
        edge_reg = 0
        for l in range(self.LCN_layer):
            edge_mask = edge_mask_list[l]
            new_edge = torch.mul(self.joint_adjaceny_matrix_normal.values(), edge_mask)  # 元素对元素的乘法
            edge_reg += new_edge.sum() / len(edge_mask)
            cur_embedding_edge_drop = torch_sparse.spmm(self.joint_adjaceny_matrix_normal.indices(), new_edge,
                                                        self.n_nodes, self.n_nodes,
                                                        cur_embedding_edge_drop)
            all_embeddings_edge_drop.append(cur_embedding_edge_drop)

        all_embeddings_edge_drop = torch.stack(all_embeddings_edge_drop, dim=0)
        all_embeddings_edge_drop = torch.mean(all_embeddings_edge_drop, dim=0, keepdim=False)
        user_embeddings_edge_drop, item_embeddings_edge_drop = torch.split(all_embeddings_edge_drop,
                                                                           [self.n_users, self.n_items])
        edge_reg = edge_reg / self.LCN_layer

        """ node_dropout_view """
        cur_embedding_node_drop = torch.cat([original_user_emb, original_item_emb], dim=0)
        all_embeddings_node_drop = [cur_embedding_node_drop]
        node_reg = 0
        for i in range(self.LCN_layer):
            node_mask = node_mask_list[i]
            # (user_num + item_num) * embedding_size
            # 它将一个节点的邻居节点嵌入根据一定的权重（在这里是随机游走概率）进行了平均
            mean_pooling_embedding = torch.mm(self.rw_adj, cur_embedding_node_drop)
            cur_embedding_node_drop = torch.mul(node_mask, cur_embedding_node_drop) + torch.mul((1 - node_mask),
                                                                                                mean_pooling_embedding)
            cur_embedding_node_drop = torch.mm(self.joint_adjaceny_matrix_normal, cur_embedding_node_drop)
            all_embeddings_node_drop.append(cur_embedding_node_drop)
            node_reg += node_mask.sum() / (self.n_users + self.n_items)

        all_embeddings_node_drop = torch.stack(all_embeddings_node_drop, dim=0)
        all_embeddings_node_drop = torch.mean(all_embeddings_node_drop, dim=0, keepdim=False)
        user_embeddings_node_drop, item_embeddings_node_drop = torch.split(all_embeddings_node_drop,
                                                                           [self.n_users, self.n_items])
        node_reg = node_reg / self.LCN_layer

        """ 计算BPR损失 """
        edge_drop_user_emb = user_embeddings_edge_drop[user_idx]
        posItem_emb = item_embeddings_edge_drop[item_idx]
        negItem_emb = item_embeddings_edge_drop[neg_item_idx]
        pos_scores = torch.sum(edge_drop_user_emb * posItem_emb, dim=1)
        neg_scores = torch.sum(edge_drop_user_emb * negItem_emb, dim=1)
        bpr_loss_edge_drop = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        node_drop_user_emb = user_embeddings_node_drop[user_idx]
        posItem_emb = item_embeddings_node_drop[item_idx]
        negItem_emb = item_embeddings_node_drop[neg_item_idx]
        pos_scores = torch.sum(node_drop_user_emb * posItem_emb, dim=1)
        neg_scores = torch.sum(node_drop_user_emb * negItem_emb, dim=1)
        bpr_loss_node_drop = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        """ 计算互信息 """
        def ssl_compute(x1, x2):
            x1, x2 = F.normalize(x1, p=2, dim=1), F.normalize(x2, p=2, dim=1)
            pos_score = torch.sum(torch.mul(x1, x2), dim=1, keepdim=False)
            all_score = torch.mm(x1, x2.t()) + 1e-8
            ssl_mi = torch.log(torch.exp(pos_score / self.ssl_temp) / torch.exp(all_score / self.ssl_temp).sum(dim=1,
                                                                                                               keepdim=False) + 1e-8)
            ssl_mi = torch.where(torch.logical_or(torch.isnan(ssl_mi), torch.isinf(ssl_mi)),
                                 torch.tensor(0.0).to(self.device), ssl_mi)
            return ssl_mi.mean()

        unique_user = torch.unique(user_idx)
        unique_posItem = torch.unique(item_idx)
        score_user_edge_drop = ssl_compute(user_embeddings_edge_drop[unique_user],
                                           cur_user_emb[unique_user])
        score_item_edge_drop = ssl_compute(item_embeddings_edge_drop[unique_posItem],
                                           cur_item_emb[unique_posItem])
        score_user_node_drop = ssl_compute(user_embeddings_node_drop[unique_user],
                                           cur_user_emb[unique_user])
        score_item_node_drop = ssl_compute(item_embeddings_node_drop[unique_posItem],
                                           cur_item_emb[unique_posItem])

        return user_emb[user_idx], item_emb[item_idx], item_emb[neg_item_idx], \
            bpr_loss_edge_drop + bpr_loss_node_drop, \
            score_user_edge_drop + score_item_edge_drop + score_user_node_drop + score_item_node_drop,\
            edge_reg + node_reg

    def bpr_loss(self, users, pos, neg):
        users_emb, pos_emb, neg_emb, IB_bprLoss, IB_loss, IB_reg = self.forward(users, pos, neg)
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users)) + IB_reg
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        loss += IB_bprLoss
        loss += self.IB_rate * IB_loss

        return loss, reg_loss

    def predict(self, user_idx, item_idx):
        ua_embedding, ia_embedding = self.DCCFs[0](self.G_indices, self.G_values, self.feature_dict)
        # metapath-based aggregation, h2
        h2 = {}
        for key in self.meta_path_patterns.keys():
            for i in range(self.han_layers):
                if i == 0:
                    h2[key] = self.hans[key](self.g, self.feature_dict[key])
                else:
                    h2[key] = self.hans[key](self.g, h2[key])
        user_emb = 0.5 * ua_embedding + 0.5 * h2[self.user_key]
        item_emb = 0.5 * ia_embedding + 0.5 * h2[self.item_key]
        # user_embeddings_edge_drop, item_embeddings_edge_drop, user_embeddings_node_drop, item_embeddings_node_drop \
        #     = self.Contrast_IB.predict(user_emb, item_emb, self.feature_dict[self.user_key],
        #                                self.feature_dict[self.item_key], user_idx, item_idx)
        # catUser_emb = torch.cat((user_emb[user_idx], user_embeddings_edge_drop, user_embeddings_node_drop), dim=-1)
        # catItem_emb = torch.cat((item_emb[item_idx], item_embeddings_edge_drop, item_embeddings_node_drop), dim=-1)
        return user_emb[user_idx], item_emb[item_idx]

    def getUsersRating(self, user_idx):
        item_idx = torch.Tensor(np.arange(self.n_items)).long().to(self.device)
        # users_emb, all_items, _ = self.forward(user_idx, item_idx, None)
        users_emb, all_items = self.predict(user_idx, item_idx)
        rating = torch.matmul(users_emb, all_items.t())
        return rating
