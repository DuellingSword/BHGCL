import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.sampling import sample_neighbors
from dgl.ops import edge_softmax
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GATConv, GraphConv
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp


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

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain)

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
        loss_sc = -torch.log(torch.sum(sim_sc2mp * pos, dim=1) + 1e-8).mean()

        sim_mp2sc = sim_mp2sc / (sim_mp2sc.sum(dim=1, keepdim=True) + 1e-8)
        loss_mp = -torch.log(torch.sum(sim_mp2sc * pos, dim=1) + 1e-8).mean()
        return self.lambda_ * loss_sc + (1 - self.lambda_) * loss_mp


class HeCoGATConv(nn.Module):

    def __init__(self, hidden_dim, attn_drop=0.1, negative_slope=0.01, activation=None):
        """HeCo作者代码中使用的GAT

        :param hidden_dim: int 隐含特征维数
        :param attn_drop: float 注意力dropout
        :param negative_slope: float, optional LeakyReLU负斜率，默认为0.01
        :param activation: callable, optional 激活函数，默认为None
        """
        super().__init__()
        self.attn_l = nn.Parameter(torch.FloatTensor(1, hidden_dim))
        self.attn_r = nn.Parameter(torch.FloatTensor(1, hidden_dim))
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.attn_l, gain)
        nn.init.xavier_normal_(self.attn_r, gain)

    def forward(self, g, feat_src, feat_dst):
        """
        :param g: DGLGraph 邻居-目标顶点二分图
        :param feat_src: tensor(N_src, d) 邻居顶点输入特征
        :param feat_dst: tensor(N_dst, d) 目标顶点输入特征
        :return: tensor(N_dst, d) 目标顶点输出特征
        """
        with g.local_scope():
            # HeCo作者代码中使用attn_drop的方式与原始GAT不同，这样是不对的，却能顶点聚类提升性能……
            attn_l = self.attn_drop(self.attn_l)
            attn_r = self.attn_drop(self.attn_r)
            el = (feat_src * attn_l).sum(dim=-1).unsqueeze(dim=-1)  # (N_src, 1)
            er = (feat_dst * attn_r).sum(dim=-1).unsqueeze(dim=-1)  # (N_dst, 1)
            g.srcdata.update({'ft': feat_src, 'el': el})
            g.dstdata['er'] = er
            g.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(g.edata.pop('e'))
            g.edata['a'] = edge_softmax(g, e)  # (E, 1)

            # 消息传递
            g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            ret = g.dstdata['ft']
            if self.activation:
                ret = self.activation(ret)
            return ret


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
                # GATConv(in_size, out_size, layer_num_heads,
                #         dropout, dropout, activation=F.elu,
                #         allow_zero_in_degree=True)
                # GraphConv(in_size, out_size, norm='both', weight=True, bias=True,
                #           activation=F.elu_, allow_zero_in_degree=True)
                GraphConv(in_size, out_size, norm='both',
                          activation=F.elu_, allow_zero_in_degree=True)
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
            # if meta_path_pattern == ("by", "yb"):
            #     new_g = dgl.to_homogeneous(new_g)
            #     coo = new_g.adj(scipy_fmt='coo', etype='_E')
            #     sp.save_npz("./data/dbookbyb.npz", coo)
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


# Relational neighbor aggregation
class RelationalAGG(nn.Module):
    def __init__(self, g, userkey, itemkey, in_size, out_size, layer_num_heads, dropout, device):
        super(RelationalAGG, self).__init__()
        self.gat_layers_node = nn.ModuleList()
        self.meta_paths_u = []
        self.meta_paths_i = []
        self.userkey = userkey
        self.itemkey = itemkey
        self.unum = g.num_nodes(userkey)
        self.inum = g.num_nodes(itemkey)
        self.device = device
        for srctype, etype, dsttype in g.canonical_etypes:
            if dsttype == userkey:
                self.meta_paths_u.append((srctype, etype, dsttype))
            elif dsttype == itemkey:
                self.meta_paths_i.append((srctype, etype, dsttype))
        for i in range(len(self.meta_paths_u) + len(self.meta_paths_i)):
            self.gat_layers_node.append(
                HeCoGATConv(in_size, dropout, activation=F.elu)
                # GraphConv(in_size, out_size, norm='both', weight=True, bias=True,
                #           activation=F.elu_, allow_zero_in_degree=True)
            )
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)

    def forward(self, g, h):
        embedding_u = []
        embedding_i = []
        embedding = []

        for i, (srctype, etype, dsttype) in enumerate(self.meta_paths_u):
            new_g = g.edge_type_subgraph([etype])
            nodes = {new_g.dsttypes[0]: new_g.dstnodes()}
            sg = sample_neighbors(new_g, nodes, 500).to(self.device)
            # new_h = torch.cat((h[srctype].data,h[dsttype].data),0)
            embedding_u.append(self.gat_layers_node[i](sg, h[srctype].data,
                                                       h[dsttype].data))

        embedding_u = torch.stack(embedding_u, dim=1)
        embedding_u = self.semantic_attention(embedding_u)

        for i, (srctype, etype, dsttype) in enumerate(self.meta_paths_i):
            k = i + len(self.meta_paths_u)
            new_g = g.edge_type_subgraph([etype])
            nodes = {new_g.dsttypes[0]: new_g.dstnodes()}
            sg = sample_neighbors(new_g, nodes, 500).to(self.device)
            # new_h = torch.cat((h[srctype].data,h[dsttype].data),0)
            embedding_i.append(self.gat_layers_node[k](sg, h[srctype].data,
                                                       h[dsttype].data))
        embedding_i = torch.stack(embedding_i, dim=1)
        embedding_i = self.semantic_attention(embedding_i)

        embedding.append(embedding_u)
        embedding.append(embedding_i)

        return embedding


class TAHIN(nn.Module):
    def __init__(self, g, meta_path_patterns, userkey, itemkey, in_size, out_size, num_heads, dropout, device):
        super(TAHIN, self).__init__()
        # embeddings for different types of nodes, h0
        self.initializer = nn.init.xavier_uniform_
        self.g = g
        self.userkey = user_key = userkey
        self.itemkey = item_key = itemkey
        self.unum = self.g.num_nodes(userkey)
        self.inum = self.g.num_nodes(itemkey)
        self.user_meta_path_patterns = meta_path_patterns[user_key]
        self.item_meta_path_patterns = meta_path_patterns[item_key]
        self.device = device
        self.feature_dict = nn.ParameterDict({
            ntype: nn.Parameter(self.initializer(torch.empty(g.num_nodes(ntype), in_size))) for ntype in g.ntypes
        })

        # relational neighbor aggregation, this produces h1
        self.RelationalAGG = RelationalAGG(g, userkey, itemkey, in_size, out_size, 1, dropout, device)
        # self.RelationalAGG = RelationalAGG(g, in_size, out_size)
        # metapath-based aggregation modules for user and item, this produces h2
        self.meta_path_patterns = meta_path_patterns
        # one HANLayer for user, one HANLayer for item
        self.hans = nn.ModuleDict({
            key: HANLayer(value, in_size, out_size, num_heads, dropout) for key, value in
            self.meta_path_patterns.items()
        })

        all = torch.zeros((g.num_nodes(user_key), g.num_nodes(user_key)))
        for meta_path_pattern in self.user_meta_path_patterns:
            # 获取 CSR压缩稀疏列 (Compressed Sparse Column) 格式的邻接矩阵
            adj = dgl.metapath_reachable_graph(g, meta_path_pattern).adj_external(ctx='cpu', scipy_fmt='csr')
            temp = torch.zeros((g.num_nodes(user_key), g.num_nodes(user_key)))
            for i in range(adj.shape[0]):  # 每一行代表 与目标类型id=i相连的srcType的节点ID
                # 从 CSC 矩阵中获取第 i 列的所有非零行索引，即所有与列索引 i 有连接的节点
                start = adj.indptr[i]
                end = adj.indptr[i + 1]
                # 例如：如果 i = 296，数组可能是 [936, ...]
                # 表示 dsttype_id=296 的节点与 srctype 的节点 936 有交互
                temp[i, adj.indices[start:end]] += 1
            temp = temp / (temp.sum(axis=-1).reshape(-1, 1))
            temp = torch.where(torch.logical_or(torch.isnan(temp), torch.isinf(temp)), torch.tensor(0.0), temp)
            all += temp

        pos = torch.zeros((g.num_nodes(user_key), g.num_nodes(user_key)))
        pos_num = 50
        for i in range(len(all)):
            one = all[i].nonzero()
            if len(one) > pos_num:
                # 使用 torch.sort() 对张量进行排序，指定 dim=0 表示按列排序
                _, indices = torch.sort(all[i][one], dim=0, descending=True)
                indices = indices[:, 0]
                sele = one[indices[:pos_num]]
                pos[i, sele] = 1
            else:
                pos[i, one] = 1

        pos = sp.coo_matrix(pos)
        pos_uu = pos

        all = torch.zeros((g.num_nodes(itemkey), g.num_nodes(itemkey)))
        for meta_path_pattern in self.item_meta_path_patterns:
            # 获取 CSR压缩稀疏列 (Compressed Sparse Column) 格式的邻接矩阵
            adj = dgl.metapath_reachable_graph(g, meta_path_pattern).adj_external(ctx='cpu', scipy_fmt='csr')
            temp = torch.zeros((g.num_nodes(itemkey), g.num_nodes(itemkey)))
            for i in range(adj.shape[0]):  # 每一行代表 与目标类型id=i相连的srcType的节点ID
                # 从 CSC 矩阵中获取第 i 列的所有非零行索引，即所有与列索引 i 有连接的节点
                start = adj.indptr[i]
                end = adj.indptr[i + 1]
                # 例如：如果 i = 296，数组可能是 [936, ...]
                # 表示 dsttype_id=296 的节点与 srctype 的节点 936 有交互
                temp[i, adj.indices[start:end]] += 1
            temp = temp / (temp.sum(axis=-1).reshape(-1, 1))
            temp = torch.where(torch.logical_or(torch.isnan(temp), torch.isinf(temp)), torch.tensor(0.0), temp)
            all += temp

        pos = torch.zeros((g.num_nodes(itemkey), g.num_nodes(itemkey)))
        pos_num = 50
        for i in range(len(all)):
            one = all[i].nonzero()
            if len(one) > pos_num:
                # 使用 torch.sort() 对张量进行排序，指定 dim=0 表示按列排序
                _, indices = torch.sort(all[i][one], dim=0, descending=True)
                indices = indices[:, 0]
                sele = one[indices[:pos_num]]
                pos[i, sele] = 1
            else:
                pos[i, one] = 1

        pos = sp.coo_matrix(pos)
        pos_ii = pos

        num_u, num_i = self.unum, self.inum
        pos_uu = (pos_uu.row, pos_uu.col)
        pos_ii = (pos_ii.row, pos_ii.col)
        pos_u = torch.zeros(num_u, num_u, dtype=torch.int, device=device)
        pos_u[pos_uu] = 1
        pos_i = torch.zeros(num_i, num_i, dtype=torch.int, device=device)
        pos_i[pos_ii] = 1
        self.pos_u = pos_u
        self.pos_i = pos_i
        self.contrast = Contrast(in_size, 1, 0.5)
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

        # metapath-based aggregation, h2
        h2 = {}
        for key in self.meta_path_patterns.keys():
            h2[key] = self.hans[key](self.g, self.feature_dict[key])
        # user_soft = torch.cat((h1[user_key].unsqueeze(1), h2[user_key].unsqueeze(1)), 1)
        # user_emb = self.semantic_attention(user_soft)
        # item_soft = torch.cat((h1[item_key].unsqueeze(1), h2[item_key].unsqueeze(1)), 1)
        # item_emb = self.semantic_attention(item_soft)
        # update node embeddings

        loss_u = self.contrast(h1[0], h2[user_key], self.pos_u)
        loss_i = self.contrast(h1[1], h2[item_key], self.pos_i)

        user_emb = torch.cat((h1[0], h2[user_key]), 1)
        item_emb = torch.cat((h1[1], h2[item_key]), 1)

        # user_emb = torch.cat((user, h2[user_key]), 1)
        # item_emb = torch.cat((item, h2[item_key]), 1)
        # user_emb = torch.cat((h1[user_key], h2[user_key]), 1)
        # item_emb = torch.cat((h1[item_key], h2[item_key]), 1)
        user_emb = self.user_layer2(user_emb)
        item_emb = self.item_layer2(item_emb)
        # user_emb = self.user_layer2(torch.cat((user_emb, self.feature_dict[user_key]), 1))
        # item_emb = self.item_layer2(torch.cat((item_emb, self.feature_dict[item_key]), 1))
        # Relu
        # user_emb = F.relu_(user_emb)
        # item_emb = F.relu_(item_emb)
        # user_emb = F.relu_(user_emb)
        # item_emb = F.relu_(item_emb)
        # layer norm
        user_emb = self.layernorm(user_emb)
        item_emb = self.layernorm(item_emb)

        # obtain users/items embeddings and their interactions
        user_feat = user_emb[user_idx]
        item_feat = item_emb[item_idx]
        neg_item_feat = item_emb[neg_item_idx]

        return user_feat, item_feat, neg_item_feat, loss_u, loss_i
        # return user_feat, item_feat, neg_item_feat

    def bpr_loss(self, users, pos, neg):
        users_emb, pos_emb, neg_emb, loss_u, loss_i = self.forward(users, pos, neg)
        # users_emb, pos_emb, neg_emb = self.forward(users, pos, neg)
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss, loss_u, loss_i
        # return loss, reg_loss

    def getUsersRating(self, user_idx):
        item_idx = torch.Tensor(np.arange(self.inum)).long().to(self.device)
        users_emb, all_items, _, loss_u, loss_i = self.forward(user_idx, item_idx, None)
        # users_emb, all_items, _ = self.forward(user_idx, item_idx, None)
        rating = torch.matmul(users_emb, all_items.t())
        return rating
