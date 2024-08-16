import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import dgl
import dgl.function as fn
from model.sgat import GAT
from dgl.nn.pytorch import GATConv, HGTConv, GraphConv
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
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class new_simHGCL(nn.Module):
    def __init__(self, g, args):
        super(new_simHGCL, self).__init__()
        # embeddings for different types of nodes, h0
        self.initializer = nn.init.xavier_uniform_
        self.g = g
        self.userkey = userkey = args.user_key
        self.itemkey = itemkey = args.item_key
        self.unum = self.g.num_nodes(userkey)
        self.inum = self.g.num_nodes(itemkey)
        self.in_size = in_size = args.in_size
        self.out_size = out_size = args.out_size
        self.num_heads = num_heads = args.num_heads
        self.dropout = dropout = args.dropout
        n_nodes = self.unum + self.inum
        self.device = args.device
        self.feature_dict = nn.ParameterDict({
            ntype: nn.Parameter(self.initializer(torch.empty(g.num_nodes(ntype), in_size))) for ntype in g.ntypes
        })

        # metapath-based aggregation modules for user and item, this produces h2
        self.meta_path_patterns = args.meta_path_patterns
        # one HANLayer for user, one HANLayer for item
        self.hans = nn.ModuleDict({
            key: HANLayer(value, in_size, out_size, num_heads, dropout) for key, value in
            self.meta_path_patterns.items()
        })

        # layers to combine h0, h1, and h2
        # used to update node embeddings
        self.user_layer1 = nn.Linear(out_size, out_size, bias=True)
        self.item_layer1 = nn.Linear(out_size, out_size, bias=True)
        row_idx = []
        col_idx = []
        adj = g.adj_external(ctx='cpu', scipy_fmt='csr', etype=args.ui_relation)
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
        col_idx = [idx + self.unum for idx in col_idx]
        # 转换为 NumPy 数组
        row_np = np.array(row_idx, dtype=np.int32)
        col_np = np.array(col_idx, dtype=np.int32)
        # 创建一个与 user_np 相同长度的全 1 数组
        ratings = np.ones_like(row_np, dtype=np.float32)
        # 构建新的稀疏矩阵
        tmp_adj = sp.csr_matrix((ratings, (row_np, col_np)), shape=(n_nodes, n_nodes), dtype=np.float32)
        self.ui_adj = tmp_adj + tmp_adj.T

        self.uiMat = self.ui_adj
        self.gcn = GCN_layer()

        self.momentum = 0.3
        # 把保存的历史嵌入矩阵
        self.u_target_his = torch.randn((self.unum, self.in_size), requires_grad=False).to(self.device)
        self.i_target_his = torch.randn((self.inum, self.in_size), requires_grad=False).to(self.device)

    def forward(self, user_idx, item_idx, neg_item_idx):
        user_key = self.userkey
        item_key = self.itemkey

        h2 = {}
        uiembedding0 = torch.cat([self.feature_dict[user_key], self.feature_dict[item_key]], 0)
        for l in range(2):
            if l == 0:
                uiembedding = self.gcn(uiembedding0, self.uiMat, self.device)
                for key in self.meta_path_patterns.keys():
                    h2[key] = self.hans[key](self.g, self.feature_dict[key])
            else:
                uiembedding = self.gcn(uiembedding, self.uiMat, self.device)
                for key in self.meta_path_patterns.keys():
                    h2[key] = self.hans[key](self.g, h2[key])

        self.ui_userEmbedding0, self.ui_itemEmbedding0 = \
            torch.split(uiembedding, [self.unum, self.inum])

        # update node embeddings
        user_emb = 0.5 * h2[user_key] + 0.5 * self.ui_userEmbedding0
        item_emb = 0.5 * h2[item_key] + 0.5 * self.ui_itemEmbedding0

        user_emb = user_emb[user_idx]
        item_emb = item_emb[item_idx]
        neg_item_feat = item_emb[neg_item_idx]
        # with torch.no_grad():
        #     u_target, i_target = self.u_target_his.clone()[user_idx, :], self.i_target_his.clone()[item_idx, :]
        #     u_target.detach()
        #     i_target.detach()
        #
        #     # self.momentum: 控制嵌入信息保留的比例
        #     u_target = u_target * self.momentum + user_emb.data * (1. - self.momentum)  # 结合当前嵌入和历史嵌入
        #     i_target = i_target * self.momentum + item_emb.data * (1. - self.momentum)
        #
        #     # 更新历史嵌入矩阵
        #     self.u_target_his[user_idx, :] = user_emb.clone()
        #     self.i_target_his[item_idx, :] = item_emb.clone()

        return user_emb, item_emb, neg_item_feat

    def loss_fn(self, p, z):  # cosine similarity
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()

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
        users_emb, all_items, _, _ = self.forward(user_idx, item_idx, None)
        rating = torch.matmul(users_emb, all_items.t())
        return rating

    def ssl_loss(self, data1, data2, index):
        index = torch.unique(index)
        embeddings1 = data1[index]
        embeddings2 = data2[index]
        norm_embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        norm_embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        pos_score = torch.sum(torch.mul(norm_embeddings1, norm_embeddings2), dim=1)
        all_score = torch.mm(norm_embeddings1, norm_embeddings2.T)
        pos_score = torch.exp(pos_score / 0.5)
        all_score = torch.sum(torch.exp(all_score / 0.5), dim=1)
        ssl_loss = (-torch.sum(torch.log(pos_score / ((all_score)))) / (len(index)))
        return ssl_loss


class GCN_layer(nn.Module):
    def __init__(self):
        super(GCN_layer, self).__init__()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        if type(sparse_mx) != sp.coo_matrix:
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data).float()
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return (d_mat_inv_sqrt).dot(adj).dot(d_mat_inv_sqrt).tocoo()

    def forward(self, features, Mat, device):
        subset_Mat = Mat
        subset_features = features
        subset_Mat = self.normalize_adj(subset_Mat)
        subset_sparse_tensor = self.sparse_mx_to_torch_sparse_tensor(subset_Mat).to(device)
        out_features = torch.spmm(subset_sparse_tensor, subset_features)

        return out_features
