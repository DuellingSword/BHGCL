import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class InitDisenLayer(nn.Module):
    def __init__(self, inp_dim, hid_dim, num_factors):
        super(InitDisenLayer, self).__init__()

        self.inp_dim = inp_dim
        self.hid_dim = (hid_dim // num_factors) * num_factors
        self.num_factors = num_factors

        self.factor_lins = nn.Linear(self.inp_dim, self.hid_dim)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, X):
        Z = self.factor_lins(X).view(-1, self.num_factors, self.hid_dim // self.num_factors)
        Z = F.normalize(torch.relu(Z), dim=2)
        return Z


# Feature disentangle layer
class RoutingLayer(nn.Module):
    def __init__(self, num_factors, routit, tau):
        super(RoutingLayer, self).__init__()
        self.num_factors = num_factors
        self.routit = routit
        self.tau = tau

    def forward(self, x, edges):
        m, src, trg = len(edges), edges[0], edges[1]
        n, k, delta_d = x.shape

        z = x  # neighbors' feature
        c = x  # node-neighbor attention aspect factor

        for t in range(self.routit):
            p = (z[trg] * c[src]).sum(dim=2, keepdim=True)  # update node-neighbor attention aspect factor
            p = F.softmax(p / self.tau, dim=1)  # (M, K, 1)
            weight_sum = (p * z[trg])  # weight sum (node attention * neighbors feature)
            c = z + torch.zeros_like(z).index_add_(0, src, weight_sum)  # update output embedding
            c = F.normalize(c, dim=2)  # embedding normalize aspect factor
        return c


class DisenGCN(nn.Module):
    def __init__(self, g, args):
        super(DisenGCN, self).__init__()
        self.initializer = nn.init.xavier_uniform_
        self.g = g
        hid_dim = in_size = args.in_size
        self.userkey = userkey = args.user_key
        self.itemkey = itemkey = args.item_key
        self.unum = self.g.num_nodes(userkey)
        self.inum = self.g.num_nodes(itemkey)
        n_nodes = self.unum + self.inum
        self.device = args.device
        self.feature_dict = nn.ParameterDict({
            ntype: nn.Parameter(self.initializer(torch.empty(g.num_nodes(ntype), in_size))) for ntype in g.ntypes
        })

        init_k = 8
        num_layers = 5
        routit = 7
        tau = 1
        delta_k = 0
        self.init_disen = InitDisenLayer(in_size, hid_dim, init_k)

        self.conv_layers = nn.ModuleList()
        k = init_k
        for l in range(num_layers):
            fac_dim = hid_dim // k
            self.conv_layers.append(RoutingLayer(k, routit, tau))
            inp_dim = fac_dim * k
            k -= delta_k

        self.dropout = args.dropout
        # self.classifier = nn.Linear(inp_dim, num_classes)
        # nn.init.xavier_uniform_(self.classifier.weight)
        # nn.init.zeros_(self.classifier.bias)
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

        # 将 NumPy 数组转换为 PyTorch tensors
        row_tensor = torch.from_numpy(row_np).long()
        col_tensor = torch.from_numpy(col_np).long()
        # 堆叠 tensors 创建一个 2xN 的 tensor
        self.edges_tensor = torch.stack((row_tensor, col_tensor), dim=0).to(args.device)

    def _dropout(self, X):
        return F.dropout(X, p=self.dropout, training=self.training)

    def forward(self, X, edges, users, pos, neg):
        Z = self.init_disen(X)
        for disen_conv in self.conv_layers:
            Z = disen_conv(Z, edges)
            Z = self._dropout(torch.relu(Z))
        # Z = self.classifier(Z.reshape(len(Z), -1))
        Z = Z.reshape(len(Z), -1)
        user_embeddings, item_embeddings = torch.split(Z, [self.unum, self.inum])
        return user_embeddings[users], item_embeddings[pos], item_embeddings[neg]

    def bpr_loss(self, users, pos, neg):
        X = torch.cat([self.feature_dict[self.userkey], self.feature_dict[self.itemkey]], dim=0)
        users_emb, pos_emb, neg_emb = self.forward(X, self.edges_tensor, users, pos, neg)
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
        X = torch.cat([self.feature_dict[self.userkey], self.feature_dict[self.itemkey]], dim=0)
        users_emb, all_items, _ = self.forward(X, self.edges_tensor, user_idx, item_idx, None)
        rating = torch.matmul(users_emb, all_items.t())
        return rating