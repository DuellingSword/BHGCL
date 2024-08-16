# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


# embedding_size: 2048
# encoder: LightGCN
# n_layers: 2
# training_neg_sample_num: 0
#
# a: 1
# degree: 4
# polyc: 1e-7
# poly_coeff: 0.2
# all_bt_coeff: 1
# bt_coeff: 0.01
# mom_coeff: 10
# momentum: 0.3


class RecDCL(nn.Module):
    def __init__(self, g, args):
        super(RecDCL, self).__init__()
        self.g = g
        # load parameters info
        self.batch_size = args.batch_size
        self.embedding_size = args.in_size
        ui_relation = args.ui_relation
        self.userkey = userkey = args.user_key
        self.itemkey = itemkey = args.item_key
        self.n_users = self.unum = self.g.num_nodes(userkey)
        self.n_items = self.inum = self.g.num_nodes(itemkey)
        n_nodes = self.unum + self.inum
        self.encoder_name = 'LightGCN'
        self.device = device = args.device
        # self.reg_weight = args.

        self.a = 1
        self.polyc = 1e-7
        self.degree = 4
        self.poly_coeff = 0.2
        self.bt_coeff = 0.01
        self.all_bt_coeff = 1
        self.mom_coeff = 10
        self.momentum = 0.3

        # define layers and loss
        if self.encoder_name == 'MF':
            self.encoder = MFEncoder(self.n_users, self.n_items, self.embedding_size)
        elif self.encoder_name == 'LightGCN':
            self.n_layers = 2
            row_idx = []
            col_idx = []
            adj = g.adj_external(ctx='cpu', scipy_fmt='csr', etype=ui_relation)
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
            tmp_adj = sp.coo_matrix((ratings, (row_np, col_np)), shape=(n_nodes, n_nodes), dtype=np.float32)
            self.interaction_matrix = tmp_adj + tmp_adj.T
            # self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
            # self.norm_adj = self.get_norm_adj_mat().to(self.device)
            self.norm_adj = self.normalize_adj(self.interaction_matrix)
            self.encoder = LGCNEncoder(self.n_users, self.n_items, self.embedding_size, self.norm_adj, self.n_layers)
        else:
            raise ValueError('Non-implemented Encoder.')

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        self.bn = nn.BatchNorm1d(self.embedding_size, affine=False)

        layers = []
        embs = str(self.embedding_size) + '-' + str(self.embedding_size) + '-' + str(self.embedding_size)
        sizes = [self.embedding_size] + list(map(int, embs.split('-')))  # in_size - in_size - in_size - in_size
        for i in range(len(sizes) - 2):  # 2: 0->1
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # parameters initialization
        # self.apply(xavier_normal_initialization)

        self.predictor = nn.Linear(self.embedding_size, self.embedding_size)

        # 把保存的历史嵌入矩阵
        self.u_target_his = torch.randn((self.n_users, self.embedding_size), requires_grad=False).to(self.device)
        self.i_target_his = torch.randn((self.n_items, self.embedding_size), requires_grad=False).to(self.device)

    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        # covert norm_adj matrix to tensor
        L = (d_mat_inv_sqrt).dot(adj).dot(d_mat_inv_sqrt).tocoo()
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape)).to(self.device)
        return SparseL

    def forward(self, user, item):
        user_e, item_e, lightgcn_all_embeddings = self.encoder(user, item)

        with torch.no_grad():
            u_target, i_target = self.u_target_his.clone()[user, :], self.i_target_his.clone()[item, :]
            u_target.detach()
            i_target.detach()

            # self.momentum: 控制嵌入信息保留的比例
            u_target = u_target * self.momentum + user_e.data * (1. - self.momentum)  # 结合当前嵌入和历史嵌入
            i_target = i_target * self.momentum + item_e.data * (1. - self.momentum)

            # 更新历史嵌入矩阵
            self.u_target_his[user, :] = user_e.clone()
            self.i_target_his[item, :] = item_e.clone()

        return user_e, item_e, lightgcn_all_embeddings, u_target, i_target

    @staticmethod
    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def bt(self, x, y):
        user_e = self.projector(x)
        item_e = self.projector(y)
        c = self.bn(user_e).T @ self.bn(item_e)
        c.div_(user_e.size()[0])
        # sum the cross-correlation matrix
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().div(self.embedding_size)
        off_diag = self.off_diagonal(c).pow_(2).sum().div(self.embedding_size)
        bt = on_diag + self.bt_coeff * off_diag
        return bt

    def poly_feature(self, x):
        user_e = self.projector(x)
        xx = self.bn(user_e).T @ self.bn(user_e)
        poly = (self.a * xx + self.polyc) ** self.degree
        return poly.mean().log()

    def loss_fn(self, p, z):  # cosine similarity
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()

    def bpr_loss(self, users, pos, neg):
        user = users
        item = pos
        user_e, item_e, embeddings_list, u_target, i_target = self.forward(user, item)
        user_e_n, item_e_n = F.normalize(user_e, dim=-1), F.normalize(item_e, dim=-1)
        user_e, item_e = self.predictor(user_e), self.predictor(item_e)
        if self.all_bt_coeff == 0:  # UIBT
            bt_loss = 0.0
        else:  # Eliminate redundancy between users and items.
            bt_loss = self.bt(user_e_n, item_e_n)

        if self.poly_coeff == 0:  # UUII
            poly_loss = 0.0
        else:  # Eliminate redundancy within users and items.
            poly_loss = self.poly_feature(user_e_n) / 2 + self.poly_feature(item_e_n) / 2

        if self.mom_coeff == 0:  # BCL
            mom_loss = 0.0
        else:
            mom_loss = self.loss_fn(user_e, i_target) / 2 + self.loss_fn(item_e, u_target) / 2
        loss = self.all_bt_coeff * bt_loss + poly_loss * self.poly_coeff + mom_loss * self.mom_coeff
        reg_loss = torch.tensor(0.0, requires_grad=True).to(self.device)
        return loss, reg_loss

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e, embeddings_list, u_target, i_target = self.forward(user, item)
        user_e_n, item_e_n = F.normalize(user_e, dim=-1), F.normalize(item_e, dim=-1)
        user_e, item_e = self.predictor(user_e), self.predictor(item_e)
        if self.all_bt_coeff == 0:
            bt_loss = 0.0
        else:
            bt_loss = self.bt(user_e_n, item_e_n)

        if self.poly_coeff == 0:
            poly_loss = 0.0
        else:
            poly_loss = self.poly_feature(user_e_n) / 2 + self.poly_feature(item_e_n) / 2

        if self.mom_coeff == 0:
            mom_loss = 0.0
        else:
            mom_loss = self.loss_fn(user_e, i_target) / 2 + self.loss_fn(item_e, u_target) / 2

        return self.all_bt_coeff * bt_loss + poly_loss * self.poly_coeff + mom_loss * self.mom_coeff

    def getUsersRating(self, user_idx):
        item_idx = torch.Tensor(np.arange(self.inum)).long().to(self.device)
        users_emb, all_items, _, _, _ = self.forward(user_idx, item_idx)
        rating = torch.matmul(users_emb, all_items.t())
        return rating

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.encoder_name == 'LightGCN':
            if self.restore_user_e is None or self.restore_item_e is None:
                self.restore_user_e, self.restore_item_e, _ = self.encoder.get_all_embeddings()
            user_e = self.restore_user_e[user]
            all_item_e = self.restore_item_e
        else:
            user_e = self.encoder.user_embedding(user)
            all_item_e = self.encoder.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)


class MFEncoder(nn.Module):
    def __init__(self, user_num, item_num, emb_size):
        super(MFEncoder, self).__init__()
        self.user_embedding = nn.Embedding(user_num, emb_size)
        self.item_embedding = nn.Embedding(item_num, emb_size)
        # no pretrain
        # xavier uniform is a better choice than normal for training model
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)

    def forward(self, user_id, item_id):
        u_embed = self.user_embedding(user_id)
        i_embed = self.item_embedding(item_id)
        return u_embed, i_embed

    def get_all_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        return user_embeddings, item_embeddings


class LGCNEncoder(nn.Module):
    def __init__(self, user_num, item_num, emb_size, norm_adj, n_layers=3):
        super(LGCNEncoder, self).__init__()
        self.n_users = user_num
        self.n_items = item_num
        self.n_layers = n_layers
        self.norm_adj = norm_adj

        self.user_embedding = torch.nn.Embedding(user_num, emb_size)
        self.item_embedding = torch.nn.Embedding(item_num, emb_size)
        # no pretrain
        # xavier uniform is a better choice than normal for training model
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def get_all_embeddings(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings, lightgcn_all_embeddings

    def forward(self, user_id, item_id):
        user_all_embeddings, item_all_embeddings, lightgcn_all_embeddings = self.get_all_embeddings()
        u_embed = user_all_embeddings[user_id]
        i_embed = item_all_embeddings[item_id]
        return u_embed, i_embed, lightgcn_all_embeddings
