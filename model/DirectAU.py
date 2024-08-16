import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectAU(nn.Module):
    def __init__(self, g, args):
        super(DirectAU, self).__init__()
        self.initializer = nn.init.xavier_uniform_
        self.g = g
        self.user_key = args.user_key
        self.item_key = args.item_key
        self.ui_relation = ui_relation = args.ui_relation
        self.n_users = self.g.num_nodes(args.user_key)
        self.n_items = self.g.num_nodes(args.item_key)
        n_nodes = self.n_users + self.n_items
        self.device = args.device

        # load parameters info
        self.embedding_size = args.in_size
        self.gamma = 1
        self.encoder_name = 'LightGCN'

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
            col_idx = [idx + self.n_users for idx in col_idx]
            # 转换为 NumPy 数组
            row_np = np.array(row_idx, dtype=np.int32)
            col_np = np.array(col_idx, dtype=np.int32)
            # 创建一个与 user_np 相同长度的全 1 数组
            ratings = np.ones_like(row_np, dtype=np.float32)
            # 构建新的稀疏矩阵
            tmp_adj = sp.coo_matrix((ratings, (row_np, col_np)), shape=(n_nodes, n_nodes), dtype=np.float32)
            self.ui_adj = (tmp_adj + tmp_adj.T).tocoo()
            self.interaction_matrix = self.ui_adj
            # self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
            self.norm_adj = self.get_norm_adj_mat().to(self.device)
            self.encoder = LGCNEncoder(self.n_users, self.n_items, self.embedding_size, self.norm_adj, self.n_layers)
        else:
            raise ValueError('Non-implemented Encoder.')

    def get_norm_adj_mat(self):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(self.interaction_matrix)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        norm_adj = (d_mat_inv_sqrt).dot(adj).dot(d_mat_inv_sqrt).tocoo()

        """Convert a scipy sparse matrix to a torch sparse tensor."""
        matrix = norm_adj.tocoo()
        indices = np.vstack((matrix.row, matrix.col))
        values = matrix.data
        shape = matrix.shape
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)

        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def forward(self, user, item):
        user_e, item_e = self.encoder(user, item)
        return F.normalize(user_e, dim=-1), F.normalize(item_e, dim=-1)

    @staticmethod
    def alignment(x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    @staticmethod
    def uniformity(x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_e, item_e = self.forward(user, item)
        align = self.alignment(user_e, item_e)
        uniform = self.gamma * (self.uniformity(user_e) + self.uniformity(item_e)) / 2

        return align + uniform

    def bpr_loss(self, users, pos, neg):
        users_emb, pos_emb = self.forward(users, pos)
        align = self.alignment(users_emb, pos_emb)
        uniform = self.gamma * (self.uniformity(users_emb) + self.uniformity(pos_emb)) / 2
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2)) / float(len(users))
        # BPR loss
        # pos_scores = torch.mul(users_emb, pos_emb)
        # pos_scores = torch.sum(pos_scores, dim=1)
        # neg_scores = torch.mul(users_emb, neg_emb)
        # neg_scores = torch.sum(neg_scores, dim=1)
        # loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        loss = align + uniform
        return loss, reg_loss

    def getUsersRating(self, user_idx):
        item_idx = torch.Tensor(np.arange(self.n_items)).long().to(self.device)
        users_emb, all_items = self.forward(user_idx, item_idx)
        rating = torch.matmul(users_emb, all_items.t())
        return rating


class MFEncoder(nn.Module):
    def __init__(self, user_num, item_num, emb_size):
        super(MFEncoder, self).__init__()
        self.user_embedding = nn.Embedding(user_num, emb_size)
        self.item_embedding = nn.Embedding(item_num, emb_size)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

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
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

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
        return user_all_embeddings, item_all_embeddings

    def forward(self, user_id, item_id):
        user_all_embeddings, item_all_embeddings = self.get_all_embeddings()
        u_embed = user_all_embeddings[user_id]
        i_embed = item_all_embeddings[item_id]
        return u_embed, i_embed
