import torch.optim as optim
import dgl
import torch
import numpy as np
import torch_sparse
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv, HGTConv, GraphConv


class GCN_layer(nn.Module):
    def __init__(self, args):
        super(GCN_layer, self).__init__()
        self.device = args.device

    def forward(self, features, Mat, mask):
        subset_sparse_tensor = Mat
        subset_features = features
        out_features = torch.sparse.mm(subset_sparse_tensor, subset_features)

        return out_features


class DCCF(nn.Module):
    def __init__(self, g, args):
        super(DCCF, self).__init__()
        self.g = g

        self.userkey = userkey = args.user_key
        self.itemkey = itemkey = args.item_key
        self.n_users = self.g.num_nodes(userkey)
        self.n_items = self.g.num_nodes(itemkey)
        n_nodes = self.n_users + self.n_items

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
        col_idx = [idx + self.n_users for idx in col_idx]
        # 转换为 NumPy 数组
        row_np = np.array(row_idx, dtype=np.int32)
        col_np = np.array(col_idx, dtype=np.int32)
        # 创建一个与 user_np 相同长度的全 1 数组
        ratings = np.ones_like(row_np, dtype=np.float32)
        # 构建新的稀疏矩阵
        tmp_adj = sp.csr_matrix((ratings, (row_np, col_np)), shape=(n_nodes, n_nodes), dtype=np.float32)
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
        self.G_indices, self.G_values = self._cal_sparse_adj()

        self.emb_dim = args.in_size
        self.n_layers = args.n_layers

    def _cal_sparse_adj(self):
        A_values = torch.ones(size=(len(self.all_h_list), 1)).view(-1).cuda()

        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=A_values,
                                             sparse_sizes=self.A_in_shape).cuda()

        D_values = A_tensor.sum(dim=1).pow(-0.5)

        G_indices, G_values = torch_sparse.spspmm(self.D_indices, D_values, self.A_indices, A_values,
                                                  self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])

        G_indices, G_values = torch_sparse.spspmm(G_indices, G_values, self.D_indices, D_values, self.A_in_shape[0],
                                                  self.A_in_shape[1], self.A_in_shape[1])

        return G_indices, G_values

    def forward(self, feature_dict):
        self.feature_dict = feature_dict
        all_embeddings = [torch.concat([self.feature_dict[self.userkey], self.feature_dict[self.itemkey]], dim=0)]
        gnn_embeddings = []
        int_embeddings = []

        for i in range(0, self.n_layers):
            # Graph-based Message Passing
            gnn_layer_embeddings = torch_sparse.spmm(self.G_indices, self.G_values, self.A_in_shape[0],
                                                     self.A_in_shape[1], all_embeddings[i])

            gnn_embeddings.append(gnn_layer_embeddings)
            # int_embeddings.append(int_layer_embeddings)
            all_embeddings.append(gnn_layer_embeddings + all_embeddings[i])

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.sum(all_embeddings, dim=1, keepdim=False)
        self.ua_embedding, self.ia_embedding = torch.split(all_embeddings, [self.n_users, self.n_items], 0)
        return self.ua_embedding, self.ia_embedding, gnn_embeddings, int_embeddings


class TAHIN(nn.Module):
    def __init__(self, g, args):
        super(TAHIN, self).__init__()
        self.g = g
        self.user_key = user_key = args.user_key
        self.item_key = item_key = args.item_key
        self.unum = self.n_users = self.g.num_nodes(user_key)
        self.inum = self.n_items = self.g.num_nodes(item_key)
        self.in_size = args.in_size
        self.device = args.device
        self.batch_size = args.batch_size
        self.n_layers = args.n_layers = 2
        self.han_layers = 2
        self.edge_bias = 0.5  # 观察到的偏差
        self.IB_rate = args.IB_rate
        self.sigma = args.sigma

        self.initializer = nn.init.xavier_uniform_
        self.feature_dict = nn.ParameterDict({
            user_key: nn.Parameter(self.initializer(torch.empty(g.num_nodes(user_key), args.in_size))),
            item_key: nn.Parameter(self.initializer(torch.empty(g.num_nodes(item_key), args.in_size)))
        })
        self.DCCF = DCCF(g, args)
        self.GCN_layer = GCN_layer(args)

        # metapath-based aggregation modules for user and item, this produces h2
        self.meta_path_patterns = args.meta_path_patterns
        # 存储邻接矩阵的字典
        self.adj_matrices = {
            user_key: {},
            item_key: {}
        }
        self.normalize_UI_adj = {
            self.user_key: {},
            self.item_key: {}
        }
        self.UI_interaction_rows = {
            user_key: {},
            item_key: {}
        }
        self.UI_interaction_cols = {
            user_key: {},
            item_key: {}
        }
        self.UI_interaction_values = {
            user_key: {},
            item_key: {}
        }
        # 生成并存储 user 侧的邻接矩阵
        for meta_path_pattern in self.meta_path_patterns[user_key]:
            subgraph = dgl.metapath_reachable_graph(g, meta_path_pattern)
            adj_matrix = subgraph.adj_external(ctx='cpu', scipy_fmt='coo')
            self.adj_matrices[user_key][str(meta_path_pattern)] = adj_matrix
            src, dst = subgraph.edges()
            self.UI_interaction_rows[user_key][str(meta_path_pattern)] = src
            self.UI_interaction_cols[user_key][str(meta_path_pattern)] = dst
        # 生成并存储 item 侧的邻接矩阵
        for meta_path_pattern in self.meta_path_patterns[item_key]:
            subgraph = dgl.metapath_reachable_graph(g, meta_path_pattern)
            adj_matrix = subgraph.adj_external(ctx='cpu', scipy_fmt='coo')
            self.adj_matrices[item_key][str(meta_path_pattern)] = adj_matrix
            src, dst = subgraph.edges()
            self.UI_interaction_rows[item_key][str(meta_path_pattern)] = src
            self.UI_interaction_cols[item_key][str(meta_path_pattern)] = dst

        # 生成并存储 user 侧的邻接矩阵
        for meta_path_pattern in self.meta_path_patterns[self.user_key]:
            subset_Mat = self.adj_matrices[self.user_key][str(meta_path_pattern)]
            subset_Mat = self.normalize_adj(subset_Mat)
            subset_sparse_tensor = self.sparse_mx_to_torch_sparse_tensor(subset_Mat).to(self.device)
            self.normalize_UI_adj[self.user_key][str(meta_path_pattern)] = subset_sparse_tensor
            self.UI_interaction_values[self.user_key][str(meta_path_pattern)] = subset_sparse_tensor._values()
        # 生成并存储 item 侧的邻接矩阵
        for meta_path_pattern in self.meta_path_patterns[self.item_key]:
            subset_Mat = self.adj_matrices[self.item_key][str(meta_path_pattern)]
            subset_Mat = self.normalize_adj(subset_Mat)
            subset_sparse_tensor = self.sparse_mx_to_torch_sparse_tensor(subset_Mat).to(self.device)
            self.normalize_UI_adj[self.item_key][str(meta_path_pattern)] = subset_sparse_tensor
            self.UI_interaction_values[self.item_key][str(meta_path_pattern)] = subset_sparse_tensor._values()

        # MaskMLP[user_key/item_key][meta_path_pattern]
        self.MaskMLP = nn.ModuleDict({
            key: nn.ModuleDict() for key in self.meta_path_patterns.keys()
        })
        for key in self.meta_path_patterns.keys():
            for meta_path_pattern in self.meta_path_patterns[key]:
                self.MaskMLP[key][str(meta_path_pattern)] = \
                    nn.Sequential(
                        nn.Linear(2 * self.in_size, self.in_size, bias=False),
                        nn.ReLU(),
                        nn.Linear(self.in_size, 1, bias=False)
                    )

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
        # adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return (d_mat_inv_sqrt).dot(adj).dot(d_mat_inv_sqrt).tocoo()

    def graph_reconstruction(self, mask_ratio=0.1):
        mask_normalize_UI_adj = {
            self.user_key: {},
            self.item_key: {}
        }
        for key in self.meta_path_patterns.keys():
            for meta_path_pattern in self.meta_path_patterns[key]:
                row, col = self.UI_interaction_rows[key][str(meta_path_pattern)], self.UI_interaction_cols[key][
                    str(meta_path_pattern)]
                num_edges = row.shape[0]
                num_masked_edges = int(num_edges * mask_ratio)

                # 随机选择要掩码的边的索引
                mask_indices = torch.randint(num_edges, (num_masked_edges,), device=row.device)

                # 获取随机选择的边的 row 和 col
                masked_row = row[mask_indices]
                masked_col = col[mask_indices]
                values = self.UI_interaction_values[key][str(meta_path_pattern)]
                mask_values = values[mask_indices]

                row_emb = self.feature_dict[key][masked_row]
                col_emb = self.feature_dict[key][masked_col]
                cat_emb = torch.cat([row_emb, col_emb], dim=1)

                logit = self.MaskMLP[key][str(meta_path_pattern)](cat_emb)
                logit = logit.view(-1)
                eps = torch.rand_like(logit)
                mask_gate_input = torch.log(eps) - torch.log(1 - eps)
                mask_gate_input = (logit + mask_gate_input) / 0.2
                mask_gate_input = torch.sigmoid(mask_gate_input) + self.edge_bias
                masked_values = mask_values * mask_gate_input
                # indices = torch.stack((row, col))
                # sparse_tensor = torch.sparse_coo_tensor(indices, masked_values, self.normalize_UI_adj[key][
                #     str(meta_path_pattern)].size()).coalesce()
                # Convert row and col to PyTorch tensors if they are not already
                if not isinstance(row, torch.Tensor):
                    row = torch.from_numpy(row).long()
                if not isinstance(col, torch.Tensor):
                    col = torch.from_numpy(col).long()
                indices = torch.stack((row, col))
                values = values.clone()  # 先克隆 `values`
                values[mask_indices] = masked_values
                shape = torch.Size(self.normalize_UI_adj[key][str(meta_path_pattern)].shape)
                mask_normalize_UI_adj[key][str(meta_path_pattern)] = torch.sparse.FloatTensor(indices, values,
                                                                                              shape).coalesce().to(self.device)

        return mask_normalize_UI_adj

    def create_masked_gcn_emb(self, mask_normalize_UI_adj):
        ua_embedding, ia_embedding, gnn_embeddings, int_embeddings = self.DCCF(self.feature_dict)
        # metapath-based aggregation, h2
        h2 = {}
        for i in range(self.han_layers):
            for key in self.meta_path_patterns.keys():
                embeddings = []
                for value in self.meta_path_patterns[key]:
                    value = str(value)
                    if i == 0:
                        embeddings.append(
                            self.GCN_layer(self.feature_dict[key], self.normalize_UI_adj[key][value], mask=False))
                    else:
                        embeddings.append(self.GCN_layer(h2[key], self.normalize_UI_adj[key][value], mask=False))
                # 对所有元路径的嵌入进行平均
                h2[key] = torch.mean(torch.stack(embeddings, dim=0), dim=0)
        user_emb = 0.5 * ua_embedding + 0.5 * h2[self.user_key]
        item_emb = 0.5 * ia_embedding + 0.5 * h2[self.item_key]

        # mask-metapath-based aggregation, h2
        h2 = {}
        for i in range(self.han_layers):
            for key in self.meta_path_patterns.keys():
                embeddings = []
                for value in self.meta_path_patterns[key]:
                    value = str(value)
                    if i == 0:
                        embeddings.append(
                            self.GCN_layer(self.feature_dict[key], mask_normalize_UI_adj[key][value], mask=True))
                    else:
                        embeddings.append(self.GCN_layer(h2[key], mask_normalize_UI_adj[key][value], mask=True))
                # 对所有元路径的嵌入进行平均
                h2[key] = torch.mean(torch.stack(embeddings, dim=0), dim=0)
        mask_user_emb = 0.5 * ua_embedding + 0.5 * h2[self.user_key]
        mask_item_emb = 0.5 * ia_embedding + 0.5 * h2[self.item_key]

        return user_emb, item_emb, mask_user_emb, mask_item_emb

    def HSIC_Graph(self, users, pos_items):
        users = torch.unique(users)
        items = torch.unique(pos_items)
        input_x = F.normalize(self.user_emb[users], p=2, dim=1)
        input_y = F.normalize(self.mask_user_emb[users], p=2, dim=1)
        Kx = kernel_matrix(input_x, self.sigma)  # σ
        Ky = kernel_matrix(input_y, self.sigma)
        loss_user = hsic(Kx, Ky, self.batch_size)
        input_x = F.normalize(self.item_emb[items], p=2, dim=1)
        input_y = F.normalize(self.mask_item_emb[items], p=2, dim=1)
        Kx = kernel_matrix(input_x, self.sigma)
        Ky = kernel_matrix(input_y, self.sigma)
        loss_item = hsic(Kx, Ky, self.batch_size)
        return loss_user + loss_item

    def forward(self, user_idx, item_idx, neg_item_idx):
        mask_normalize_UI_adj = self.graph_reconstruction()
        user_emb, item_emb, mask_user_emb, mask_item_emb = self.create_masked_gcn_emb(mask_normalize_UI_adj)
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.mask_user_emb = mask_user_emb
        self.mask_item_emb = mask_item_emb
        return mask_user_emb[user_idx], mask_item_emb[item_idx], mask_item_emb[neg_item_idx]

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

        # Compute the HSIC loss
        IB_loss = self.HSIC_Graph(users, pos) * self.IB_rate
        loss += IB_loss

        return loss, reg_loss

    def getUsersRating(self, user_idx):
        item_idx = torch.Tensor(np.arange(self.inum)).long().to(self.device)
        users_emb, all_items, _ = self.forward(user_idx, item_idx, None)
        rating = torch.matmul(users_emb, all_items.t())
        return rating


def kernel_matrix(x, sigma):
    return torch.exp((torch.matmul(x, x.t()) - 1) / sigma)


def hsic(Kx, Ky, m):
    Kxy = torch.matmul(Kx, Ky)
    h = torch.trace(Kxy) / m ** 2 + Kx.mean() * Ky.mean() - 2 * Kxy.mean() / m
    return h * (m / (m - 1)) ** 2


# Example of how to use the model for a forward pass (assuming args and dataset are defined):
# model = GBSR_SLightGCN(args, dataset)
# users = torch.LongTensor([0, 1, 2])  # example user indices
# pos_items = torch.LongTensor([3, 4, 5])  # example positive item indices
# neg_items = torch.LongTensor([6, 7, 8])  # example negative item indices
# loss, auc = model(users, pos_items, neg_items)
# print(f"Loss: {loss.item()}, AUC: {auc}")
# if __name__ == '__main__':
#     users = torch.LongTensor([0, 1, 2])  # example user indices
#     pos_items = torch.LongTensor([3, 4, 5])  # example positive item indices
#     neg_items = torch.LongTensor([6, 7, 8])  # example negative item indices
#     loss, auc = model(users, pos_items, neg_items)
#     print(f"Loss: {loss.item()}, AUC: {auc}")
