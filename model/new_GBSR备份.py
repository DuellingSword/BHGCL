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

    def forward(self, features, Mat):
        subset_Mat = Mat
        subset_features = features
        subset_Mat = self.normalize_adj(subset_Mat)
        subset_sparse_tensor = self.sparse_mx_to_torch_sparse_tensor(subset_Mat).to(self.device)
        out_features = torch.spmm(subset_sparse_tensor, subset_features)

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
        self.device = args.device
        self.n_layers = args.n_layers = 2
        self.han_layers = 2

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
        # 生成并存储 user 侧的邻接矩阵
        for meta_path_pattern in self.meta_path_patterns[user_key]:
            subgraph = dgl.metapath_reachable_graph(g, meta_path_pattern)
            adj_matrix = subgraph.adj_external(ctx='cpu', scipy_fmt='coo')
            self.adj_matrices[user_key][str(meta_path_pattern)] = adj_matrix

        # 生成并存储 item 侧的邻接矩阵
        for meta_path_pattern in self.meta_path_patterns[item_key]:
            subgraph = dgl.metapath_reachable_graph(g, meta_path_pattern)
            adj_matrix = subgraph.adj_external(ctx='cpu', scipy_fmt='coo')
            self.adj_matrices[item_key][str(meta_path_pattern)] = adj_matrix

    def beforeForward(self):
        1

    def forward(self, user_idx, item_idx, neg_item_idx):
        ua_embedding, ia_embedding, gnn_embeddings, int_embeddings = self.DCCF(self.feature_dict)
        # metapath-based aggregation, h2
        h2 = {}
        for i in range(self.han_layers):
            for key in self.meta_path_patterns.keys():
                embeddings = []
                for value in self.meta_path_patterns[key]:
                    value = str(value)
                    if i == 0:
                        embeddings.append(self.GCN_layer(self.feature_dict[key], self.adj_matrices[key][value]))
                    else:
                        embeddings.append(self.GCN_layer(h2[key], self.adj_matrices[key][value]))
                # 对所有元路径的嵌入进行平均
                h2[key] = torch.mean(torch.stack(embeddings, dim=0), dim=0)
        user_emb = 0.5 * ua_embedding + 0.5 * h2[self.user_key]
        item_emb = 0.5 * ia_embedding + 0.5 * h2[self.item_key]

        return user_emb[user_idx], item_emb[item_idx], item_emb[neg_item_idx]

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
        return rating



def kernel_matrix(x, sigma):
    return torch.exp((torch.matmul(x, x.t()) - 1) / sigma)


def hsic(Kx, Ky, m):
    Kxy = torch.matmul(Kx, Ky)
    h = torch.trace(Kxy) / m ** 2 + Kx.mean() * Ky.mean() - 2 * Kxy.mean() / m
    return h * (m / (m - 1)) ** 2


class GBSR_SLightGCN(nn.Module):
    def __init__(self, args, dataset):
        super(GBSR_SLightGCN, self).__init__()
        self.gcn_layer = args.gcn_layer
        self.sigma = args.sigma
        self.beta = args.beta
        self.num_inter = len(dataset.training_user) * 2
        self.adj_indices, self.adj_values, adj_shape = dataset.convert_csr_to_sparse_tensor_inputs(dataset.uu_i_matrix)
        self.social_index = dataset.social_index_in_social_lightgcn()
        self.social_u = self.adj_indices[self.social_index, 0]
        self.social_v = self.adj_indices[self.social_index, 1]
        self.social_weight = self.adj_values[self.social_index]

        self.adj_matrix = torch.sparse_coo_tensor(self.adj_indices.t(), self.adj_values, adj_shape).coalesce()
        self.Mask_MLP1 = nn.Linear(args.latent_dim, args.latent_dim)
        self.Mask_MLP2 = nn.Linear(args.latent_dim, 1)
        self.Mask_MLP3 = nn.Linear(args.latent_dim, args.latent_dim)
        self.Mask_MLP4 = nn.Linear(args.latent_dim, 1)
        self.Mask_MLP5 = nn.Linear(args.latent_dim, args.latent_dim)
        self.Mask_MLP6 = nn.Linear(args.latent_dim, 1)
        self.Mask_MLP7 = nn.Linear(args.latent_dim, args.latent_dim)
        self.Mask_MLP8 = nn.Linear(args.latent_dim, 1)
        self.edge_bias = args.edge_bias
        self.user_latent_emb = nn.Parameter(torch.Tensor(args.num_users, args.latent_dim))
        self.item_latent_emb = nn.Parameter(torch.Tensor(args.num_items, args.latent_dim))
        nn.init.xavier_uniform_(self.user_latent_emb)
        nn.init.xavier_uniform_(self.item_latent_emb)
        self.lr = args.lr
        self.batch_size = args.batch_size
        self._build_graph()

    def graph_reconstruction(self, ego_emb, layer):
        row, col = self.social_u, self.social_v
        row_emb = ego_emb[row]
        col_emb = ego_emb[col]
        cat_emb = torch.cat([row_emb, col_emb], dim=1)
        if layer == 0:
            logit = self.Mask_MLP2(F.relu(self.Mask_MLP1(cat_emb)))
        elif layer == 1:
            logit = self.Mask_MLP4(F.relu(self.Mask_MLP3(cat_emb)))
        elif layer == 2:
            logit = self.Mask_MLP6(F.relu(self.Mask_MLP5(cat_emb)))
        elif layer == 3:
            logit = self.Mask_MLP8(F.relu(self.Mask_MLP7(cat_emb)))
        logit = logit.view(-1)
        eps = torch.rand_like(logit)
        mask_gate_input = torch.log(eps) - torch.log(1 - eps)
        mask_gate_input = (logit + mask_gate_input) / 0.2
        mask_gate_input = torch.sigmoid(mask_gate_input) + self.edge_bias
        masked_values = self.social_weight * mask_gate_input
        masked_values_all = self.adj_values.clone()
        masked_values_all[self.social_index] = masked_values
        masked_adj_matrix = torch.sparse_coo_tensor(self.adj_matrix.indices(), masked_values_all,
                                                    self.adj_matrix.size()).coalesce()
        return masked_adj_matrix, mask_gate_input.mean(), mask_gate_input

    def _create_lightgcn_emb(self, ego_emb):
        all_emb = [ego_emb]
        for _ in range(self.gcn_layer):
            tmp_emb = torch.sparse.mm(self.adj_matrix, all_emb[-1])
            all_emb.append(tmp_emb)
        all_emb = torch.stack(all_emb, dim=1)
        mean_emb = all_emb.mean(dim=1)
        return mean_emb

    def _create_masked_lightgcn_emb(self, ego_emb, masked_adj_matrix):
        all_emb = [ego_emb]
        all_emb_masked = [ego_emb]
        for _ in range(self.gcn_layer):
            tmp_emb = torch.sparse.mm(self.adj_matrix, all_emb[-1])
            all_emb.append(tmp_emb)
            cur_emb = torch.sparse.mm(masked_adj_matrix, all_emb_masked[-1])
            all_emb_masked.append(cur_emb)
        all_emb = torch.stack(all_emb, dim=1)
        mean_emb = all_emb.mean(dim=1)
        out_user_emb, out_item_emb = mean_emb.split([self.num_user, self.num_item], dim=0)
        all_emb_masked = torch.stack(all_emb_masked, dim=1)
        mean_emb_masked = all_emb_masked.mean(dim=1)
        out_user_emb_masked, out_item_emb_masked = mean_emb_masked.split([self.num_user, self.num_item], dim=0)
        return out_user_emb, out_item_emb, out_user_emb_masked, out_item_emb_masked

    def _create_multiple_masked_lightgcn_emb(self, ego_emb):
        all_emb = [ego_emb]
        all_emb_masked = [ego_emb]
        for i in range(self.gcn_layer):
            masked_adj_matrix, _ = self.graph_reconstruction(all_emb[-1], i)
            tmp_emb = torch.sparse.mm(self.adj_matrix, all_emb[-1])
            all_emb.append(tmp_emb)
            cur_emb = torch.sparse.mm(masked_adj_matrix, all_emb_masked[-1])
            all_emb_masked.append(cur_emb)
        all_emb = torch.stack(all_emb, dim=1)
        mean_emb = all_emb.mean(dim=1)
        out_user_emb, out_item_emb = mean_emb.split([self.num_user, self.num_item], dim=0)
        all_emb_masked = torch.stack(all_emb_masked, dim=1)
        mean_emb_masked = all_emb_masked.mean(dim=1)
        out_user_emb_masked, out_item_emb_masked = mean_emb_masked.split([self.num_user, self.num_item], dim=0)
        return out_user_emb, out_item_emb, out_user_emb_masked, out_item_emb_masked

    def _build_graph(self):
        ego_emb = torch.cat([self.user_latent_emb, self.item_latent_emb], dim=0)
        self.masked_adj_matrix, self.masked_gate_input, self.masked_values = self.graph_reconstruction(ego_emb, 0)
        self.user_emb_old, self.item_emb_old, self.user_emb, self.item_emb = \
            self._create_masked_lightgcn_emb(ego_emb, self.masked_adj_matrix)
        self.ranking_loss, self.regu_loss, self.auc = self.compute_bpr_loss(self.user_emb, self.item_emb)
        self.IB_loss = self.HSIC_Graph() * self.beta
        self.loss = self.ranking_loss + self.regu_loss + self.IB_loss
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def HSIC_Graph(self):
        users = torch.unique(self.users)
        items = torch.unique(self.pos_items)
        input_x = F.normalize(self.user_emb_old[users], p=2, dim=1)
        input_y = F.normalize(self.user_emb[users], p=2, dim=1)
        Kx = kernel_matrix(input_x, self.sigma)
        Ky = kernel_matrix(input_y, self.sigma)
        loss_user = hsic(Kx, Ky, self.batch_size)
        input_x = F.normalize(self.item_emb_old[items], p=2, dim=1)
        input_y = F.normalize(self.item_emb[items], p=2, dim=1)
        Kx = kernel_matrix(input_x, self.sigma)
        Ky = kernel_matrix(input_y, self.sigma)
        loss_item = hsic(Kx, Ky, self.batch_size)
        return loss_user + loss_item

    def compute_bpr_loss(self, user_emb, item_emb):
        # implement BPR loss computation
        pass

    def forward(self, users, pos_items, neg_items):
        ego_emb = torch.cat([self.user_latent_emb, self.item_latent_emb], dim=0)
        masked_adj_matrix, masked_gate_input, _ = self.graph_reconstruction(ego_emb, 0)
        user_emb_old, item_emb_old, user_emb, item_emb = self._create_masked_lightgcn_emb(ego_emb, masked_adj_matrix)

        # Extract the embeddings for the users and items involved in the batch
        user_embedding = user_emb[users]
        pos_item_embedding = item_emb[pos_items]
        neg_item_embedding = item_emb[neg_items]

        # Compute the BPR loss
        ranking_loss, regu_loss, auc = self.compute_bpr_loss(user_embedding, pos_item_embedding, neg_item_embedding)

        # Compute the HSIC loss
        IB_loss = self.HSIC_Graph() * self.beta

        # Total loss
        total_loss = ranking_loss + regu_loss + IB_loss

        return total_loss, auc

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
