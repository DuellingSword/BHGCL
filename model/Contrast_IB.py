import torch
import dgl
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import random_walk


class Contrast_IB(nn.Module):
    def __init__(self, g, args):
        super(Contrast_IB, self).__init__()
        self.device = args.device
        self.user_key = args.user_key
        self.item_key = args.item_key
        self.in_size = args.in_size
        self.out_size = args.out_size
        self.walk_length = args.walk_length
        self.choosing_tmp = args.choosing_tmp
        self.ssl_temp = args.ssl_temp
        self.LCN_layer = args.LCN_layer
        self.ui_relation = args.ui_relation
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
        self.latent_size = args.in_size
        self.user_num = g.num_nodes(self.user_key)
        self.item_num = g.num_nodes(self.item_key)
        n_nodes = self.user_num + self.item_num

        """ 提取 ui 信息 """
        row_idx = []
        col_idx = []
        adj = g.adj_external(ctx='cpu', scipy_fmt='csr', etype=self.ui_relation)
        for i in range(adj.shape[0]):  # 每一行代表 与目标类型id=i相连的srcType的节点ID
            # 从 CSC 矩阵中获取第 i 列的所有非零行索引，即所有与列索引 i 有连接的节点
            start = adj.indptr[i]
            end = adj.indptr[i + 1]
            cols = adj.indices[start:end]
            # 记录行索引和列索引
            for col in cols:
                row_idx.append(i)
                col_idx.append(col)

        """ 构建归一化邻接矩阵, 大小为 (self.user_num + self.item_num, self.user_num + self.item_num) """
        temp_col_idx = [idx + self.user_num for idx in col_idx]  # 将列索引转换成物品索引，确保它们在用户索引之后
        col_idx = temp_col_idx.copy()
        for i, col in enumerate(temp_col_idx):  # 反转边来添加物品到用户的边
            row_idx.append(col)  # 物品到用户的边
            col_idx.append(row_idx[i])  # 对应的用户节点
        ui_index = [row_idx, col_idx]
        value = [1.0] * len(row_idx)
        self.joint_adjaceny_matrix = torch.sparse_coo_tensor(ui_index, value, (n_nodes, n_nodes))
        degree = torch.sparse.sum(self.joint_adjaceny_matrix, dim=1).to_dense()
        degree = torch.pow(degree + 1e-8, -0.5)
        degree[torch.isinf(degree)] = 0
        D_inverse = torch.diag(degree, diagonal=0).to_sparse()  # D^(-1/2) * A * D^(-1/2)
        self.joint_adjaceny_matrix_normal = torch.sparse.mm(torch.sparse.mm(D_inverse, self.joint_adjaceny_matrix),
                                                            D_inverse).coalesce().to(self.device)
        self.row = torch.tensor(row_idx).long()
        self.col = torch.tensor(col_idx).long()

        """ 构建随机游走矩阵 """
        start = torch.arange(n_nodes)
        walk = random_walk(torch.tensor(row_idx).long(), torch.tensor(col_idx).long(), start,
                           walk_length=self.walk_length)
        self.rw_adj = torch.zeros((n_nodes, n_nodes))
        self.rw_adj = torch.scatter(self.rw_adj, 1, walk, 1).to_sparse()
        degree = torch.sparse.sum(self.rw_adj, dim=1).to_dense()
        degree = torch.pow(degree + 1e-8, -1)
        degree[torch.isinf(degree)] = 0
        D_inverse = torch.diag(degree, diagonal=0).to_sparse()
        # 左乘 self.rw_adj 来进行归一化，确保了从每个节点出发的转移概率加起来等于1
        # size=(n_user+n_item, n_user+n_item)
        self.rw_adj = torch.sparse.mm(D_inverse, self.rw_adj).coalesce().to(self.device)

    def predict(self, cur_user_emb, cur_item_emb, original_user_emb, original_item_emb, users, items):
        """ 重参数化 """
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
                                                        self.user_num + self.item_num, self.user_num + self.item_num,
                                                        cur_embedding_edge_drop)
            all_embeddings_edge_drop.append(cur_embedding_edge_drop)

        all_embeddings_edge_drop = torch.stack(all_embeddings_edge_drop, dim=0)
        all_embeddings_edge_drop = torch.mean(all_embeddings_edge_drop, dim=0, keepdim=False)
        user_embeddings_edge_drop, item_embeddings_edge_drop = torch.split(all_embeddings_edge_drop,
                                                                           [self.user_num, self.item_num])
        edge_reg = edge_reg / self.LCN_layer

        """ node_dropout_view """
        # user_num + item_num * embedding_size
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
            node_reg += node_mask.sum() / (self.user_num + self.item_num)

        all_embeddings_node_drop = torch.stack(all_embeddings_node_drop, dim=0)
        all_embeddings_node_drop = torch.mean(all_embeddings_node_drop, dim=0, keepdim=False)
        user_embeddings_node_drop, item_embeddings_node_drop = torch.split(all_embeddings_node_drop,
                                                                           [self.user_num, self.item_num])
        node_reg = node_reg / self.LCN_layer

        return user_embeddings_edge_drop[users], item_embeddings_edge_drop[items], user_embeddings_node_drop[users], item_embeddings_node_drop[items]

    def forward(self, cur_user_emb, cur_item_emb, original_user_emb, original_item_emb,
                user_id, pos_item, neg_item):
        """ 重参数化 """
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
                                                        self.user_num + self.item_num, self.user_num + self.item_num,
                                                        cur_embedding_edge_drop)
            all_embeddings_edge_drop.append(cur_embedding_edge_drop)

        all_embeddings_edge_drop = torch.stack(all_embeddings_edge_drop, dim=0)
        all_embeddings_edge_drop = torch.mean(all_embeddings_edge_drop, dim=0, keepdim=False)
        user_embeddings_edge_drop, item_embeddings_edge_drop = torch.split(all_embeddings_edge_drop,
                                                                           [self.user_num, self.item_num])
        edge_reg = edge_reg / self.LCN_layer

        """ node_dropout_view """
        # user_num + item_num * embedding_size
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
            node_reg += node_mask.sum() / (self.user_num + self.item_num)

        all_embeddings_node_drop = torch.stack(all_embeddings_node_drop, dim=0)
        all_embeddings_node_drop = torch.mean(all_embeddings_node_drop, dim=0, keepdim=False)
        user_embeddings_node_drop, item_embeddings_node_drop = torch.split(all_embeddings_node_drop,
                                                                           [self.user_num, self.item_num])
        node_reg = node_reg / self.LCN_layer

        """ 计算BPR损失 """
        user_emb = user_embeddings_edge_drop[user_id]
        posItem_emb = item_embeddings_edge_drop[pos_item]
        negItem_emb = item_embeddings_edge_drop[neg_item]
        pos_scores = torch.sum(user_emb * posItem_emb, dim=1)
        neg_scores = torch.sum(user_emb * negItem_emb, dim=1)
        bpr_loss_edge_drop = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        user_emb = user_embeddings_node_drop[user_id]
        posItem_emb = item_embeddings_node_drop[pos_item]
        negItem_emb = item_embeddings_node_drop[neg_item]
        pos_scores = torch.sum(user_emb * posItem_emb, dim=1)
        neg_scores = torch.sum(user_emb * negItem_emb, dim=1)
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

        unique_user = torch.unique(user_id)
        unique_posItem = torch.unique(pos_item)
        score_user_edge_drop = ssl_compute(user_embeddings_edge_drop[unique_user],
                                             cur_user_emb[unique_user])
        score_item_edge_drop = ssl_compute(item_embeddings_edge_drop[unique_posItem],
                                             cur_item_emb[unique_posItem])
        score_user_node_drop = ssl_compute(user_embeddings_node_drop[unique_user],
                                             cur_user_emb[unique_user])
        score_item_node_drop = ssl_compute(item_embeddings_node_drop[unique_posItem],
                                             cur_item_emb[unique_posItem])
        # score_unique_user_node = ssl_compute(unique_user_mask_emb, cur_emb[torch.unique(users)])
        # score_unique_item_node = ssl_compute(unique_item_mask_emb, cur_emb[unique_posItem + self.user_num])
        # score_unique_user_node = ssl_compute(unique_user_mask_emb, final_unique_user_emb)
        # score_unique_item_node = ssl_compute(unique_item_mask_emb, final_unique_posItem_emb)

        return bpr_loss_edge_drop, bpr_loss_node_drop, \
            score_user_edge_drop, score_item_edge_drop, \
            score_user_node_drop, score_item_node_drop, edge_reg, node_reg
