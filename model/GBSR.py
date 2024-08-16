import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


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
