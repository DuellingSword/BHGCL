import torch
from torch import nn
import scipy.sparse as sp
import numpy as np
from utility.dataloader import Data


class NGCF(nn.Module):
    def __init__(self, config, dataset: Data, device):
        super(NGCF, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        if config.mess_dropout:
            self.mess_dropout = eval(config.mess_keep_prob)
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_users, embedding_dim=self.config.dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_items, embedding_dim=self.config.dim)

        # self.user_embedding.weight = torch.nn.Parameter(torch.load('./data/user_emb.pth'))
        # self.item_embedding.weight = torch.nn.Parameter(torch.load('./data/movie_emb.pth'))

        # no pretrain
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)

        initializer = nn.init.xavier_uniform_
        self.weight_dict = nn.ParameterDict()
        layers = [self.config.dim] + eval(self.config.layer_size)

        for layer in range(self.config.GCNLayer):
            self.weight_dict.update(
                {'W_gcn_%d' % layer: nn.Parameter(initializer(torch.empty(layers[layer], layers[layer + 1])))})
            self.weight_dict.update(
                {'b_gcn_%d' % layer: nn.Parameter(initializer(torch.empty(1, layers[layer + 1])))})
            self.weight_dict.update(
                {'W_bi_%d' % layer: nn.Parameter(initializer(torch.empty(layers[layer], layers[layer + 1])))})
            self.weight_dict.update(
                {'b_bi_%d' % layer: nn.Parameter(initializer(torch.empty(1, layers[layer + 1])))})

        self.Graph = self.dataset.sparse_adjacency_matrix()  # sparse matrix
        self.Graph = self.convert_sp_mat_to_sp_tensor(self.Graph)  # sparse tensor
        self.Graph = self.Graph.coalesce().to(self.device)  # Sort the edge index and remove redundancy

        self.activation = nn.Sigmoid()

    def convert_sp_mat_to_sp_tensor(self, sp_mat):
        """
            coo.row: x in user-item graph
            coo.col: y in user-item graph
            coo.data: [value(x,y)]
        """
        coo = sp_mat.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()

        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        value = torch.FloatTensor(coo.data)
        # from a sparse matrix to a sparse float tensor
        sp_tensor = torch.sparse.FloatTensor(index, value, torch.Size(coo.shape))
        return sp_tensor

    def node_dropout(self, graph, keep_prob):
        size = graph.size()
        index = graph.indices().t()
        values = graph.values()
        random_index = torch.rand(len(values)) + (1 - keep_prob)
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/(1 - keep_prob)
        new_graph = torch.sparse.FloatTensor(index.t(), values, size)
        return new_graph

    def aggregate(self):
        # [user + item, emb_dim]
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight])

        # no dropout
        all_embeddings = [ego_embeddings]

        if self.config.node_dropout:
            if self.training:
                dropout_graph = self.node_dropout(self.Graph, self.config.node_keep_prob)
            else:
                dropout_graph = self.Graph
        else:
            dropout_graph = self.Graph

        # aggregate
        for layer in range(self.config.GCNLayer):
            # # [node, emb_dim]
            side_embeddings = torch.sparse.mm(dropout_graph, ego_embeddings)

            # (L+I)·E·W1
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gcn_%d' % layer]) \
                             + self.weight_dict['b_gcn_%d' % layer]

            # Ei * Eu
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)

            # L·E*E·W2
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % layer]) \
                             + self.weight_dict['b_bi_%d' % layer]

            # LeakyReLU((L+I)·E·W1 + L·E*E·W2)
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)

            # message dropout
            ego_embeddings = nn.Dropout(self.mess_dropout[layer])(ego_embeddings)

            norm_embeddings = nn.functional.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings.append(norm_embeddings)

        final_embeddings = torch.cat(all_embeddings, dim=1)  # [node, layer+1, emb_dim]

        users_emb, items_emb = torch.split(final_embeddings, [self.dataset.num_users, self.dataset.num_items])
        return users_emb, items_emb

    def get_bpr_loss(self, user, positive, negative):
        all_user_embeddings, all_item_embeddings = self.aggregate()

        user_embedding = all_user_embeddings[user.long()]

        positive_embedding = all_item_embeddings[positive.long()]
        negative_embedding = all_item_embeddings[negative.long()]

        ego_user_emb = self.user_embedding(user)
        ego_pos_emb = self.item_embedding(positive)
        ego_neg_emb = self.item_embedding(negative)

        reg_loss = (1 / 2) * (ego_user_emb.norm(2).pow(2) + ego_pos_emb.norm(2).pow(2) +
                              ego_neg_emb.norm(2).pow(2)) / float(len(user))

        pos_score = torch.sum(torch.mul(user_embedding, positive_embedding), dim=1)

        neg_score = torch.sum(torch.mul(user_embedding, negative_embedding), dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_score - pos_score))

        reg_loss = reg_loss * self.config.l2

        return loss, reg_loss

    def getUsersRating(self, user):
        all_user_embeddings, all_item_embeddings = self.aggregate()

        user_embeddings = all_user_embeddings[user.long()]

        rating = self.activation(torch.matmul(user_embeddings, all_item_embeddings.t()))

        return rating