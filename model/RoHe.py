import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from model.rohe_gat import GATConv
import numpy as np
import scipy.sparse as sp


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
        z = z.to('cuda:0')
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        return (beta * z).sum(1)  # (N, D * K)


# Metapath-based aggregation (the same as the HANLayer)
class HANLayer(nn.Module):
    def __init__(self, meta_path_patterns, in_size, out_size, layer_num_heads, dropout, settings):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_path_patterns)):
            self.gat_layers.append(
                GATConv(
                    in_size,
                    out_size,
                    layer_num_heads,
                    0.5,
                    0.5,
                    activation=F.elu,
                    settings=settings[i],
                )
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
            new_g = self._cached_coalesced_graph[meta_path_pattern].to(device)
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)

class TAHIN(nn.Module):
    def __init__(self, dataset, g, meta_path_patterns, userkey, itemkey, in_size, out_size, num_heads, dropout, device):
        super(TAHIN, self).__init__()
        # embeddings for different types of nodes, h0
        self.initializer = nn.init.xavier_uniform_
        self.dataset = dataset
        self.g = g
        self.userkey = userkey
        self.itemkey = itemkey
        self.unum = self.g.num_nodes(userkey)
        self.inum = self.g.num_nodes(itemkey)
        self.device = device
        self.feature_dict = nn.ParameterDict({
            ntype: nn.Parameter(self.initializer(torch.empty(g.num_nodes(ntype), in_size))) for ntype in g.ntypes
        })

        # metapath-based aggregation modules for user and item, this produces h2
        self.meta_path_patterns = meta_path_patterns
        # one HANLayer for user, one HANLayer for item
        self.hans = nn.ModuleDict()
        for key, meta_path in self.meta_path_patterns.items():
            given_hete_adjs = get_hete_adjs(g, meta_path)
            trans_adj_list = get_transition(given_hete_adjs, meta_path)
            settings = get_setting(self.dataset, key,
                                   trans_adj_list)  # movielens   amazon这里如果想跑别的数据集应该手动更改（注意与mian函数中的数据集保持一致）"amazon"   "movielens"
            self.hans.update({key: HANLayer(meta_path, in_size, out_size, num_heads, dropout, settings)})

        # layers to combine h0, h1, and h2
        # used to update node embeddings
        self.user_layer1 = nn.Linear(out_size, out_size, bias=True)
        self.item_layer1 = nn.Linear(out_size, out_size, bias=True)

        # layernorm
        self.layernorm = nn.LayerNorm(out_size)

        # network to score the node pairs
        self.pred = nn.Linear(out_size, out_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_size, 1)

    def forward(self, user_idx, item_idx, neg_item_idx):
        user_key = self.userkey
        item_key = self.itemkey

        # metapath-based aggregation, h2
        h2 = {}
        for key in self.meta_path_patterns.keys():
            h2[key] = self.hans[key](self.g, self.feature_dict[key])

        # update node embeddings
        user_emb = h2[user_key]
        item_emb = h2[item_key]
        user_emb = self.user_layer1(user_emb)
        item_emb = self.item_layer1(item_emb)
        # Relu
        user_emb = F.relu_(user_emb)
        item_emb = F.relu_(item_emb)

        # layer norm
        user_emb = self.layernorm(user_emb)
        item_emb = self.layernorm(item_emb)

        # obtain users/items embeddings and their interactions
        user_feat = user_emb[user_idx]
        item_feat = item_emb[item_idx]
        neg_item_feat = item_emb[neg_item_idx]

        return user_feat, item_feat, neg_item_feat

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


device = 'cuda:0'
dbook_topk = 32


def get_setting(datasetname, key, Transmatrix):
    if (datasetname == "Movielens"):

        if (key == "user"):
            settings_umu = {'T': 16, 'device': device, 'TransM': Transmatrix[0]}
            return [settings_umu]
        elif (key == "movie"):
            settings_mum = {'T': 16, 'device': device, 'TransM': Transmatrix[0]}
            settings_mgm = {'T': 16, 'device': device, 'TransM': Transmatrix[1]}
            return [settings_mum, settings_mgm]
    elif (datasetname == "Amazon"):
        if (key == "user"):
            settings_uiu = {'T': 32, 'device': device, 'TransM': Transmatrix[0]}
            return [settings_uiu]
        elif (key == "item"):
            settings_iui = {'T': 32, 'device': device, 'TransM': Transmatrix[0]}
            settings_ici = {'T': 32, 'device': device, 'TransM': Transmatrix[1]}
            settings_ibi = {'T': 32, 'device': device, 'TransM': Transmatrix[2]}
            settings_ivi = {'T': 32, 'device': device, 'TransM': Transmatrix[3]}
            return [settings_iui, settings_ici, settings_ibi, settings_ivi]
    elif (datasetname == "DoubanBook"):
        if (key == "user"):
            settings_ubu = {'T': dbook_topk, 'device': device, 'TransM': Transmatrix[0]}
            settings_ugu = {'T': dbook_topk, 'device': device, 'TransM': Transmatrix[1]}
            return [settings_ubu, settings_ugu]
        elif (key == "book"):
            settings_bub = {'T': dbook_topk, 'device': device, 'TransM': Transmatrix[0]}
            settings_bab = {'T': dbook_topk, 'device': device, 'TransM': Transmatrix[1]}
            # settings_byb = {'T': dbook_topk, 'device': device, 'TransM': Transmatrix[2]}
            return [settings_bub, settings_bab]
    #     , settings_byb
    elif (datasetname == "DoubanMovie"):
        if (key == "user"):
            settings_umu = {'T': dbook_topk, 'device': device, 'TransM': Transmatrix[0]}
            settings_ugu = {'T': dbook_topk, 'device': device, 'TransM': Transmatrix[1]}
            settings_uuu = {'T': dbook_topk, 'device': device, 'TransM': Transmatrix[2]}
            return [settings_umu, settings_ugu, settings_uuu]
        elif (key == "movie"):
            settings_mum = {'T': dbook_topk, 'device': device, 'TransM': Transmatrix[0]}
            settings_mtm = {'T': dbook_topk, 'device': device, 'TransM': Transmatrix[1]}
            # settings_byb = {'T': dbook_topk, 'device': device, 'TransM': Transmatrix[2]}
            return [settings_mum, settings_mtm]
    elif (datasetname == "Yelp"):
        if (key == "user"):
            settings_ubu = {'T': dbook_topk, 'device': device, 'TransM': Transmatrix[0]}
            return [settings_ubu]
        elif (key == "business"):
            settings_bub = {'T': dbook_topk, 'device': device, 'TransM': Transmatrix[0]}
            settings_bcb = {'T': dbook_topk, 'device': device, 'TransM': Transmatrix[1]}
            settings_bccb = {'T': dbook_topk, 'device': device, 'TransM': Transmatrix[2]}
            return [settings_bub, settings_bcb, settings_bccb]


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def get_hete_adjs(g, meta_paths):  # 输入：异构图g，关键字的元路径。输出：不同元路径的概率传递矩阵
    hete_adjs = {}
    for i in range(len(meta_paths)):
        for value in meta_paths[i]:
            # adj_matrixb = g.adjacency_matrix(etype=value, scipy_fmt='coo')
            # g.adj_external(ctx='cpu', scipy_fmt='csr', etype=args.ui_relation)
            adj_matrixb = g.adj_external(ctx='cuda:0', etype=value, scipy_fmt='coo')
            adj_matrixb = adj_matrixb.tocsc()
            hete_adjs.update({value: adj_matrixb})
    return hete_adjs


def get_transition(given_hete_adjs, metapath_info):
    # transition
    hete_adj_dict_tmp = {}
    for key in given_hete_adjs.keys():
        deg = given_hete_adjs[key].sum(1)
        hete_adj_dict_tmp[key] = given_hete_adjs[key] / (np.where(deg > 0, deg, 1))  # make sure deg>0
    homo_adj_list = []
    for i in range(len(metapath_info)):  # 遍历pa,ap,pf,fp这几个CSC_matrix当然这是规范后的，上面那个循环就是将原本的矩阵进行规范化
        adj = hete_adj_dict_tmp[metapath_info[i][0]]  #
        for etype in metapath_info[i][1:]:
            adj = adj.dot(hete_adj_dict_tmp[etype])
        homo_adj_list.append(sp.csc_matrix(adj))
    return homo_adj_list  # 不同将异构矩阵（hete_adjs）不同key的不同元路径传进去它会得到不同元路径传输概率矩阵