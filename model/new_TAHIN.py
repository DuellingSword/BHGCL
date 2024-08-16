# best epoch: 104, bset_recall: 0.10846, best_ndcg: 0.091618
import dgl
import torch
import numpy as np
import torch_sparse
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv, HGTConv, GraphConv
from torch_cluster import random_walk
from model.Contrast_IB import Contrast_IB
from model.Contrast_User import Contrast_User
from model.Contrast_Item import Contrast_Item


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
            # new_g = dgl.to_homogeneous(new_g)
            # coo = new_g.adj(scipy_fmt='coo', etype='_E')
            # csr_matrix = coo.tocsr()
            # semantic_embeddings.append(self.gat_layers[i](h, csr_matrix).flatten(1))
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


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
        self.n_layers = 2
        self.n_intents = 128
        self.temp = 1

        self.batch_size = args.batch_size
        self.emb_reg = 2.5e-5
        self.cen_reg = 5e-3
        self.ssl_reg = 1e-1

        """
        *********************************************************
        Create Model Parameters
        """
        _user_intent = torch.empty(self.emb_dim, self.n_intents)
        nn.init.xavier_normal_(_user_intent)
        self.user_intent = torch.nn.Parameter(_user_intent, requires_grad=True)
        _item_intent = torch.empty(self.emb_dim, self.n_intents)
        nn.init.xavier_normal_(_item_intent)
        self.item_intent = torch.nn.Parameter(_item_intent, requires_grad=True)

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

            # Intent-aware Information Aggregation
            u_embeddings, i_embeddings = torch.split(all_embeddings[i], [self.n_users, self.n_items], 0)
            u_int_embeddings = torch.softmax(u_embeddings @ self.user_intent, dim=1) @ self.user_intent.T
            i_int_embeddings = torch.softmax(i_embeddings @ self.item_intent, dim=1) @ self.item_intent.T
            int_layer_embeddings = torch.concat([u_int_embeddings, i_int_embeddings], dim=0)

            gnn_embeddings.append(gnn_layer_embeddings)
            int_embeddings.append(int_layer_embeddings)
            all_embeddings.append(gnn_layer_embeddings + int_layer_embeddings + all_embeddings[i])

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
        self.unum = self.g.num_nodes(user_key)
        self.inum = self.g.num_nodes(item_key)
        self.device = args.device
        self.han_layers = 1

        self.initializer = nn.init.xavier_uniform_
        self.feature_dict = nn.ParameterDict({
            ntype: nn.Parameter(self.initializer(torch.empty(g.num_nodes(ntype), args.in_size))) for ntype in g.ntypes
        })
        self.DCCF = DCCF(g, args)
        # metapath-based aggregation modules for user and item, this produces h2
        self.meta_path_patterns = args.meta_path_patterns
        # one HANLayer for user, one HANLayer for item
        self.hans = nn.ModuleDict({
            key: HANLayer(value, args.in_size, args.out_size, args.num_heads, args.dropout) for key, value in
            self.meta_path_patterns.items()
        })
        self.cl_rate = args.cl_rate
        # 信息瓶颈-对比学习器
        self.IB_rate = args.IB_rate
        self.mask_reg = args.mask_reg
        self.Contrast_IB = Contrast_IB(g, args)

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

    def forward(self, user_idx, item_idx, neg_item_idx):
        ua_embedding, ia_embedding, gnn_embeddings, int_embeddings = self.DCCF(self.feature_dict)
        # metapath-based aggregation, h2
        h2 = {}
        for key in self.meta_path_patterns.keys():
            for i in range(self.han_layers):
                if i == 0:
                    h2[key] = self.hans[key](self.g, self.feature_dict[key])
                else:
                    h2[key] = self.hans[key](self.g, h2[key])
        user_emb = 0.5 * ua_embedding + 0.5 * h2[self.user_key]
        item_emb = 0.5 * ia_embedding + 0.5 * h2[self.item_key]

        # 计算对比损失
        ssl_loss_user = self.ssl_loss(h2[self.user_key], ua_embedding, user_idx)
        ssl_loss_item = self.ssl_loss(h2[self.item_key], ia_embedding, item_idx)
        ssl_loss = ssl_loss_user + ssl_loss_item

        # 计算 IB 损失
        bpr_loss_edge_drop, bpr_loss_node_drop, \
            score_unique_user_edge, score_unique_item_edge, \
            score_unique_user_node, score_unique_item_node, edge_reg, node_reg \
            = self.Contrast_IB(user_emb, item_emb,
                               self.feature_dict[self.user_key],
                               self.feature_dict[self.item_key],
                               user_idx, item_idx, neg_item_idx)

        return user_emb[user_idx], item_emb[item_idx], item_emb[neg_item_idx], \
            bpr_loss_edge_drop + bpr_loss_node_drop, \
            score_unique_user_edge + score_unique_item_edge + score_unique_user_node + score_unique_item_node,\
            edge_reg + node_reg, ssl_loss

        # return user_emb[user_idx], item_emb[item_idx], item_emb[neg_item_idx], \
        #     0, \
        #     0, \
        #     0, ssl_loss

    def bpr_loss(self, users, pos, neg):
        users_emb, pos_emb, neg_emb, IB_bprLoss, IB_loss, IB_reg, cl_loss = self.forward(users, pos, neg)
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users)) + IB_reg
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        loss += IB_bprLoss
        loss += self.IB_rate * IB_loss
        loss += self.cl_rate * cl_loss

        return loss, reg_loss

    def predict(self, user_idx, item_idx):
        ua_embedding, ia_embedding, gnn_embeddings, int_embeddings = self.DCCF(
            self.feature_dict)
        # metapath-based aggregation, h2
        h2 = {}
        for key in self.meta_path_patterns.keys():
            for i in range(self.han_layers):
                if i == 0:
                    h2[key] = self.hans[key](self.g, self.feature_dict[key])
                else:
                    h2[key] = self.hans[key](self.g, h2[key])
        user_emb = 0.5 * ua_embedding + 0.5 * h2[self.user_key]
        item_emb = 0.5 * ia_embedding + 0.5 * h2[self.item_key]
        user_embeddings_edge_drop, item_embeddings_edge_drop, user_embeddings_node_drop, item_embeddings_node_drop \
            = self.Contrast_IB.predict(user_emb, item_emb, self.feature_dict[self.user_key],
                                       self.feature_dict[self.item_key], user_idx, item_idx)
        user_emb = torch.cat((user_emb[user_idx], user_embeddings_edge_drop, user_embeddings_node_drop), dim=-1)
        item_emb = torch.cat((item_emb[item_idx], item_embeddings_edge_drop, item_embeddings_node_drop), dim=-1)
        return user_emb, item_emb

    def getUsersRating(self, user_idx):
        # x = [2742, 2741, 2743, 2700, 1192, 1976, 2736, 2740, 1201, 2744, 2745, 2739, 2731, 2738, 2161, 1262, 2737, 2722, 2734, 2735, 2733, 2721, 2706, 2729, 849, 2712, 1850, 2732, 2707, 2704, 1853, 2117, 2724, 2551, 2730, 1564, 2726, 2728, 2695, 2727, 2664, 2599, 2718, 2442, 2313, 2716, 2717, 2725, 2554, 2711, 2008, 992, 2698, 2182, 2345, 2713, 2714, 2723, 2719, 2720, 2479, 1430, 1150, 2414, 2642, 2088, 2709, 2648, 1500, 2692, 1436, 2708, 1506, 2701, 2686, 2673, 2705, 2703, 2687, 2267, 2715, 2048, 2689, 2694, 1677, 2683, 2377, 2697, 2702, 2578, 2336, 2489, 2639, 2587, 2688, 2545, 1649, 2653, 2699, 2512, 1018, 2640, 2690, 2107, 2710, 2691, 2669, 2693, 2620, 1719, 2696, 2408, 955, 2649, 1562, 2556, 2362, 1985, 2591, 2680, 1683, 2630, 2681, 2657, 2580, 2679, 2656, 2662, 2607, 2666, 2563, 2651, 2081, 2349, 2685, 2016, 2530, 2558, 2590, 2561, 2627, 2598, 2568, 2663, 2670, 2629, 2575, 2634, 2233, 2659, 2674, 2661, 2641, 2007, 2684, 2402, 1639, 2463, 2682, 2611, 1909, 2660, 1561, 2515, 2595, 2610, 2672, 2645, 2608, 2559, 2652, 2644, 2625, 2617, 2021, 2542, 2655, 2577, 2084, 2643, 1970, 2204, 1676, 2677, 2675, 1442, 2667, 2537, 2628, 2564, 2678, 2613, 1739, 2668, 2658, 2571, 2472, 2676, 2291, 2671, 2665, 2363, 2650, 2635, 1765, 2633, 1779, 2654, 2626, 2615, 2536, 2612, 2525, 2583, 2404, 2400, 2355, 2462, 1937, 2597, 2394, 2570, 2114, 1913, 2356, 2535, 2227, 2621, 2500, 2623, 1299, 2549, 2526, 2646, 1941, 2543, 2596, 2637, 2619, 2638, 2647, 2636, 1174, 2322, 1959, 2170, 1998, 1761, 2293, 2309, 2506, 2268, 2533, 2609, 2490, 2453, 2518, 2366, 2631, 2465, 2602, 2552, 2112, 2352, 2508, 1778, 2614, 1942, 2565, 2624, 1139, 2517, 2606, 2632, 2532, 2138, 2585, 2576, 2618, 2569, 623, 2592, 1245, 2546, 2418, 2622, 2398, 2566, 2616, 2399, 2303, 2338, 1843, 2498, 2514, 2604, 2544, 2448, 2494, 2521, 2478, 930, 2510, 1431, 2409, 2340, 1701, 2555, 2531, 2053, 2522, 2593, 1999, 2105, 2579, 2605, 2449, 2254, 2573, 1588, 2594, 2428, 2452, 2547, 2288, 2553, 2589, 2603, 789, 2523, 1284, 2513, 2135, 2401, 2422, 2582, 2475, 2455, 2492, 843, 1980, 2601, 2541, 2502, 2534, 2213, 2371, 2421, 1965, 2025, 2341, 2295, 1468, 1709, 2560, 2183, 1784, 2314, 2483, 2332, 2420, 2069, 2600, 2180, 2493, 1911, 2584, 2586, 2567, 2588, 2562, 2486, 2347, 2003, 2574, 1990, 2389, 2528, 1275, 1364, 2343, 986, 2524, 2124, 2225, 1242, 2440, 2519, 2415, 2230, 2459, 2250, 2456, 2507, 2433, 2488, 2447, 1809, 2368, 2328, 2405, 2464, 2264, 2503, 2477, 1944, 2384, 2481, 1225, 2520, 2429, 2550, 2473, 1629, 2581, 2424, 2470, 1573, 2548, 1859, 2540, 2538, 2557, 2572, 2294, 1489, 2485, 2484, 2058, 2393, 2443, 2375, 1993, 1953, 2504, 2509, 2511, 2416, 2219, 2299, 2330, 2191, 2457, 2487, 2496, 2495, 2132, 2317, 2407, 1664, 2411, 2430, 2469, 2235, 2306, 1805, 1714, 1700, 2278, 2396, 2461, 2539, 2505, 2256, 2527, 2412, 2207, 1600, 760, 2480, 2381, 2427, 2529, 1735, 2392, 2410, 2186, 1710, 1545, 2413, 944, 2397, 1791, 1974, 2296, 2499, 1625, 2323, 2361, 2467, 2279, 2497, 1637, 2282, 2284, 1617, 2026, 2491, 1493, 1247, 1813, 2471, 2458, 2441, 1394, 1910, 2450, 2468, 2175, 2210, 2325, 2376, 2426, 2027, 2445, 1380, 2482, 2344, 2516, 1323, 2247, 1295, 2476, 432, 2331, 1713, 1628, 2231, 2029, 1962, 2333, 2111, 2365, 2451, 2252, 2241, 2312, 2310, 2370, 2301, 2438, 2185, 2228, 1772, 2382, 2251, 2197, 2168, 2041]
        # item_idx = torch.Tensor(x).long().to(self.device)
        item_idx = torch.Tensor(np.arange(self.inum)).long().to(self.device)
        # users_emb, all_items, _ = self.forward(user_idx, item_idx, None)
        users_emb, all_items = self.predict(user_idx, item_idx)
        rating = torch.matmul(users_emb, all_items.t())
        return rating