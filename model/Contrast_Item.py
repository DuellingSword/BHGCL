import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


####################################################
# This tool is to generate positive set with a thre-
# shold "pos_num".
# dataset  pos_num
# acm      5
# dblp     1000
# aminer   15
# freebase 80
#
#
# Notice: The best pos_num of acm is 7 reported in
# paper, but we find there is no much difference
# between 5 and 7 in practice.
####################################################
class Contrast_Item(nn.Module):
    def __init__(self, g, args):
        super(Contrast_Item, self).__init__()
        in_size = args.in_size
        hidden_dim = args.cl_hidden_dim
        self.device = args.device
        self.temperature = args.temperature
        self.lam = args.lam
        self.pos_num = args.pos_num
        item_key = args.item_key
        dataset = args.dataset
        self.item_meta_path_patterns = args.meta_path_patterns[item_key]
        self.proj = nn.Sequential(
            nn.Linear(in_size, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.pos = self.find_pos(g, item_key).to(self.device)
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def find_pos(self, g, item_key):
        all = torch.zeros((g.num_nodes(item_key), g.num_nodes(item_key)))
        for meta_path_pattern in self.item_meta_path_patterns:
            # 获取 CSR压缩稀疏列 (Compressed Sparse Column) 格式的邻接矩阵
            adj = dgl.metapath_reachable_graph(g, meta_path_pattern).adj_external(ctx='cpu', scipy_fmt='csr')
            temp = torch.zeros((g.num_nodes(item_key), g.num_nodes(item_key)))
            for i in range(adj.shape[0]):  # 每一行代表 与目标类型id=i相连的srcType的节点ID
                # 从 CSC 矩阵中获取第 i 列的所有非零行索引，即所有与列索引 i 有连接的节点
                start = adj.indptr[i]
                end = adj.indptr[i + 1]
                # 例如：如果 i = 296，数组可能是 [936, ...]
                # 表示 dsttype_id=296 的节点与 srctype 的节点 936 有交互
                temp[i, adj.indices[start:end]] += 1
            temp = temp / (temp.sum(axis=-1).reshape(-1, 1))
            temp = torch.where(torch.logical_or(torch.isnan(temp), torch.isinf(temp)), torch.tensor(0.0), temp)
            all += temp

        pos = torch.zeros((g.num_nodes(item_key), g.num_nodes(item_key)))
        pos_num = self.pos_num
        for i in range(len(all)):
            one = all[i].nonzero()
            if len(one) > pos_num:
                # 使用 torch.sort() 对张量进行排序，指定 dim=0 表示按列排序
                _, indices = torch.sort(all[i][one], dim=0, descending=True)
                indices = indices[:, 0]
                sele = one[indices[:pos_num]]
                pos[i, sele] = 1
            else:
                pos[i, one] = 1
        return pos

    def sim(self, z1, z2):
        z1_normalized = F.normalize(z1, p=2, dim=1)
        z2_normalized = F.normalize(z2, p=2, dim=1)
        dot_numerator = torch.matmul(z1_normalized, z2_normalized.t())
        sim_matrix = torch.exp(dot_numerator / self.temperature)
        return sim_matrix

    def forward(self, z_mp, z_sc):
        z_proj_sc = self.proj(z_sc)
        z_proj_mp = self.proj(z_mp)

        matrix_mp2sc = self.sim(z_proj_mp, z_proj_sc)
        matrix_sc2mp = matrix_mp2sc.t()

        matrix_mp2sc = matrix_mp2sc / (torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8)
        matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
        pos = self.pos
        lori_sc = -torch.log(matrix_sc2mp.mul(pos).sum(dim=-1) + 1e-8)  # mul是元素级别的乘法运算，也就是对应位置的元素相乘
        lori_mp = -torch.log(matrix_mp2sc.mul(pos).sum(dim=-1) + 1e-8)
        # print("test:", -torch.log(matrix_sc2mp.mul(self.pos).sum(dim=-1)), " ", -torch.log(matrix_mp2sc.mul(self.pos).sum(dim=-1)))
        ans = self.lam * lori_mp + (1 - self.lam) * lori_sc
        ans = torch.where(torch.logical_or(torch.isnan(ans), torch.isinf(ans)), torch.tensor(0.0).to(self.device), ans)
        return ans
