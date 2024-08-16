import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


# Relational neighbor aggregation
class RelationalAGG(nn.Module):
    def __init__(self, g, args):
        super(RelationalAGG, self).__init__()
        self.in_size = in_size = args.in_size
        self.dropout = dropout = args.atten_dropout
        self.out_size = args.out_size  # 暂时没用到
        self.device = args.device

        # Transform weights for different types of edges
        # 把in_size的src节点维度 转成 in_size维度
        self.W_T = nn.ModuleDict(
            {
                name: nn.Linear(in_size, in_size, bias=False)
                for name in g.etypes
            }
        )

        # Attention weights for different types of edges
        self.W_A = nn.ModuleDict(
            {name: nn.Linear(in_size, 1, bias=False) for name in g.etypes}
        )

        # layernorm
        self.layernorm = nn.LayerNorm(in_size)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

    # feat_dict为节点特征字典
    def forward(self, g, feat_dict):
        funcs = {}
        for srctype, etype, dsttype in g.canonical_etypes:
            g.nodes[dsttype].data["h"] = feat_dict[dsttype]
            g.nodes[srctype].data["h"] = feat_dict[srctype]
            g.nodes[srctype].data["t_h"] = self.W_T[etype](
                feat_dict[srctype]
            )  # src nodes' transformed feature

            # compute the attention numerator (exp)
            # etype=etype 表示只对指定类型的边执行这个操作
            g.apply_edges(fn.u_mul_v("t_h", "h", "x"), etype=etype)  # 对每条边上的 t_h 和 h 执行逐元素乘法，结果存储在目标张量 x 中
            g.edges[etype].data["x"] = torch.exp(
                # 注意力权重系数向量和g.edges[etype].data["x"]做点乘
                self.W_A[etype](g.edges[etype].data["x"])
            )

            # first update to compute the attention denominator (\sum exp)
            funcs[etype] = (fn.copy_e("x", "m"), fn.sum("m", "att"))  # 按边类型聚合消息
        g.multi_update_all(funcs, "sum")  # 首先按照边类型进行聚合funcs[etype]，然后在不同类型之间进行聚合"sum"，最后更新所有节点的特征

        # u --e--> v
        funcs = {}
        for srctype, etype, dsttype in g.canonical_etypes:
            g.apply_edges(
                fn.e_div_v("x", "att", "att"), etype=etype
            )  # compute attention weights (numerator/denominator)
            funcs[etype] = (
                fn.u_mul_e("h", "att", "m"),
                fn.sum("m", "h"),
            )  # \sum(h0*att) -> h1
        # second update to obtain h1
        g.multi_update_all(funcs, "sum")

        # apply activation, layernorm, and dropout
        feat_dict = {}
        for ntype in g.ntypes:
            feat_dict[ntype] = self.dropout(
                self.layernorm(F.relu_(g.nodes[ntype].data["h"]))
            )  # apply activation, layernorm, and dropout

        return feat_dict
