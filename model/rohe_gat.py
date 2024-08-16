import torch
from torch import nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.utils import expand_as_pair
from dgl.nn.pytorch.softmax import edge_softmax
from dgl.nn.pytorch.utils import Identity


class GATConv(nn.Module):
    def __init__(self,
                 in_feats,  # 输入维度;1903
                 out_feats,  # 输出维度:8
                 num_heads,  # 聚合头数:8
                 feat_drop=0.,  # drop:0.6
                 attn_drop=0.,  # 0.6
                 negative_slope=0.2,  # 0.2这个是干嘛的？
                 residual=False,  # 偏置
                 activation=None,
                 settings={'K': 10, 'P': 0.6, 'tau': 0.1, 'Flag': "None"}):  # T：2，device：0，transM(稀疏矩阵):4025*4025

        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self.settings = settings
        self._in_src_feats, self._in_dst_feats = expand_as_pair(
            in_feats)  # self._in_src_feats：1903, self._in_dst_feats：1903
        self._out_feats = out_feats  # ：8
        # a = isinstance(in_feats, tuple)#fasle
        # print(a)
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)  # 是投影网络层吗？
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(0.0)
        self.leaky_relu = nn.LeakyReLU(negative_slope)  # 带有负斜率的激活函数：negative_slope负斜率为：0.2
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)  # gain权重缩放因子
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def mask(self, attM):
        T = self.settings['T']
        indices_to_remove = attM < torch.clamp(torch.topk(attM, T)[0][..., -1, None], min=0)
        attM[indices_to_remove] = -9e15
        return attM

    def forward(self, graph, feat):
        # GATConv(feat:4025*1093
        #   (fc): Linear(in_features=1903, out_features=64, bias=False)
        #   (feat_drop): Dropout(p=0.6, inplace=False)
        #   (attn_drop): Dropout(p=0.0, inplace=False)
        #   (leaky_relu): LeakyReLU(negative_slope=0.2)
        # )
        graph = graph.local_var()  # 防内存泄露

        if isinstance(feat, tuple):
            h_src = self.feat_drop(feat[0])
            h_dst = self.feat_drop(feat[1])
            feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
            feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
        else:
            h_src = h_dst = self.feat_drop(feat)  # 944*128
            # h_src = h_dst = feat
            a = self.fc(h_src)
            feat_src = feat_dst = self.fc(h_src).view(
                -1, self._num_heads, self._out_feats)  # 4025*8*8，划分为八个头将单个节点：1903训练为8*8
        N = graph.nodes().shape[0]  # 图中节点数量，这里的图是基于元路径的子图944
        N_e = graph.edges()[0].shape[0]  # 边的数量605076

        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # feat_src = feat_src.to(device)
        graph.srcdata.update({'ft': feat_src})  # 将图中的特征更新为训练的特征

        # introduce transiting prior
        e_trans = torch.FloatTensor(self.settings['TransM'].data).view(N_e, 1)  # 边（节点，节点，传播概率）57853*1
        # a = e_trans.repeat(1,8)
        e_trans = e_trans.repeat(1, self._num_heads).resize_(N_e, self._num_heads, 1)  # 57853*8*1
        # e_trans = e_trans.repeat(1,8).resize_(N_e,8,1)
        # a = feat_src[:,0,:].view(N,self._out_feats)
        # b =  feat_src[:,0,:].t()
        # a = torch.matmul(feat_src[:,0,:].view(N,self._out_feats),\
        #         feat_src[:,0,:].t().view(self._out_feats,N))#944
        # a = torch.cat([torch.matmul(feat_src[:,i,:].view(N,self._out_feats),\
        #         feat_src[:,i,:].t().view(self._out_feats,N))[graph.edges()[0], graph.edges()[1]].view(N_e,1)\
        #             for i in range(self._num_heads)],dim=1)
        # feature-based similarity 跟据投影的特征相乘得到最原始的权重
        e = torch.cat([torch.matmul(feat_src[:, i, :].view(N, self._out_feats), \
                                    feat_src[:, i, :].t().view(self._out_feats, N))[
                           graph.edges()[0], graph.edges()[1]].view(N_e, 1) \
                       for i in range(self._num_heads)], dim=1).view(N_e, self._num_heads,
                                                                     1)  # self._num_heads#view(N_e,8,1)
        # E:57853*8*1
        total_edge = torch.cat((graph.edges()[0].view(1, N_e), graph.edges()[1].view(1, N_e)), 0)  # 所有边2*57853（边，边）
        # confidence score in Eq(7)6
        # e = e.to('cpu')
        # print(e.device)
        # print(total_edge.device)
        # print(e_trans.device)
        # attn:4025*4025
        # print(torch.cuda.is_available())#:false,是这个问题:原因torch与cuda不兼容
        # print(self.settings['device'])
        # exit()
        attn = torch.sparse.FloatTensor(total_edge, \
                                        torch.squeeze((e * e_trans.to(self.settings['device'])).sum(-2)),
                                        torch.Size([N, N])).to(self.settings['device'])

        # purification mask in Eq(8)7
        attn = self.mask(attn.to_dense()).t()
        e[attn[graph.edges()[0], graph.edges()[1]].view(N_e, 1).repeat(1, self._num_heads).view(N_e, self._num_heads,
                                                                                                1) < -100] = -9e15
        # e[attn[graph.edges()[0],graph.edges()[1]].view(N_e,1).repeat(1,8).view(N_e,8,1)<-100] = -9e15
        # obtain purified final attention in Eq(9)
        graph.edata['a'] = edge_softmax(graph, e)  # 边缘权重更新

        # message passing   fn.u_mul_e('ft', 'a', 'm')：将权重与特征相乘得到新的特征存储在M中    fn.sum('m', 'ft')将所有邻居节点的特征进行聚合
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.dstdata['ft']  # 更新后的节点特征

        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
            rst = rst + resval

        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst