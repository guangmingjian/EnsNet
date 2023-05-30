#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/24 9:07
# @Author  : mingjian
# @Version : 1.0
# @File    : SAGPool.py

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, GraphSizeNorm, BatchNorm,SAGEConv
from torch_geometric.nn.pool.topk_pool import topk, filter_adj



# import torchsnooper
class SAGPool(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, hidden, device, ratio=0.8, dropout=0.0, arc_type="H",graph_norm=True):
        """
        :param in_channels:
        :param out_channels:
        :param num_layers:
        :param hidden:
        :param ratio:
        :param dropout:
        :param arc_type: H 层级，G全局
        """
        super(SAGPool, self).__init__()
        self.graph_norm = graph_norm
        self.arc_type = arc_type
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.dropout = dropout
        self.convs.extend([
            SAGEConv(hidden if i > 0 else in_channels, hidden, aggr='mean')
            for i in range(num_layers)
        ])
        if self.arc_type == "G":
            self.pools.extend([SAGPooling(hidden * num_layers, ratio)])
            self.lin1 = Linear(2 * hidden * num_layers, hidden)
        else:
            self.pools.extend([SAGPooling(hidden, ratio) for i in range(num_layers)])
            self.lin1 = Linear(2 * hidden, hidden)
        if self.graph_norm:
            self.gsnorm = GraphSizeNorm()
            self.bns = torch.nn.ModuleList([BatchNorm(hidden) for _ in range(num_layers)])
        self.lin2 = Linear(hidden, hidden // 2)
        self.lin3 = Linear(hidden // 2, out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        if self.graph_norm:
            for bn in self.bns:
                bn.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    # @torchsnooper.snoop()
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_list = []
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            if self.graph_norm:
                x = self.gsnorm(x, batch)
                x = self.bns[i](x)
            if self.arc_type == "G":
                x_list.append(x)
            elif self.arc_type == "H":
                x, edge_index, _, batch, _ = self.pools[i](x, edge_index, None, batch=batch)
                if i == 0:
                    xs = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
                else:
                    xs += torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)

        if self.arc_type == "G":
            xs = torch.cat(x_list, dim=1)
            xs, edge_index, _, batch, _ = self.pools[0](xs, edge_index, None, batch=batch)
            xs = torch.cat([global_mean_pool(xs, batch), global_max_pool(xs, batch)], dim=1)
        x = F.relu(self.lin1(xs))
        self.xs = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin2(x))
        # x = F.log_softmax(self.lin3(x), dim=-1)
        return self.lin3(x)


class SAGPooling(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.8, Conv=GCNConv, non_linearity=torch.tanh):
        super(SAGPooling, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels, 1)
        self.non_linearity = non_linearity
        self.reset_parameters()

    def reset_parameters(self):
        self.score_layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        score = self.score_layer(x, edge_index).squeeze()

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm
