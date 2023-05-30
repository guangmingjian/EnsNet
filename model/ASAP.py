#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/26 21:53
# @Author  : mingjian
# @Version : 1.0
# @File    : ASAP.py

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import (ASAPooling, GraphConv, global_mean_pool, global_max_pool,GraphSizeNorm, BatchNorm,
                                JumpingKnowledge,SAGEConv)
import torchsnooper

class ASAP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, hidden, device, ratio=0.8, dropout=0, arc_type="H",graph_norm=True):
        super(ASAP, self).__init__()
        self.graph_norm = graph_norm
        self.arc_type = arc_type
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            GraphConv(hidden if i > 0 else in_channels, hidden, aggr='mean')
            for i in range(num_layers)
        ])
        if self.arc_type == "G":
            self.pools.extend([ASAPooling(hidden * num_layers, ratio)])
            self.lin1 = Linear(2 * hidden * num_layers, hidden)
        else:
            self.pools.extend([
                ASAPooling(hidden, ratio, dropout=dropout)
                for i in range(num_layers)
            ])
            self.lin1 = Linear(2 * hidden, hidden)
        if self.graph_norm:
            self.gsnorm = GraphSizeNorm()
            self.bns = torch.nn.ModuleList([BatchNorm(hidden) for _ in range(num_layers)])
        # self.jump = JumpingKnowledge(mode='cat')
        self.lin2 = Linear(hidden, out_channels)
        self.dropout = dropout

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

    # @torchsnooper.snoop()
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_weight = None
        x_list = []
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index, edge_weight=edge_weight))
            if self.graph_norm:
                x = self.gsnorm(x, batch)
                x = self.bns[i](x)
            if self.arc_type == "G":
                x_list.append(x)
            elif self.arc_type == "H":
                x, edge_index, edge_weight, batch, perm = self.pools[i](x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
                if i == 0:
                    xs = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
                else:
                    xs += torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)

        if self.arc_type == "G":
            xs = torch.cat(x_list, dim=1)
            xs, edge_index, edge_weight, batch, perm = self.pools[0](x=xs, edge_index=edge_index, edge_weight=edge_weight,
                                                                    batch=batch)
            xs = torch.cat([global_mean_pool(xs, batch), global_max_pool(xs, batch)], dim=1)
        x = F.relu(self.lin1(xs))
        self.xs = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x
