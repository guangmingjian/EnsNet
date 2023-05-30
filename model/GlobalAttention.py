import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, GlobalAttention,GraphSizeNorm, BatchNorm


class GlobalAttentionNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, device, hidden, dropout,graph_norm=True):
        super(GlobalAttentionNet, self).__init__()
        self.graph_norm = graph_norm
        self.convs = torch.nn.ModuleList()
        self.convs.extend([
            SAGEConv(hidden if i > 0 else in_channels, hidden, aggr='mean')
            for i in range(num_layers)
        ])
        self.att = GlobalAttention(Linear(hidden, 1))
        self.dropout = torch.nn.Dropout(dropout)
        if self.graph_norm:
            self.gsnorm = GraphSizeNorm()
            self.bns = torch.nn.ModuleList([BatchNorm(hidden) for _ in range(num_layers)])
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.graph_norm:
            for bn in self.bns:
                bn.reset_parameters()
        self.att.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            if self.graph_norm:
                x = self.gsnorm(x, batch)
                x = self.bns[i](x)
        x = self.att(x, batch)
        x = F.relu(self.lin1(x))
        self.xs = x
        x = self.dropout(x)
        x = self.lin2(x)
        # return F.log_softmax(x, dim=-1)
        return x

