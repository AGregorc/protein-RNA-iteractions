import typing

import torch
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from torch import nn
from torch.nn import Linear


class NetGATConv(nn.Module):
    def __init__(self, in_features=64, out_features=2,
                 hidden_gat_sizes=None,
                 dropout_p=0.2):
        super(NetGATConv, self).__init__()

        if hidden_gat_sizes is None:
            hidden_gat_sizes = [10]

        self.dropout = nn.Dropout(p=dropout_p)

        prev_hidden = in_features
        self.hidden_gat_layers = []
        self.hidden_b_norms = []
        for out_hidden in hidden_gat_sizes:
            num_heads = 2
            self.hidden_gat_layers.append(GATConv(prev_hidden, out_hidden, num_heads=num_heads))
            self.hidden_b_norms.append(nn.BatchNorm1d(out_hidden))
            prev_hidden = out_hidden
        self.hidden_gat_layers = nn.ModuleList(self.hidden_gat_layers)
        self.hidden_b_norms = nn.ModuleList(self.hidden_b_norms)

        self.last_linear_layer = Linear(prev_hidden, out_features)

    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)

    def forward(self, g_and_features):
        g, h = g_and_features

        for GATConvLayer, b_norm in zip(self.hidden_gat_layers, self.hidden_b_norms):
            h = GATConvLayer(g, h)
            h = torch.mean(h, dim=1)
            h = b_norm(h)
            h = F.relu(h)
            h = self.dropout(h)

        if len(h.shape) == 3:
            h = torch.mean(h, dim=1)
        h = self.last_linear_layer(h)

        return g, h
