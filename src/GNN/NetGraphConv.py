import typing

import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from torch import nn
from torch.nn import Linear


class NetGraphConv(nn.Module):
    def __init__(self, in_features=64, out_features=2,
                 hidden_conv_sizes=None,
                 dropout_p=0.2):
        super(NetGraphConv, self).__init__()

        if hidden_conv_sizes is None:
            hidden_conv_sizes = [10]

        self.dropout = nn.Dropout(p=dropout_p)

        prev_hidden = in_features
        self.hidden_conv_layers = []
        self.hidden_b_norms = []
        for out_hidden in hidden_conv_sizes:
            self.hidden_conv_layers.append(GraphConv(prev_hidden, out_hidden))
            self.hidden_b_norms.append(nn.BatchNorm1d(out_hidden))
            prev_hidden = out_hidden
        self.hidden_conv_layers = nn.ModuleList(self.hidden_conv_layers)
        self.hidden_b_norms = nn.ModuleList(self.hidden_b_norms)

        self.last_linear_layer = Linear(prev_hidden, out_features)

    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)

    def forward(self, g_and_features):
        g, h = g_and_features

        for graphConvLayer, b_norm in zip(self.hidden_conv_layers, self.hidden_b_norms):
            h = self.dropout(F.relu(b_norm(graphConvLayer(g, h))))
        h = self.last_linear_layer(h)

        return g, h
