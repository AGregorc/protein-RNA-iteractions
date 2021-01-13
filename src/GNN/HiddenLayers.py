import typing

import torch.nn.functional as F
from dgl.nn import GraphConv, GATConv
from torch import nn
from torch.nn import Linear


class HiddenLayers(nn.Module):
    def __init__(self, LayersTypeClass, graph_needed, in_features=64, out_features=2,
                 hidden_sizes=None, dropout_p=0.2):
        super(HiddenLayers, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [10]

        self.graph_needed = graph_needed
        self.dropout = nn.Dropout(p=dropout_p)

        prev_hidden = in_features
        self.hidden_layers = []
        self.hidden_b_norms = []
        for out_hidden in hidden_sizes:
            self.hidden_layers.append(LayersTypeClass(prev_hidden, out_hidden))
            self.hidden_b_norms.append(nn.BatchNorm1d(out_hidden))
            prev_hidden = out_hidden
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.hidden_b_norms = nn.ModuleList(self.hidden_b_norms)

        self.last_linear_layer = Linear(prev_hidden, out_features)

    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)

    def forward(self, g_and_features):
        g, h = g_and_features

        for hiddenLayer, b_norm in zip(self.hidden_layers, self.hidden_b_norms):
            if self.graph_needed:
                h = hiddenLayer(g, h)
            else:
                h = hiddenLayer(h)
            self.dropout(F.relu(b_norm(h)))

        h = self.last_linear_layer(h)
        return g, h


class NetLinear(HiddenLayers):
    def __init__(self, in_features=64, out_features=2,
                 dropout_p=0.2, hidden_linear_sizes=None):
        super().__init__(Linear, False, in_features, out_features, hidden_linear_sizes, dropout_p)


class NetGraphConv(HiddenLayers):
    def __init__(self, in_features=64, out_features=2,
                 hidden_conv_sizes=None, dropout_p=0.2):
        super().__init__(GraphConv, True, in_features, out_features, hidden_conv_sizes, dropout_p)


class NetGATConv(HiddenLayers):
    def __init__(self, in_features=64, out_features=2,
                 hidden_gat_sizes=None, dropout_p=0.2):
        super().__init__(GATConv, True, in_features, out_features, hidden_gat_sizes, dropout_p)

