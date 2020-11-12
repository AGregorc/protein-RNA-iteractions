import typing

import torch.nn.functional as F
from torch import nn
from torch.nn import Linear


class NetLinear(nn.Module):
    def __init__(self, in_features=64 + 8, out_features=2,
                 dropout_p=0.4, hidden_linear_sizes=None):
        super(NetLinear, self).__init__()

        if hidden_linear_sizes is None:
            hidden_linear_sizes = [10]

        self.dropout = nn.Dropout(p=dropout_p)

        prev_hidden = in_features

        self.hidden_linear_layers = []
        for size in hidden_linear_sizes:
            self.hidden_linear_layers.append(Linear(prev_hidden, size))
            prev_hidden = size
        self.hidden_linear_layers = nn.ModuleList(self.hidden_linear_layers)
        self.last_linear_layer = Linear(prev_hidden, out_features)

    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)

    def forward(self, g_and_features):
        g, h = g_and_features

        for linearLayer in self.hidden_linear_layers:
            h = F.relu(self.dropout(linearLayer(h)))
        h = self.last_linear_layer(h)

        return g, h
