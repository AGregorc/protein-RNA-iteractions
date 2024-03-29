import typing

from torch import nn
from torch.nn import Linear

from GNN.NetGraphConv import NetGraphConv
from GNN.NetLinear import NetLinear


class NetFirstGraphConvThenLinear(nn.Module):
    def __init__(self, in_features=64, out_features=2,
                 hidden_conv_sizes=None, hidden_linear_sizes=None):
        super(NetFirstGraphConvThenLinear, self).__init__()

        if hidden_linear_sizes is None:
            hidden_linear_sizes = [10]
        if hidden_conv_sizes is None:
            hidden_conv_sizes = [10]

        conv_out_features = hidden_conv_sizes[-1]
        self.hidden_conv_net = NetGraphConv(in_features=in_features,
                                            out_features=conv_out_features,
                                            hidden_conv_sizes=hidden_conv_sizes)

        lin_out_features = hidden_linear_sizes[-1]
        self.hidden_linear_net = NetLinear(in_features=conv_out_features,
                                           out_features=lin_out_features,
                                           hidden_linear_sizes=hidden_linear_sizes)
        self.last_linear_layer = Linear(lin_out_features, out_features)

    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)

    def forward(self, g_and_features):
        g, h = g_and_features
        g, h = self.hidden_conv_net((g, h))
        g, h = self.hidden_linear_net((g, h))
        h = self.last_linear_layer(h)
        return g, h
