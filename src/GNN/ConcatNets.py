import torch
import typing

from dgl import DGLGraph
from torch import nn

from Constants import NODE_FEATURES_NAME


class ConcatNets(nn.Module):
    def __init__(self, nets):
        super(ConcatNets, self).__init__()
        self.nets = nn.ModuleList(nets)

    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)

    def forward(self, g_and_features):
        if type(g_and_features) is tuple:
            g = h = None
            for e in g_and_features:
                if type(e) is DGLGraph:
                    g = e
                if type(e) is torch.Tensor:
                    h = e
            if h is None:
                h = g.ndata[NODE_FEATURES_NAME]
            g_and_h = (g, h)
        else:
            g = g_and_features
            g_and_h = (g, g.ndata[NODE_FEATURES_NAME])

        h_list = []
        for net in self.nets:
            _, output_h = net(g_and_h)
            h_list.append(output_h)
        h_list = torch.cat(h_list, dim=1)
        return g, h_list
