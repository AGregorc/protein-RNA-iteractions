import typing

import torch
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from torch import nn
from torch.nn import Linear

from Constants import NODE_FEATURES_NUM, EDGE_FEATURE_NUM
from GNN.EdgeLayer import EdgeLayer
from GNN.NodeEmbeddingLayer import NodeEmbeddingLayer


class Net(nn.Module):
    def __init__(self, node_in_feats=NODE_FEATURES_NUM, node_out_feats=16,
                 edge_in_feats=EDGE_FEATURE_NUM, edge_out_feats=8,
                 hidden_conv_sizes=[10], num_classes=2,
                 dropout_p=0.4, hidden_linear_sizes=[10]):
        super(Net, self).__init__()
        assert len(hidden_conv_sizes) > 0
        self.dropout = nn.Dropout(p=dropout_p)

        self.edge_layer = EdgeLayer(edge_in_feats, edge_out_feats)
        self.node_layer = NodeEmbeddingLayer(node_in_feats, node_out_feats)

        prev_hidden = node_out_feats + edge_out_feats
        #         print('prev_hidden', prev_hidden)
        self.hidden_conv_layers = []
        for size in hidden_conv_sizes:
            self.hidden_conv_layers.append(GraphConv(prev_hidden, size))
            prev_hidden = size
        self.hidden_conv_layers = nn.ModuleList(self.hidden_conv_layers)

        self.hidden_linear_layers = []
        for size in hidden_linear_sizes:
            self.hidden_linear_layers.append(Linear(prev_hidden, size))
            prev_hidden = size
        self.hidden_linear_layers = nn.ModuleList(self.hidden_linear_layers)
        self.last_linear_layer = Linear(prev_hidden, num_classes)

    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)

    def forward(self, g, features=None):
        edge_h = F.relu(self.edge_layer(g, features))
        node_h = F.relu(self.node_layer(g))
        h = torch.cat((edge_h, node_h), 1)
        #         print('h', h.size())

        for layer in self.hidden_conv_layers:
            h = F.relu(self.dropout(layer(g, h)))

        for layer in self.hidden_linear_layers:
            h = F.relu(self.dropout(layer(h)))
        h = self.last_linear_layer(h)

        return h
