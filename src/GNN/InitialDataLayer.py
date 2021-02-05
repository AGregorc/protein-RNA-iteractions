import typing

import torch
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from torch import nn
from torch.nn import Linear

from Constants import NODE_FEATURES_NUM, EDGE_FEATURE_NUM
from GNN.EdgeLayer import EdgeLayer
from GNN.NodeEmbeddingLayer import NodeEmbeddingLayer


class InitialDataLayer(nn.Module):
    def __init__(self, node_in_feats=NODE_FEATURES_NUM, node_out_feats=64,
                 edge_in_feats=EDGE_FEATURE_NUM, edge_out_feats=8,
                 dropout_p=0.4, word_to_ixs=None, ignore_columns=None):
        super(InitialDataLayer, self).__init__()
        # assert len(hidden_conv_sizes) > 0

        # self.edge_layer = EdgeLayer(edge_in_feats, edge_out_feats)
        self.node_layer = NodeEmbeddingLayer(node_in_feats, node_out_feats, dropout_p=dropout_p,
                                             word_to_ixs=word_to_ixs, ignore_columns=ignore_columns)
        # out_num = node_out_feats + edge_out_feats

    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)

    def forward(self, g_and_features):
        if type(g_and_features) is tuple:
            g, _ = g_and_features
        else:
            g = g_and_features

        # edge_h = F.relu(self.edge_layer(g))
        node_h = self.node_layer(g)
        # h = torch.cat((edge_h, node_h), 1)
        h = node_h
        return g, h
