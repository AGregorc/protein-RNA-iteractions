import typing

import torch
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn.pytorch import GraphConv
from torch import nn
from torch.nn import Linear

from Constants import NODE_FEATURES_NUM, EDGE_FEATURE_NUM, NODE_FEATURES_NAME
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

        # edge_h = F.relu(self.edge_layer(g))
        node_h = self.node_layer(g_and_h)
        # h = torch.cat((edge_h, node_h), 1)
        h = node_h
        return g, h
