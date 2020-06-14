import torch
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from torch import nn

from Constants import NODE_FEATURES_NUM, EDGE_FEATURE_NUM
from GNN.EdgeLayer import EdgeLayer
from GNN.NodeEmbeddingLayer import NodeEmbeddingLayer


class Net(nn.Module):
    def __init__(self, node_in_feats=NODE_FEATURES_NUM, node_out_feats=16,
                 edge_in_feats=EDGE_FEATURE_NUM, edge_out_feats=8,
                 hidden_sizes=[10], num_classes=2,
                 dropout_p=0.4):
        super(Net, self).__init__()
        assert len(hidden_sizes) > 0

        self.edge_layer = EdgeLayer(edge_in_feats, edge_out_feats)
        self.node_layer = NodeEmbeddingLayer(node_in_feats, node_out_feats)

        prev_hidden = node_out_feats + edge_out_feats
        #         print('prev_hidden', prev_hidden)

        self.hidden_layers = []
        for size in hidden_sizes:
            self.hidden_layers.append(GraphConv(prev_hidden, size))
            prev_hidden = size
        self.hidden_layers = nn.ModuleList(self.hidden_layers)

        self.last_layer = GraphConv(prev_hidden, num_classes)

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, g, features=None):
        edge_h = F.relu(self.edge_layer(g, features))
        node_h = F.relu(self.node_layer(g))
        h = torch.cat((edge_h, node_h), 1)
        #         print('h', h.size())

        for layer in self.hidden_layers:
            h = F.relu(self.dropout(layer(g, h)))
        h = self.last_layer(g, h)

        return h
