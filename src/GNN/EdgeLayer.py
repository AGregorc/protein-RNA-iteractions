import dgl.function as fn
import torch.nn as nn

from Constants import EDGE_FEATURE_NAME

in_out_key = 'h'
edge_layer_msg = fn.copy_edge(edge=in_out_key, out='m')
edge_layer_reduce = fn.sum(msg='m', out=in_out_key)


class EdgeLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(EdgeLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, features=None):
        if features is None:
            features = g.edata[EDGE_FEATURE_NAME]
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.edata[in_out_key] = features
            g.update_all(edge_layer_msg, edge_layer_reduce)
            h = g.ndata[in_out_key]
            return self.linear(h)
