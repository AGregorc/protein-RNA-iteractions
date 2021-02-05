import torch
from dgl import DGLGraph
from torch import nn
import torch.nn.functional as F

from Constants import NODE_FEATURES_NAME


class NodeEmbeddingLayer(nn.Module):

    def __init__(self, in_feats, out_feats, embedding_dim=2, dropout_p=0.4, word_to_ixs=None, ignore_columns=None):
        super(NodeEmbeddingLayer, self).__init__()
        if ignore_columns is None:
            ignore_columns = []
        # if vocab_sizes is None:
        #     vocab_sizes = get_feat_wti_lens()
        self.dropout = nn.Dropout(p=dropout_p)

        # mask ignored columns
        self.mask_ignored = [i for i in range(in_feats) if i not in ignore_columns]
        in_feats = len(self.mask_ignored)

        self.col_to_embedding = {}
        self.emb_size = in_feats
        for col, wti in word_to_ixs.items():
            col = int(col)
            if col in ignore_columns:
                continue
            col -= len([i for i in ignore_columns if i < col])
            vocab_size = len(wti)
            self.col_to_embedding[str(col)] = nn.Embedding(vocab_size, embedding_dim)
            self.emb_size += embedding_dim - 1  # -1 because we eliminate one col from in_feats

        self.numerical_cols = [i for i in range(in_feats) if str(i) not in self.col_to_embedding.keys()]
        self.col_to_embedding = nn.ModuleDict(self.col_to_embedding)

        self.b_norm_in = nn.BatchNorm1d(num_features=self.emb_size)
        self.linear = nn.Linear(self.emb_size, out_feats)
        self.b_norm_out = nn.BatchNorm1d(num_features=out_feats)

    def forward(self, g_and_features):
        g, features = g_and_features

        # mask ignored columns
        features = features[:, self.mask_ignored]

        result = features[:, self.numerical_cols]
        for col, embedding in self.col_to_embedding.items():
            col = int(col)
            embeds = embedding(features[:, col].long())
            #             print('dtypes: ', embeds.dtype, result.dtype)
            result = torch.cat((result, embeds), 1)

        return self.dropout(F.relu(self.b_norm_out(self.linear(self.b_norm_in(result)))))
