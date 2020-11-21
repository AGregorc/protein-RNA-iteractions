import torch
from torch import nn

from Constants import NODE_FEATURES_NAME


class NodeEmbeddingLayer(nn.Module):

    def __init__(self, in_feats, out_feats, embedding_dim=2, word_to_ixs=None):
        super(NodeEmbeddingLayer, self).__init__()
        # if vocab_sizes is None:
        #     vocab_sizes = get_feat_wti_lens()

        self.numerical_cols = [i for i in range(in_feats) if i not in word_to_ixs.keys()]
        self.col_to_embedding = {}
        self.emb_size = in_feats
        for col, wti in word_to_ixs.items():
            vocab_size = len(wti)
            self.col_to_embedding[str(col)] = nn.Embedding(vocab_size, embedding_dim)
            self.emb_size += embedding_dim - 1  # -1 because we eliminate one col from in_feats
        self.col_to_embedding = nn.ModuleDict(self.col_to_embedding)

        self.linear = nn.Linear(self.emb_size, out_feats)

    def forward(self, g_and_features):
        if type(g_and_features) is tuple:
            g, _ = g_and_features
        else:
            g = g_and_features

        features = g.ndata[NODE_FEATURES_NAME]

        result = features[:, self.numerical_cols]
        for col, embedding in self.col_to_embedding.items():
            col = int(col)
            embeds = embedding(features[:, col].long())
            #             print('dtypes: ', embeds.dtype, result.dtype)
            result = torch.cat((result, embeds), 1)

        return self.linear(result)
