import torch
from torch import nn

from Constants import NODE_FEATURES_NAME
from Preprocess import get_feat_wti_lens


class NodeEmbeddingLayer(nn.Module):

    def __init__(self, in_feats, out_feats, embedding_dim=5, vocab_sizes=None):
        super(NodeEmbeddingLayer, self).__init__()
        if vocab_sizes is None:
            vocab_sizes = get_feat_wti_lens()

        self.numerical_cols = [i for i in range(in_feats) if str(i) not in vocab_sizes.keys()]
        self.col_to_embedding = {}
        self.emb_size = in_feats
        for col, vocab_size in vocab_sizes.items():
            self.col_to_embedding[str(col)] = nn.Embedding(vocab_size, embedding_dim)
            self.emb_size += embedding_dim - 1  # -1 because we eliminate one col from in_feats
        self.col_to_embedding = nn.ModuleDict(self.col_to_embedding)

        self.linear = nn.Linear(self.emb_size, out_feats)

    def forward(self, g, features=None):
        if features is None:
            features = g.ndata[NODE_FEATURES_NAME]

        result = features[:, self.numerical_cols]
        for col, embedding in self.col_to_embedding.items():
            col = int(col)
            embeds = embedding(features[:, col].long())
            #             print('dtypes: ', embeds.dtype, result.dtype)
            result = torch.cat((result, embeds), 1)

        return self.linear(result)
