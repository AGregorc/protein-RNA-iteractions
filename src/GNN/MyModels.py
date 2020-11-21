import torch
from torch import nn

from GNN.ConcatNets import ConcatNets
from GNN.InitialDataLayer import InitialDataLayer
from GNN.NetFirstGraphConvThenLinear import NetFirstGraphConvThenLinear
from GNN.NetFirstLinearThenGraphConv import NetFirstLinearThenGraphConv
from GNN.NetGraphConv import NetGraphConv
from GNN.NetLinear import NetLinear
from GNN.NetSequenceWrapper import NetSequenceWrapper


class MyModels:
    def __init__(self, word_to_ixs, seed=7):
        torch.manual_seed(seed)
        self.my_models = {
            'just_linear':
                NetSequenceWrapper(
                    InitialDataLayer(word_to_ixs=word_to_ixs),
                    NetLinear(out_features=2, hidden_linear_sizes=[256, 128, 128, 64, 64, 32, 16])
                ),
            'first_one_GraphConv_then_linear':
                NetSequenceWrapper(
                    InitialDataLayer(word_to_ixs=word_to_ixs),
                    NetFirstGraphConvThenLinear(hidden_conv_sizes=[256],
                                                hidden_linear_sizes=[128, 64, 64, 32, 16])
                ),
            'first_more_GraphConvs_then_linear':  # sploh ne overfitta enega grafa
                NetSequenceWrapper(
                    InitialDataLayer(word_to_ixs=word_to_ixs),
                    NetFirstGraphConvThenLinear(hidden_conv_sizes=[256, 400, 400],
                                                hidden_linear_sizes=[128, 128, 64, 64, 32, 16])
                ),
            'first_linear_then_more_GraphConvs_then_linear':  # sploh ne overfitta enega grafa
                NetSequenceWrapper(
                    InitialDataLayer(word_to_ixs=word_to_ixs),
                    NetFirstLinearThenGraphConv(out_features=64, hidden_linear_sizes=[256, 128, 64, 32, 16],
                                                hidden_conv_sizes=[64]),
                    NetLinear(in_features=64, out_features=2, hidden_linear_sizes=[32, 16, 8, 4])
                ),

            'two_branches_small':  # lahko overfitta <3
                NetSequenceWrapper(
                    ConcatNets([
                        nn.Sequential(
                            InitialDataLayer(word_to_ixs=word_to_ixs),
                            NetLinear(out_features=128, hidden_linear_sizes=[256, 128]),
                        ),
                        nn.Sequential(
                            InitialDataLayer(word_to_ixs=word_to_ixs),
                            NetGraphConv(out_features=32, hidden_conv_sizes=[64, 64, 32])
                        ),
                    ]),
                    NetLinear(in_features=128 + 32, out_features=2, hidden_linear_sizes=[64, 64, 32, 16])
                ),

            'two_branches':  # sploh ne overfitta enega grafa
                NetSequenceWrapper(
                    ConcatNets([
                        nn.Sequential(
                            InitialDataLayer(word_to_ixs=word_to_ixs),
                            NetLinear(out_features=128, hidden_linear_sizes=[256, 128, 64, 64, 32]),
                        ),
                        nn.Sequential(
                            InitialDataLayer(word_to_ixs=word_to_ixs),
                            NetGraphConv(out_features=32, hidden_conv_sizes=[64, 64, 128, 128])
                        ),
                    ]),
                    NetLinear(in_features=128 + 32, out_features=2, hidden_linear_sizes=[128, 64, 32, 16])
                )
        }

