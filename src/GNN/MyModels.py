
from GNN.ConcatNets import ConcatNets
from GNN.InitialDataLayer import InitialDataLayer
from GNN.NetFirstGraphConvThenLinear import NetFirstGraphConvThenLinear
from GNN.NetLinear import NetLinear
from GNN.NetSequenceWrapper import NetSequenceWrapper


class MyModels:
    def __init__(self, word_to_ixs):

        self.my_models = {
            'first_one_GraphConv_then_linear':
                NetSequenceWrapper(
                    InitialDataLayer(word_to_ixs=word_to_ixs),
                    NetFirstGraphConvThenLinear(hidden_conv_sizes=[32],
                                                hidden_linear_sizes=[128, 128, 64, 64, 32, 16])
                ),
            'first_more_GraphConvs_then_linear':
                NetSequenceWrapper(
                    InitialDataLayer(word_to_ixs=word_to_ixs),
                    NetFirstGraphConvThenLinear(hidden_conv_sizes=[32, 32, 32],
                                                hidden_linear_sizes=[128, 128, 64, 64, 32, 16])
                ),
            'two_branches':
                NetSequenceWrapper(
                    InitialDataLayer(word_to_ixs=word_to_ixs),
                    ConcatNets([
                        NetLinear(out_features=32, hidden_linear_sizes=[128, 128, 64, 64, 32]),
                        NetFirstGraphConvThenLinear(out_features=32, hidden_conv_sizes=[64, 32, 32],
                                                    hidden_linear_sizes=[128, 64, 32, 16])
                    ]),
                    NetLinear(in_features=32 + 32, out_features=2, hidden_linear_sizes=[128, 64, 32, 16])
                )
        }
