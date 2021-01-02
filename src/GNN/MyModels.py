import copy
from os import listdir
from os.path import isfile, join, splitext

import torch
from ignite.handlers import Checkpoint
from torch import nn

import Constants
from GNN.ConcatNets import ConcatNets
from GNN.InitialDataLayer import InitialDataLayer
from GNN.NetFirstGraphConvThenLinear import NetFirstGraphConvThenLinear
from GNN.NetFirstLinearThenGraphConv import NetFirstLinearThenGraphConv
from GNN.NetGraphConv import NetGraphConv
from GNN.NetLinear import NetLinear
from GNN.NetSequenceWrapper import NetSequenceWrapper

# General net structure:
# -> Neural Layer -> BatchNorm -> ReLu(or other activation) -> Dropout -> Neural Layer ->


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
                    NetFirstGraphConvThenLinear(hidden_conv_sizes=[256, 128, 128],
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

    def load_models(self, model_name, device):
        model = copy.deepcopy(self.my_models[model_name])
        to_load = {'model': model}
        # checkpoint_fp = "../data/models/best/best_model_364_loss=-0.3888.pt"
        path = join(Constants.MODELS_PATH, model_name)
        model_files = [f for f in listdir(path) if isfile(join(path, f)) and splitext(f)[1] == '.pt']
        best_file = None
        best_loss = 1000
        for name in model_files:
            loss = abs(float(name.split('=')[1][:-3]))
            if loss <= best_loss:
                best_loss = loss
                best_file = name

        checkpoint = torch.load(join(path, best_file), map_location=device)
        Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)

        return model, best_loss

