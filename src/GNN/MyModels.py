import copy
import json
from collections import OrderedDict
from os import listdir
from os.path import isfile, join, splitext, getmtime
from pathlib import Path

import torch
from ignite.handlers import Checkpoint
from torch import nn

import Constants
from GNN.ConcatNets import ConcatNets
from GNN.InitialDataLayer import InitialDataLayer
from GNN.NetFirstGraphConvThenLinear import NetFirstGraphConvThenLinear
from GNN.NetFirstLinearThenGraphConv import NetFirstLinearThenGraphConv
from GNN.NetGATConv import NetGATConv
from GNN.NetGraphConv import NetGraphConv
from GNN.NetLinear import NetLinear
from GNN.NetSequenceWrapper import NetSequenceWrapper


# General net structure:
# -> Neural Layer -> BatchNorm -> ReLu(or other activation) -> Dropout -> Neural Layer ->


class MyModels:
    def __init__(self, word_to_ixs, ignore_columns=None, seed=7):
        torch.manual_seed(seed)
        self.threshold_filename = 'thresholds.json'
        self.my_models = {
            'just_linear':
                NetSequenceWrapper(
                    InitialDataLayer(word_to_ixs=word_to_ixs, ignore_columns=ignore_columns),
                    NetLinear(out_features=2, hidden_linear_sizes=[256, 128, 128, 64, 64, 32, 16])
                ),
            'first_one_GraphConv_then_linear':
                NetSequenceWrapper(
                    InitialDataLayer(word_to_ixs=word_to_ixs, ignore_columns=ignore_columns),
                    NetFirstGraphConvThenLinear(hidden_conv_sizes=[256],
                                                hidden_linear_sizes=[128, 64, 64, 32, 16])
                ),
            'first_more_GraphConvs_then_linear':  # sploh ne overfitta enega grafa
                NetSequenceWrapper(
                    InitialDataLayer(word_to_ixs=word_to_ixs, ignore_columns=ignore_columns),
                    NetFirstGraphConvThenLinear(hidden_conv_sizes=[256, 128, 128],
                                                hidden_linear_sizes=[128, 128, 64, 64, 32, 16])
                ),
            'first_linear_then_GraphConvs':  # sploh ne overfitta enega grafa
                NetSequenceWrapper(
                    InitialDataLayer(word_to_ixs=word_to_ixs, ignore_columns=ignore_columns),
                    NetFirstLinearThenGraphConv(hidden_conv_sizes=[64, 128, 128],
                                                hidden_linear_sizes=[128, 128, 64, 64, 32, 16])
                ),
            'first_linear_then_more_GraphConvs_then_linear':
                NetSequenceWrapper(
                    InitialDataLayer(word_to_ixs=word_to_ixs, ignore_columns=ignore_columns),
                    NetFirstLinearThenGraphConv(out_features=64, hidden_linear_sizes=[256, 128, 64, 32, 16],
                                                hidden_conv_sizes=[64]),
                    NetLinear(in_features=64, out_features=2, hidden_linear_sizes=[32, 16, 8, 4])
                ),
            'design_space_inspired':
                NetSequenceWrapper(
                    InitialDataLayer(word_to_ixs=word_to_ixs, ignore_columns=ignore_columns),
                    NetFirstLinearThenGraphConv(out_features=64, hidden_linear_sizes=[256, 128, 64],
                                                hidden_conv_sizes=[64, 100, 64]),
                    NetLinear(in_features=64, out_features=2, hidden_linear_sizes=[32, 16, 8, 4])
                ),
            'design_space_gat':
                NetSequenceWrapper(
                    InitialDataLayer(word_to_ixs=word_to_ixs, ignore_columns=ignore_columns),
                    NetLinear(out_features=64, hidden_linear_sizes=[256, 128, 64]),
                    NetGATConv(in_features=64, out_features=64, hidden_gat_sizes=[64, 100, 64]),
                    NetLinear(in_features=64, out_features=2, hidden_linear_sizes=[32, 16, 8, 4])
                ),
            'two_branches_small':  # lahko overfitta <3
                NetSequenceWrapper(
                    ConcatNets([
                        nn.Sequential(
                            InitialDataLayer(word_to_ixs=word_to_ixs, ignore_columns=ignore_columns),
                            NetLinear(out_features=128, hidden_linear_sizes=[256, 128]),
                        ),
                        nn.Sequential(
                            InitialDataLayer(word_to_ixs=word_to_ixs, ignore_columns=ignore_columns),
                            NetGraphConv(out_features=32, hidden_conv_sizes=[64, 64, 32])
                        ),
                    ]),
                    NetLinear(in_features=128 + 32, out_features=2, hidden_linear_sizes=[64, 64, 32, 16])
                ),

            'two_branches':
                NetSequenceWrapper(
                    ConcatNets([
                        nn.Sequential(
                            InitialDataLayer(word_to_ixs=word_to_ixs, ignore_columns=ignore_columns),
                            NetLinear(out_features=128, hidden_linear_sizes=[256, 128, 64, 64, 32]),
                        ),
                        nn.Sequential(
                            InitialDataLayer(word_to_ixs=word_to_ixs, ignore_columns=ignore_columns),
                            NetGraphConv(out_features=32, hidden_conv_sizes=[64, 64, 128, 128])
                        ),
                    ]),
                    NetLinear(in_features=128 + 32, out_features=2, hidden_linear_sizes=[128, 64, 32, 16])
                ),

            'two_branches_gat':
                NetSequenceWrapper(
                    ConcatNets([
                        nn.Sequential(
                            InitialDataLayer(word_to_ixs=word_to_ixs, ignore_columns=ignore_columns),
                            NetLinear(out_features=128, hidden_linear_sizes=[256, 128, 64, 64, 32]),
                        ),
                        nn.Sequential(
                            InitialDataLayer(word_to_ixs=word_to_ixs, ignore_columns=ignore_columns),
                            NetGATConv(out_features=32, hidden_gat_sizes=[64, 64, 128, 128])
                        ),
                    ]),
                    NetLinear(in_features=128 + 32, out_features=2, hidden_linear_sizes=[128, 64, 32, 16])
                )
        }

    def save_thresholds(self, model_name, thresholds):
        path = join(Constants.MODELS_PATH, model_name, self.threshold_filename)
        with open(path, 'w') as f:
            json.dump(thresholds, f, indent=2)

    def get_thresholds(self, model_name):
        path = join(Constants.MODELS_PATH, model_name, self.threshold_filename)
        # return {}
        try:
            with open(path, 'r') as f:
                thresholds = json.load(f)
            return thresholds
        except FileNotFoundError:
            return None

    def get_model(self, model_name, device, prefix='', path=None):
        if path is None:
            path = join(Constants.MODELS_PATH, model_name)
        best_file, best_loss = get_model_filename(path, prefix)
        model = copy.deepcopy(self.my_models[model_name])
        to_load = {'model': model}

        checkpoint = torch.load(join(path, best_file), map_location=device)
        Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)

        return model, best_loss, self.get_thresholds(model_name)


def rename_state_dict_keys(source, key_transformation, device, target=None):
    """
    source             -> Path to the saved state dict.
    key_transformation -> Function that accepts the old key names of the state
                          dict as the only argument and returns the new key name.
    target (optional)  -> Path at which the new state dict should be saved
                          (defaults to `source`)
    Example:
    Rename the key `layer.0.weight` `layer.1.weight` and keep the names of all
    other keys.
    ```py
    def key_transformation(old_key):
        if old_key == "layer.0.weight":
            return "layer.1.weight"
        return old_key
    rename_state_dict_keys(state_dict_path, key_transformation)
    ```
    """
    if target is None:
        target = source

    state_dict = torch.load(source, map_location=device)
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_key = key_transformation(key)
        new_state_dict[new_key] = value

    torch.save(new_state_dict, target)


def get_model_filename(path, prefix=''):
    model_files = [f for f in listdir(path) if isfile(join(path, f)) and splitext(f)[1] == '.pt']
    best_file = None
    best_loss = 1000
    for name in model_files:
        if prefix not in name:
            continue
        loss = abs(float(name.split('=')[1][:-3]))
        if loss <= best_loss:
            best_loss = loss
            best_file = name
    return best_file, best_loss


def list_models(path, limit=None):
    model_files = [f.name for f in sorted(Path(path).iterdir(), key=getmtime, reverse=True)
                   if isfile(f) and splitext(f.name)[1] == '.pt']
    if limit is not None:
        model_files = model_files[:limit]
    return model_files
