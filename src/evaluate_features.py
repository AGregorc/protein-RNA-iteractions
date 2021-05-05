import dgl
import torch
from captum.attr import IntegratedGradients, NoiseTunnel, DeepLift, GradientShap, FeatureAblation
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

from Constants import NODE_FEATURES_NUM, NODE_FEATURES_NAME
from Data.Evaluate import calculate_metrics
from GNN.MyModels import MyModels
from GNN.run_ignite import run
from main import data


def train_ignore_features():
    data_limit = 10
    model_name = 'first_linear_then_more_GraphConvs_then_linear'
    predict_type = 'y_combine_all_percent'

    train_d, train_ids, val_d, val_ids, word_to_ixs = data(data_limit)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Number of features', NODE_FEATURES_NUM)
    x_list = list(range(NODE_FEATURES_NUM))
    auc_list = []

    for i in x_list:
        my_models = MyModels(word_to_ixs, ignore_columns=[i])
        net = my_models.my_models[model_name]
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 6.0], device=device))
        training_h, validation_h, whole_training_h = run(net, train_d, val_d,
                                                         device=device,
                                                         criterion=criterion,
                                                         batch_size=10,
                                                         epochs=2,
                                                         model_name=model_name,
                                                         save=False)
        thresholds, aucs = calculate_metrics(val_d, net, print_model_name=None, do_plot=False, save=False)
        auc_list.append(aucs[predict_type])

    plt.figure()
    plt.bar(x_list, auc_list)
    plt.show()


if __name__ == '__main__':
    train_ignore_features()
