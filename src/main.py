import copy

import torch
import torch.nn as nn

from Constants import NODE_FEATURES_NUM
from Data.Data import get_dataset, save_dataset
from Data.Evaluate import calculate_metrics, majority_model_metrics, dataset_info, feature_importance
from Data.PlotMPL import plot_training_history, visualize_model
from GNN.MyModels import MyModels
from GNN.run_ignite import run
from split_dataset import get_train_val_test_data


def data(limit=1424, save=False):
    dataset, dataset_filenames, word_to_ixs, standardize = get_dataset(limit=limit, individual=False)

    if save:
        save_dataset(dataset, dataset_filenames, word_to_ixs, *standardize, limit=limit)

    train_d, train_f, val_d, val_f, test_d, test_f = get_train_val_test_data(dataset, dataset_filenames)
    dataset_info(train_d, val_d, test_d)

    del dataset, dataset_filenames, test_d, test_f
    return train_d, train_f, val_d, val_f, word_to_ixs


def tune_hyperparameter(my_models, model_name, train_d, val_d, device, weights=None):
    if weights is None:
        weights = [1.0, 2.0, 3.0, 5.0, 6.0, 7.54, 9.0]
    net_original = my_models.my_models[model_name]
    print(net_original, model_name)
    best_auc = 0
    best_hyperparam = None

    for weight in weights:
        print(f'Training with weight: {weight}')
        net = copy.deepcopy(net_original)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, weight], device=device))
        training_h, validation_h, whole_training_h = run(net, train_d, val_d,
                                                         device=device,
                                                         criterion=criterion,
                                                         batch_size=10,
                                                         epochs=1000,
                                                         model_name=model_name,
                                                         model_name_prefix='w_'+str(weight))
        thresholds, auc = calculate_metrics(val_d, net, print_model_name=model_name, do_plot=False, save=False)
        curr_auc = auc['y_combine_all_smooth_percent']
        # my_models.save_thresholds(model_name, thresholds)
        if curr_auc >= best_auc:
            best_hyperparam = weight
            best_auc = curr_auc
    print('Best weight: ', best_hyperparam)
    return best_hyperparam


def train_load_model(my_models, model_name, do_train, train_d, val_d, device, calc_metrics=True, calc_feat_i=False):
    net = my_models.my_models[model_name]

    print(net, model_name)
    if do_train:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 6.0], device=device))
        training_h, validation_h, whole_training_h = run(net, train_d, val_d,
                                                         device=device,
                                                         criterion=criterion,
                                                         batch_size=10,
                                                         epochs=10000,
                                                         model_name=model_name)
        print(f'Run for {model_name} is done\n\n')

        plot_training_history(training_h, validation_h, model_name=model_name, save=True)
    else:
        net, loss, thresholds = my_models.load_models(model_name, device)
    if calc_metrics:
        thresholds, auc = calculate_metrics(val_d, net, print_model_name=model_name, save=True)
        my_models.save_thresholds(model_name, thresholds)
    if calc_feat_i:
        feature_importance(net, val_d, model_name, save=True)
    return net


def main():
    class WhatUWannaDoNow:
        TRAIN = 0
        TUNE_HYPERPARAMS = 1
        VISUALIZE_MODELS = 2
        FEATURE_IMPORTANCE = 3

    data_limit = 1424
    model_names = ['first_linear_then_more_GraphConvs_then_linear',
                   'design_space_inspired',
                   'design_space_gat',
                   'two_branches_small',
                   'two_branches']
    # model_names = 'all'
    what_to_do = WhatUWannaDoNow.TUNE_HYPERPARAMS
    metrics = False

    train_d, train_f, val_d, val_f, word_to_ixs = data(data_limit, save=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Number of features {NODE_FEATURES_NUM}, device {device}')

    models = MyModels(word_to_ixs, ignore_columns=[])
    if model_names == 'all':
        model_names = list(models.my_models)
    elif not isinstance(model_names, list):
        model_names = [model_names]

    for model_name in model_names:
        if what_to_do == WhatUWannaDoNow.TRAIN:
            train_load_model(models, model_name, True, train_d, val_d, device, metrics, False)
        if what_to_do == WhatUWannaDoNow.TUNE_HYPERPARAMS:
            tune_hyperparameter(models, model_name, train_d, val_d, device)
        if what_to_do == WhatUWannaDoNow.VISUALIZE_MODELS:
            visualize_model(models.my_models[model_name], model_name, val_d[0])
        if what_to_do == WhatUWannaDoNow.FEATURE_IMPORTANCE:
            train_load_model(models, model_name, False, train_d, val_d, device, metrics, True)

    if metrics:
        majority_model_metrics(val_d, save=True)

    # use_new_window()
    # #
    # plot_from_file(train_f[0],
    #                lambda atom: LABEL_POSITIVE_COLOR if is_labeled_positive(atom) else LABEL_NEGATIVE_COLOR,
    #                word_to_ixs, standardize=standardize)
    # plot_predicted(train_f[0], net, word_to_ixs, standardize=standardize)


if __name__ == '__main__':
    main()
