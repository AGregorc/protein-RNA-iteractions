import torch
from ignite.handlers import Checkpoint
from sklearn.model_selection import train_test_split
import torch.nn as nn

from Constants import LABEL_POSITIVE_COLOR, LABEL_NEGATIVE_COLOR, NODE_FEATURES_NUM
from Data.Data import get_dataset, save_dataset
from Data.Evaluate import calculate_metrics, majority_model_metrics, dataset_info, feature_importance
from GNN.MyModels import MyModels
from GNN.run_ignite import run
from GNN.InitialDataLayer import InitialDataLayer
from GNN.NetFirstGraphConvThenLinear import NetFirstGraphConvThenLinear
from GNN.NetSequenceWrapper import NetSequenceWrapper
from Data.PlotMPL import plot_from_file, plot_predicted, use_new_window, plot_training_history
from Data.Preprocess import is_labeled_positive
from split_dataset import get_train_val_test_data


def data(limit=1424, save=False):
    dataset, dataset_filenames, word_to_ixs, standardize = get_dataset(limit=limit)

    if save:
        save_dataset(dataset, dataset_filenames, word_to_ixs, *standardize, limit=limit)

    train_d, train_f, val_d, val_f, test_d, test_f = get_train_val_test_data(dataset, dataset_filenames)
    dataset_info(train_d, val_d, test_d)

    del dataset, dataset_filenames, test_d, test_f
    return train_d, train_f, val_d, val_f, word_to_ixs


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
        feature_importance(net, val_d)
    return net


def main():
    data_limit = 10
    model_name = 'first_linear_then_more_GraphConvs_then_linear'
    do_train = False
    metrics = False
    feat_importance = True

    train_d, train_f, val_d, val_f, word_to_ixs = data(data_limit, save=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Number of features', NODE_FEATURES_NUM)

    models = MyModels(word_to_ixs, ignore_columns=[])
    if model_name == 'all':
        for model_name in models.my_models:
            train_load_model(models, model_name, do_train, train_d, val_d, device, metrics, feat_importance)
    else:
        train_load_model(models, model_name, do_train, train_d, val_d, device, metrics, feat_importance)

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
