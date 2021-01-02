import torch
from ignite.handlers import Checkpoint
from sklearn.model_selection import train_test_split
import torch.nn as nn

from Constants import LABEL_POSITIVE_COLOR, LABEL_NEGATIVE_COLOR, NODE_FEATURES_NUM
from Data.Data import get_dataset, save_dataset
from Data.Evaluate import calculate_metrics, majority_model_metrics
from GNN.MyModels import MyModels
from GNN.run_ignite import run
from GNN.InitialDataLayer import InitialDataLayer
from GNN.NetFirstGraphConvThenLinear import NetFirstGraphConvThenLinear
from GNN.NetSequenceWrapper import NetSequenceWrapper
from Data.PlotMPL import plot_from_file, plot_predicted, use_new_window, plot_training_history
from Data.Preprocess import is_labeled_positive


def data(limit=1424, save=False):
    dataset, dataset_filenames, word_to_ixs, standardize = get_dataset(limit=limit)

    if save:
        save_dataset(dataset, dataset_filenames, word_to_ixs, *standardize, limit=limit)
    train_d, test_val_d, train_f, test_val_f = train_test_split(dataset, dataset_filenames, shuffle=False,
                                                                test_size=0.4)
    val_d, test_d, val_f, test_f = train_test_split(test_val_d, test_val_f, shuffle=False, test_size=0.5)

    del dataset, dataset_filenames, test_val_d, test_val_f, test_d, test_f
    return train_d, train_f, val_d, val_f, word_to_ixs


def train_load_model(my_models, model_name, do_train, train_d, val_d, device, calc_metrics=True):
    net = my_models.my_models[model_name]

    print(net)
    if do_train:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.6], device=device))
        training_h, validation_h, whole_training_h = run(net, train_d, val_d,
                                                         device=device,
                                                         criterion=criterion,
                                                         batch_size=10,
                                                         epochs=1000,
                                                         model_name=model_name)
        print(f'Run for {model_name} is done\n\n')

        plot_training_history(training_h, validation_h, model_name=model_name, save=True)
    else:
        net, loss = my_models.load_models(model_name, device)
    if calc_metrics:
        calculate_metrics(val_d, net, print_model_name=model_name, save=True)
    return net


def main():
    data_limit = 1424
    model_name = 'just_linear'
    do_train = True
    metrics = True

    train_d, train_f, val_d, val_f, word_to_ixs = data(data_limit, save=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Number of features', NODE_FEATURES_NUM)

    models = MyModels(word_to_ixs)
    if model_name == 'all':
        for model_name in models.my_models:
            train_load_model(models, model_name, do_train, train_d, val_d, device, metrics)
    else:
        train_load_model(models, model_name, do_train, train_d, val_d, device, metrics)

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
