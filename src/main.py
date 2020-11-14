import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn

from Constants import LABEL_POSITIVE_COLOR, LABEL_NEGATIVE_COLOR
from Data import save_dataset, get_dataset
from Evaluate import calculate_metrics
from GNN.GNNModel import GNNModel
from GNN.InitialDataLayer import InitialDataLayer
from GNN.NetFirstGraphConvThenLinear import NetFirstGraphConvThenLinear
from GNN.NetSequenceWrapper import NetSequenceWrapper
from PlotMPL import plot_from_file, plot_predicted, use_new_window
from Preprocess import is_labeled_positive


def main():
    limit = 5
    dataset, dataset_filenames, word_to_ixs, standardize = get_dataset(limit=limit)
    # load_filename=Constants.SAVED_GRAPHS_PATH + 'graph_data_2.bin')

    save_dataset(dataset, dataset_filenames, word_to_ixs, *standardize, limit=limit)
    # dataset, dataset_filenames = load_dataset()
    train_d, test_d, train_f, test_f = train_test_split(dataset, dataset_filenames, test_size=0.22)
    del dataset, dataset_filenames

    net = nn.Sequential(
        InitialDataLayer(word_to_ixs=word_to_ixs),
        NetFirstGraphConvThenLinear(hidden_conv_sizes=[32], hidden_linear_sizes=[128, 128, 64, 64, 32, 16], dropout_p=0.5)
    )
    my_model = GNNModel(net)
    #
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 8.5], device=my_model.device))
    my_model.train(train_d, test_d, criterion=criterion, batch_size=5, epochs=30)
    #
    calculate_metrics(test_d, my_model, my_model.get_name())

    use_new_window()
    #
    plot_from_file('1a1t.pdb', lambda atom: LABEL_POSITIVE_COLOR if is_labeled_positive(atom) else LABEL_NEGATIVE_COLOR,
                   word_to_ixs, standardize=standardize)
    plot_predicted('1a1t.pdb', my_model, word_to_ixs, standardize=standardize)

    print('press any key to continue ...')
    # input()


if __name__ == '__main__':
    main()
