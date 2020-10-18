import torch
from Bio.PDB import DSSP
from sklearn.model_selection import train_test_split

import Constants
from Data import create_dataset, load_dataset, save_dataset, get_dataset
from Evaluate import calculate_metrics
from GNN.GNNModel import GNNModel
from GNN.Net import Net
from PlotMPL import plot_from_file, plot_predicted, use_new_window
from Preprocess import get_labeled_color, save_feat_word_to_ixs


def main():
    dataset, dataset_filenames = create_dataset(limit=50)
    # load_filename=Constants.SAVED_GRAPHS_PATH + 'graph_data_2.bin')

    # save_dataset(dataset, dataset_filenames)
    # dataset, dataset_filenames = load_dataset()
    train_d, test_d, train_f, test_f = train_test_split(dataset, dataset_filenames, test_size=0.17)
    del dataset, dataset_filenames

    net = Net(hidden_conv_sizes=[32], hidden_linear_sizes=[128, 128, 64, 64, 32, 16], dropout_p=0.5)
    my_model = GNNModel(net)
    #
    my_model.train(train_d, loss_weights=[1.0, 8.6], batch_size=5, epochs=30)
    #
    calculate_metrics(test_d, my_model, my_model.get_name())

    use_new_window()
    #
    plot_from_file('1a1t.pdb', lambda atom: None)
    plot_predicted('1a1t.pdb', my_model)

    # print('press any key to continue ...')
    # input()


if __name__ == '__main__':
    main()
