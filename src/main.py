import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn

from Constants import LABEL_POSITIVE_COLOR, LABEL_NEGATIVE_COLOR
from Data.Data import get_dataset
from Data.Evaluate import calculate_metrics
from GNN.MyModels import MyModels
from GNN.run_ignite import run
from GNN.InitialDataLayer import InitialDataLayer
from GNN.NetFirstGraphConvThenLinear import NetFirstGraphConvThenLinear
from GNN.NetSequenceWrapper import NetSequenceWrapper
from Data.PlotMPL import plot_from_file, plot_predicted, use_new_window
from Data.Preprocess import is_labeled_positive


def main():
    limit = 500
    dataset, dataset_filenames, word_to_ixs, standardize = get_dataset(limit=limit)
    # load_filename=Constants.SAVED_GRAPHS_PATH + 'graph_data_2.bin')

    # save_dataset(dataset, dataset_filenames, word_to_ixs, *standardize, limit=limit)
    # dataset, dataset_filenames = load_dataset()
    train_d, test_d, train_f, test_f = train_test_split(dataset, dataset_filenames, test_size=0.22)
    del dataset, dataset_filenames
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = MyModels(word_to_ixs)
    net = models.my_models['first_one_GraphConv_then_linear']
    # to_load = {'model': net}
    # checkpoint_fp = "../data/models/best/best_model_364_loss=-0.3888.pt"
    # checkpoint = torch.load(checkpoint_fp)
    # Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)
    #
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 8.5], device=device))
    run(net, train_d, test_d, criterion=criterion, batch_size=5, epochs=1000, log_dir='/tmp/tensorboard_logs/')
    #
    calculate_metrics(test_d, net, 'main_run')

    use_new_window()
    #
    plot_from_file('1a1t.pdb', lambda atom: LABEL_POSITIVE_COLOR if is_labeled_positive(atom) else LABEL_NEGATIVE_COLOR,
                   word_to_ixs, standardize=standardize)
    plot_predicted('1a1t.pdb', net, word_to_ixs, standardize=standardize)

    print('press any key to continue ...')
    # input()


if __name__ == '__main__':
    main()
