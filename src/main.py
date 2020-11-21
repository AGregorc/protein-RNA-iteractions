import torch
from ignite.handlers import Checkpoint
from sklearn.model_selection import train_test_split
import torch.nn as nn

from Constants import LABEL_POSITIVE_COLOR, LABEL_NEGATIVE_COLOR, NODE_FEATURES_NUM
from Data.Data import get_dataset, save_dataset
from Data.Evaluate import calculate_metrics
from GNN.MyModels import MyModels
from GNN.run_ignite import run
from GNN.InitialDataLayer import InitialDataLayer
from GNN.NetFirstGraphConvThenLinear import NetFirstGraphConvThenLinear
from GNN.NetSequenceWrapper import NetSequenceWrapper
from Data.PlotMPL import plot_from_file, plot_predicted, use_new_window, plot_training_history
from Data.Preprocess import is_labeled_positive


def main():
    limit = 500
    dataset, dataset_filenames, word_to_ixs, standardize = get_dataset(limit=limit)
    # load_filename=Constants.SAVED_GRAPHS_PATH + 'graph_data_2.bin')

    save_dataset(dataset, dataset_filenames, word_to_ixs, *standardize, limit=limit)
    # dataset, dataset_filenames = load_dataset()
    train_d, test_d, train_f, test_f = train_test_split(dataset, dataset_filenames, shuffle=False, test_size=0.22)
    del dataset, dataset_filenames
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(NODE_FEATURES_NUM)

    models = MyModels(word_to_ixs)
    model_name = 'two_branches_small'
    net = models.my_models[model_name]
    print(net)
    # to_load = {'model': net}
    # checkpoint_fp = "../data/models/best/best_model_364_loss=-0.3888.pt"
    # checkpoint = torch.load(checkpoint_fp)
    # Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)
    #
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.6], device=device))
    # criterion = nn.CrossEntropyLoss()
    training_h, validation_h, whole_training_h = run(net, train_d, test_d, criterion=criterion,
                                                     batch_size=10, epochs=1000,
                                                     model_name=model_name,
                                                     log_dir='/tmp/tensorboard_logs/')
    plot_training_history(training_h, validation_h)
    calculate_metrics(test_d, net, print_model_name=model_name)
    # calculate_metrics(train_d, net, print_model_name=model_name)

    use_new_window()
    #
    plot_from_file(train_f[0], lambda atom: LABEL_POSITIVE_COLOR if is_labeled_positive(atom) else LABEL_NEGATIVE_COLOR,
                   word_to_ixs, standardize=standardize)
    plot_predicted(train_f[0], net, word_to_ixs, standardize=standardize)

    print('press any key to continue ...')
    # input()


if __name__ == '__main__':
    main()
