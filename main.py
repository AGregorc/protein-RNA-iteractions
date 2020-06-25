import torch

import Constants
from Data import create_dataset, load_dataset, save_dataset, get_dataset
from Evaluate import calculate_metrics
from GNN.GNNModel import GNNModel
from PlotMPL import plot_from_file, plot_predicted, use_new_window
from Preprocess import get_labeled_color, save_feat_word_to_ixs

dataset, dataset_filenames = get_dataset(limit=3)
save_dataset(dataset, dataset_filenames)
dataset, dataset_filenames = load_dataset()

# save_feat_word_to_ixs()


def split_dataset(data, data_filenames, pos_neg_ratio=0.14):
    interactions_percentages = []
    for graph in data:
        positives = torch.sum((graph.ndata[Constants.LABEL_NODE_NAME] == Constants.LABEL_POSITIVE).int())
        interactions_percentages.append(positives.item() / graph.ndata[Constants.LABEL_NODE_NAME].shape[0])

    sorted_data = sorted(zip(interactions_percentages, data, data_filenames), reverse=True)
    interactions_percentages, data, data_filenames = list(zip(*sorted_data))

    split_index = int(len(data) / 2)
    for i, p in enumerate(interactions_percentages):
        if p < pos_neg_ratio:
            split_index = i
            break
    train_dataset_1 = data[:split_index]
    train_filenames_1 = data_filenames[:split_index]
    print(f'Split index: {split_index}')

    train_dataset_2 = data[split_index:]
    train_filenames_2 = data_filenames[split_index:]

    return train_dataset_1, train_filenames_1, train_dataset_2, train_filenames_2


my_model = GNNModel()

# my_model.train(train_dataset_1)
# my_model.train(train_dataset_2)
my_model.train(dataset)

calculate_metrics(dataset, my_model, my_model.get_name())

# use_new_window()
#
# plot_from_file(dataset_filenames[10], get_labeled_color)
# plot_predicted(dataset_filenames[10], my_model)

# print('press any key to continue ...')
# input()
