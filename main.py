import torch

import Constants
from Data import create_dataset
from GNN.GNNModel import GNNModel
from PlotMPL import plot_from_file, plot_predicted, use_new_window
from Preprocess import get_labeled_color

dataset, dataset_filenames = create_dataset(Constants.PDB_PATH, limit=20)


###########################################################################################
interactions_percentages = []
for graph in dataset:
    positives = torch.sum((graph.ndata[Constants.LABEL_NODE_NAME] == Constants.LABEL_POSITIVE).int())
    interactions_percentages.append(positives.item()/graph.ndata[Constants.LABEL_NODE_NAME].shape[0])

sorted_data = sorted(zip(interactions_percentages, dataset, dataset_filenames), reverse=True)
interactions_percentages, dataset, dataset_filenames = list(zip(*sorted_data))

SPLIT_TRAINING_INTER_PER = 0.14
split_index = int(len(dataset)/2)
for i, p in enumerate(interactions_percentages):
    if p < SPLIT_TRAINING_INTER_PER:
        split_index = i
        break
train_dataset_1 = dataset[:split_index]
train_filenames_1 = dataset_filenames[:split_index]
print(f'Split index: {split_index}')

train_dataset_2 = dataset[split_index:]
train_filenames_2 = dataset_filenames[split_index:]
###########################################################################################

my_model = GNNModel(epochs=10)

my_model.train(train_dataset_1)

# my_model.train(train_dataset_2)


use_new_window()

# plot_from_file(dataset_filenames[0], get_labeled_color)
# plot_predicted(dataset_filenames[0], my_model)
