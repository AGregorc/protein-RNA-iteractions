import json
import os
import threading
import time
import warnings
from multiprocessing import Process, Pool
from queue import Empty, Queue

import torch

from Bio.PDB import PDBParser
from dgl.data import save_graphs, load_graphs

import Constants
from Preprocess import create_graph_sample, save_feat_word_to_ixs, load_feat_word_to_ixs


def my_pdb_parser(filename, directory_path=Constants.PDB_PATH):
    parser = PDBParser()
    with warnings.catch_warnings(record=True):
        structure = parser.get_structure(os.path.splitext(filename)[0], os.path.join(directory_path, filename))
    model = structure[0]
    # print(structure)
    # chains = list(model.get_chains())
    return create_graph_sample(model)


def create_graph_process(args):
    my_filename, directory_path = args
    # print(f'[{os.getpid()}] got something to work :O')
    start_time = time.time()
    graph, atoms, pairs, labels = my_pdb_parser(my_filename, directory_path)
    print(f'[{os.getpid()}] File {my_filename} added in {(time.time() - start_time):.1f}s')
    return my_filename, graph


def create_dataset(directory_path=Constants.PDB_PATH, limit=None):
    directory = os.fsencode(directory_path)

    idx = 0
    dataset_dict = {}
    filename_queue = Queue()
    dataset_filenames = []

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".pdb"):
            dataset_dict[filename] = idx
            filename_queue.put(filename)
            dataset_filenames.append(filename)
            idx += 1

        if limit is not None and idx >= limit:
            break

    dataset = [None] * len(dataset_filenames)

    pool = Pool(processes=Constants.NUM_THREADS)
    start_time = time.time()
    result = pool.map(create_graph_process, map(lambda df: (df, directory_path), dataset_filenames))
    for filename, graph in result:
        dataset[dataset_dict[filename]] = graph

    print(f'Dataset created in {(time.time() - start_time):.1f}s')

    return dataset, dataset_filenames


def save_dataset(dataset, dataset_filenames, filename=None):
    if filename is None:
        filename = Constants.SAVED_GRAPHS_PATH_DEFAULT_FILE + '_' + str(len(dataset)) + Constants.GRAPH_EXTENSION
    save_graphs(filename, dataset)

    fn_no_extension = filename.split('.')[0]
    filename_df = fn_no_extension + '_filenames.json'
    with open(filename_df, 'w') as f:
        # store the data as binary data stream
        json.dump(dataset_filenames, f)

    filename_wti = fn_no_extension + '_word_to_ix'
    save_feat_word_to_ixs(filename_wti)


def load_dataset(filename=Constants.SAVED_GRAPHS_PATH_DEFAULT_FILE):
    dataset = load_graphs(filename)
    dataset = dataset[0]

    fn_no_extension = filename.split('.')[0]
    filename_df = fn_no_extension + '_filenames.json'
    with open(filename_df, 'r') as f:
        # read the data as binary data stream
        dataset_filenames = json.load(f)

    filename_wti = fn_no_extension + '_word_to_ix'
    load_feat_word_to_ixs(filename_wti)
    return dataset, dataset_filenames


def get_dataset(load_filename=None, directory_path=Constants.PDB_PATH, limit=None):
    if load_filename is None:
        load_filename = Constants.SAVED_GRAPHS_PATH_DEFAULT_FILE + '_' + str(limit) + Constants.GRAPH_EXTENSION
    try:
        start_time = time.time()
        dataset, dataset_filenames = load_dataset(load_filename)
        print(f'Dataset loaded in {(time.time() - start_time):.1f}s')
    except Exception as e:
        print(f'Load from file {load_filename} didn\'t succeed, now creating new dataset {e}')
        dataset, dataset_filenames = create_dataset(directory_path, limit)
    return dataset, dataset_filenames


def sort_split_dataset(data, data_filenames, pos_neg_ratio=0.14):
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
