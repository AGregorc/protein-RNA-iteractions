import json
import os
import time
import warnings
from multiprocessing import Pool, Manager
from queue import Queue

import numpy as np
import torch
from Bio.PDB import PDBParser
from dgl.data import save_graphs, load_graphs

import Constants
from Preprocess import create_graph_sample, save_feat_word_to_ixs, load_feat_word_to_ixs


def my_pdb_parser(filename, directory_path=Constants.PDB_PATH, word_to_ixs=None, lock=None, standardize=None):
    if word_to_ixs is None:
        word_to_ixs = {}
    parser = PDBParser()
    with warnings.catch_warnings(record=True):
        structure = parser.get_structure(os.path.splitext(filename)[0], os.path.join(directory_path, filename))
    model = structure[0]
    # print(structure)
    # chains = list(model.get_chains())
    graph, atoms, pairs, labels = create_graph_sample(model, word_to_ixs, lock)
    if standardize:
        numerical_cols = [i for i in range(Constants.NODE_FEATURES_NUM) if i not in word_to_ixs.keys()]
        filename, graph = standardize_graph_process((filename, graph, standardize[0], standardize[1], numerical_cols))
    return graph, atoms, pairs, labels


def create_graph_process(args):
    my_filename, directory_path, word_to_ixs, lock = args
    # print(f'[{os.getpid()}] got something to work :O')
    start_time = time.time()
    graph, atoms, pairs, labels = my_pdb_parser(my_filename, directory_path, word_to_ixs, lock)
    print(f'[{os.getpid()}] File {my_filename} added in {(time.time() - start_time):.1f}s')
    return my_filename, graph


def standardize_graph_process(result):
    filename, graph, mean, std, numerical_cols = result
    # numerical_cols = [i for i in range(Constants.NODE_FEATURES_NUM) if i not in get_feat_word_to_ixs().keys()]

    graph.ndata[Constants.NODE_FEATURES_NAME][:, numerical_cols] -= mean
    graph.ndata[Constants.NODE_FEATURES_NAME][:, numerical_cols] /= std

    return filename, graph


def create_dataset(directory_path=Constants.PDB_PATH, limit=None):
    directory = os.fsencode(directory_path)
    manager = Manager()

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
    word_to_ixs = manager.dict()
    lock = manager.Lock()

    pool = Pool(processes=Constants.NUM_THREADS)
    start_time = time.time()
    result = pool.map(create_graph_process, map(lambda df: (df, directory_path, word_to_ixs, lock), dataset_filenames))

    numerical_cols = [i for i in range(Constants.NODE_FEATURES_NUM) if i not in word_to_ixs.keys()]
    num_for_standardization = min(100, len(result))
    means = torch.empty((num_for_standardization, len(numerical_cols)), dtype=torch.float64)
    variances = torch.empty((num_for_standardization, len(numerical_cols)), dtype=torch.float64)
    for i in range(num_for_standardization):
        numerical_features = result[i][1].ndata[Constants.NODE_FEATURES_NAME][:, numerical_cols]
        m = torch.mean(numerical_features, dim=0)
        v = torch.var(numerical_features, dim=0)
        means[i] = m
        variances[i] = v
    mean = torch.mean(means, dim=0)
    std = torch.sqrt(torch.mean(variances, dim=0))

    result = pool.map(standardize_graph_process, map(lambda f_and_g: (f_and_g[0], f_and_g[1], mean, std, numerical_cols), result))
    for filename, graph in result:
        dataset[dataset_dict[filename]] = graph

    print(f'Dataset created in {(time.time() - start_time):.1f}s')

    # Wait for all thread.
    pool.close()
    pool.join()
    return dataset, dataset_filenames, word_to_ixs, (mean, std)


def save_dataset(dataset, dataset_filenames, word_to_ixs, mean, std, filename=None):
    if filename is None:
        filename = Constants.SAVED_GRAPHS_PATH_DEFAULT_FILE + '_' + str(len(dataset)) + Constants.GRAPH_EXTENSION
    save_graphs(filename, dataset)

    fn_no_extension = filename.split('.')[0]
    filename_df = fn_no_extension + '_filenames.json'
    with open(filename_df, 'w') as f:
        # store the data as binary data stream
        json.dump(dataset_filenames, f)

    filename_standardize = fn_no_extension + '_standardize.npy'
    with open(filename_standardize, 'wb') as f:
        torch.save(mean, f)
        torch.save(std, f)

    filename_wti = fn_no_extension + '_word_to_ix'
    save_feat_word_to_ixs(filename_wti, word_to_ixs)


def load_dataset(filename=Constants.SAVED_GRAPHS_PATH_DEFAULT_FILE):
    dataset = load_graphs(filename)
    dataset = dataset[0]

    fn_no_extension = filename.split('.')[0]
    filename_df = fn_no_extension + '_filenames.json'
    with open(filename_df, 'r') as f:
        # read the data as binary data stream
        dataset_filenames = json.load(f)

    filename_standardize = fn_no_extension + '_standardize.npy'
    with open(filename_standardize, 'rb') as f:
        mean = torch.load(f)
        std = torch.load(f)

    filename_wti = fn_no_extension + '_word_to_ix'
    word_to_ixs = load_feat_word_to_ixs(filename_wti)
    return dataset, dataset_filenames, word_to_ixs, (mean, std)


def get_dataset(load_filename=None, directory_path=Constants.PDB_PATH, limit=None):
    if load_filename is None:
        load_filename = Constants.SAVED_GRAPHS_PATH_DEFAULT_FILE + '_' + str(limit) + Constants.GRAPH_EXTENSION
    try:
        start_time = time.time()
        dataset, dataset_filenames, word_to_ixs, norm = load_dataset(load_filename)
        print(f'Dataset loaded in {(time.time() - start_time):.1f}s')
    except Exception as e:
        print(f'Load from file {load_filename} didn\'t succeed, now creating new dataset {e}')
        dataset, dataset_filenames, word_to_ixs, norm = create_dataset(directory_path, limit)
    return dataset, dataset_filenames, word_to_ixs, norm


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
