import gc
import json
import os
import time
import warnings
from multiprocessing import Pool, Manager
from os import listdir

import torch
from Bio.PDB import PDBParser
from dgl.data import save_graphs, load_graphs

import Constants
from Data.Preprocess import create_graph_sample, save_feat_word_to_ixs, load_feat_word_to_ixs

from sys import platform

if platform == "linux" or platform == "linux2":
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))


def my_pdb_parser(filename, directory_path=Constants.PDB_PATH, word_to_ixs=None, lock=None, standardize=None, save=False):
    if word_to_ixs is None:
        word_to_ixs = {}
    parser = PDBParser()
    with warnings.catch_warnings(record=True):
        with open(os.path.join(directory_path, filename)) as f:
            structure = parser.get_structure(os.path.splitext(filename)[0], f)
    model = structure[0]
    # print(structure)
    # chains = list(model.get_chains())
    graph, atoms, pairs, labels = create_graph_sample(model, word_to_ixs, lock, save)
    if standardize:
        numerical_cols = [i for i in range(Constants.NODE_FEATURES_NUM) if i not in word_to_ixs.keys()]
        filename, graph = standardize_graph_process((filename, graph, standardize[0], standardize[1], numerical_cols))

    del parser, structure, model
    return graph, atoms, pairs, labels


def call_gc():
    garbage = gc.collect()
    # print(f'Garbage removed: {garbage}')


def create_graph_process(args):
    my_filename, directory_path, word_to_ixs, lock, save = args
    start_time = time.time()
    try:
        # print(f'[{os.getpid()}] got something to work :O')
        graph, atoms, pairs, labels = my_pdb_parser(my_filename, directory_path, word_to_ixs, lock, save=save)
        print(f'[{os.getpid()}] File {my_filename} added in {(time.time() - start_time):.1f}s')

        if save:
            fn = os.path.splitext(my_filename)[0]
            pf = os.path.join(Constants.SAVED_GRAPH_PATH, fn + Constants.GRAPH_EXTENSION)
            save_graphs(pf, [graph])
    except Exception as e:
        print(f'[{os.getpid()}] Error from file {my_filename} {(time.time() - start_time):.1f}s; {e}')
        call_gc()
        return my_filename, None
    call_gc()
    return my_filename, graph


def standardize_graph_process(result):
    filename, graph, mean, std, numerical_cols = result
    if graph is None:
        return filename, None
    # numerical_cols = [i for i in range(Constants.NODE_FEATURES_NUM) if i not in get_feat_word_to_ixs().keys()]

    graph.ndata[Constants.NODE_FEATURES_NAME][:, numerical_cols] -= mean
    graph.ndata[Constants.NODE_FEATURES_NAME][:, numerical_cols] /= std

    # set all NaNs to 0
    t = graph.ndata[Constants.NODE_FEATURES_NAME]
    graph.ndata[Constants.NODE_FEATURES_NAME][torch.isnan(t)] = 0.0
    graph.ndata[Constants.NODE_FEATURES_NAME][torch.isinf(t)] = 0.0

    return filename, graph


def update_dataset(directory_path=Constants.PDB_PATH, limit=None, save_individual=True):
    directory = os.fsencode(directory_path)
    manager = Manager()

    idx = 0
    # dataset_dict = {}
    dataset_filenames = []

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".pdb"):
            # dataset_dict[filename] = idx
            if save_individual:
                fn = os.path.splitext(filename)[0]
                if not os.path.isfile(os.path.join(Constants.SAVED_GRAPH_PATH, fn + Constants.GRAPH_EXTENSION)):
                    dataset_filenames.append(filename)
                    idx += 1
            else:
                dataset_filenames.append(filename)
                idx += 1
        if limit is not None and idx >= limit:
            break

    if save_individual:
        wti = load_feat_word_to_ixs(Constants.GENERAL_WORD_TO_IDX_PATH)
        word_to_ixs = manager.dict(wti)
    else:
        word_to_ixs = manager.dict()
    lock = manager.Lock()

    start_time = time.time()
    print(f'Starting to create {len(dataset_filenames)} graphs with multiprocessing')
    with Pool(processes=Constants.NUM_PROCESSES) as pool:
        result = pool.map(create_graph_process,
                          map(lambda df: (df, directory_path, word_to_ixs, lock, save_individual), dataset_filenames))
    call_gc()

    intermediate_time = time.time()
    print(f'Create graphs finished in {intermediate_time - start_time:.1f}s')

    # numerical_cols = [i for i in range(Constants.NODE_FEATURES_NUM) if i not in word_to_ixs.keys()]
    # num_for_standardization = min(100, len(result))
    # means = torch.empty((num_for_standardization, len(numerical_cols)), dtype=torch.float64)
    # variances = torch.empty((num_for_standardization, len(numerical_cols)), dtype=torch.float64)
    # count_errors = 0
    # for i in range(num_for_standardization):
    #     num = i - count_errors
    #     if result[i][1] is None:
    #         count_errors += 1
    #         continue
    #     numerical_features = result[i][1].ndata[Constants.NODE_FEATURES_NAME][:, numerical_cols]
    #     m = torch.mean(numerical_features, dim=0)
    #     v = torch.var(numerical_features, dim=0)
    #     means[num] = m
    #     variances[num] = v
    # if count_errors > 0:
    #     means = means[:-count_errors]
    #     variances = variances[:-count_errors]
    # mean = torch.mean(means, dim=0, dtype=torch.float32)
    # std = torch.sqrt(torch.mean(variances, dim=0, dtype=torch.float32))
    #
    # print(f'Starting to standardize graphs with multiprocessing in {time.time() - intermediate_time:.1f}s')
    # intermediate_time = time.time()
    # with Pool(processes=Constants.NUM_PROCESSES) as pool:
    #     result = pool.map(standardize_graph_process,
    #                       map(lambda f_and_g: (f_and_g[0], f_and_g[1], mean, std, numerical_cols), result))
    # call_gc()
    # print(f'Filter unsuccessful graphs in {time.time() - intermediate_time:.1f}s')
    # Wait for all processes.
    pool.close()

    if save_individual:
        save_feat_word_to_ixs(Constants.GENERAL_WORD_TO_IDX_PATH, word_to_ixs)
        return
    else:
        dataset = []
        dataset_filenames = []
        for filename, graph in result:
            if graph is None:
                print(f'{filename} is None')
                continue
            dataset.append(graph)
            dataset_filenames.append(filename)

        print(f'Dataset created in {(time.time() - start_time):.1f}s')

        return dataset, dataset_filenames, word_to_ixs, (None, None)


def save_dataset(dataset, dataset_filenames, word_to_ixs, mean, std, filename=None, limit=None, individual=True):
    if filename is None:
        filename = file_name(limit=limit)

    if individual:
        for graph, fn in zip(dataset, dataset_filenames):
            fn = os.path.splitext(fn)[0]
            pf = os.path.join(Constants.SAVED_GRAPH_PATH, fn + Constants.GRAPH_EXTENSION)
            save_graphs(pf, [graph])
    else:
        save_graphs(filename, dataset)

    fn_no_extension = os.path.splitext(filename)[0]
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


def load_dataset(filename=None, limit=None, individual=True):
    if filename is None:
        filename = file_name(limit=limit)

    fn_no_extension = os.path.splitext(filename)[0]
    if individual:
        filenames = listdir(Constants.SAVED_GRAPH_PATH)
        i = 0
        dataset = []
        dataset_filenames = []
        for fn in filenames:
            graph = load_graphs(os.path.join(Constants.SAVED_GRAPH_PATH, fn))
            graph = graph[0][0]
            dataset.append(graph)
            dataset_filenames.append(os.path.splitext(fn)[0])
            i += 1
            if i >= limit:
                break
    else:
        dataset = load_graphs(filename)
        dataset = dataset[0]

        filename_df = fn_no_extension + '_filenames.json'
        with open(filename_df, 'r') as f:
            # read the data as binary data stream
            dataset_filenames = json.load(f)

    filename_standardize = fn_no_extension + '_standardize.npy'
    with open(filename_standardize, 'rb') as f:
        mean = torch.load(f)
        std = torch.load(f)

    if individual:
        word_to_ixs = load_feat_word_to_ixs(Constants.GENERAL_WORD_TO_IDX_PATH)
    else:
        filename_wti = fn_no_extension + '_word_to_ix'
        word_to_ixs = load_feat_word_to_ixs(filename_wti)
    return dataset, dataset_filenames, word_to_ixs, (mean, std)


def get_dataset(load_filename=None, directory_path=Constants.PDB_PATH, limit=None, individual=True):
    if load_filename is None:
        load_filename = file_name(limit=limit)
    try:
        start_time = time.time()
        dataset, dataset_filenames, word_to_ixs, norm = load_dataset(load_filename, limit=limit, individual=individual)
        print(f'Dataset loaded in {(time.time() - start_time):.1f}s')
    except Exception as e:
        print(f'Load from file {load_filename} didn\'t succeed, now creating new dataset {e}')
        dataset, dataset_filenames, word_to_ixs, norm = update_dataset(directory_path, limit)
    return dataset, dataset_filenames, word_to_ixs, norm


def file_name(path_with_default_name=Constants.SAVED_GRAPHS_PATH_DEFAULT_FILE, limit=None,
              only_res=Constants.GET_ONLY_CA_ATOMS):
    return path_with_default_name + '_' + str(limit) + '_' + (
        'res_only' if only_res else 'all_atoms') + Constants.GRAPH_EXTENSION


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
