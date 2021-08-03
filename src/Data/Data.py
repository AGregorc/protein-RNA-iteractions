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


def my_pdb_parser(filename, directory_path=Constants.PDB_PATH, word_to_ixs=None, lock=None, standardize=None,
                  save=False):
    if word_to_ixs is None:
        word_to_ixs = {}
    parser = PDBParser()
    with warnings.catch_warnings(record=True):
        with Constants.open_data_file(directory_path, filename) as f:
            structure = parser.get_structure(Constants.filename_to_pdb_id(filename), f)
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
    my_filename = Constants.to_pdb_filename(Constants.filename_to_pdb_id(my_filename))

    start_time = time.time()
    try:
        # print(f'[{os.getpid()}] got something to work :O')
        graph, atoms, pairs, labels = my_pdb_parser(my_filename, directory_path, word_to_ixs, lock, save=save)
        print(f'[{os.getpid()}] File {my_filename} added in {(time.time() - start_time):.1f}s')

        if save:
            pdb_id = Constants.filename_to_pdb_id(my_filename)
            pf = os.path.join(Constants.SAVED_GRAPH_PATH, pdb_id + Constants.GRAPH_EXTENSION)
            save_graphs(pf, [graph])
    except Exception as e:
        print(f'[{os.getpid()}] Error from file {my_filename} {(time.time() - start_time):.1f}s; {e}')
        call_gc()
        return my_filename, None
    call_gc()
    return Constants.filename_to_pdb_id(my_filename), graph


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


def update_dataset(pdb_list=None, directory_path=Constants.PDB_PATH, limit=None, save_individual=True):
    manager = Manager()
    if pdb_list is None:
        directory = os.fsencode(directory_path)

        pdb_error_list = []
        with open(Constants.PDB_ERROR_LIST) as f:
            for pdb in f:
                pdb = pdb.strip()
                if len(pdb) > 0:
                    pdb_error_list.append(pdb)

        idx = 0
        # dataset_dict = {}
        dataset_pdb_ids = []

        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if Constants.is_pdb(filename):
                # dataset_dict[filename] = idx
                if save_individual:
                    pdb_id = Constants.filename_to_pdb_id(filename)
                    if (not os.path.isfile(os.path.join(Constants.SAVED_GRAPH_PATH, pdb_id + Constants.GRAPH_EXTENSION))
                            and filename not in pdb_error_list):
                        dataset_pdb_ids.append(filename)
                        idx += 1
                else:
                    dataset_pdb_ids.append(filename)
                    idx += 1
            if limit is not None and idx >= limit:
                break
    else:
        dataset_pdb_ids = pdb_list

    if save_individual:
        wti = load_feat_word_to_ixs(Constants.GENERAL_WORD_TO_IDX_PATH)
        word_to_ixs = manager.dict(wti)
    else:
        word_to_ixs = manager.dict()
    lock = manager.Lock()

    start_time = time.time()
    print(f'Starting to create {len(dataset_pdb_ids)} graphs with multiprocessing')
    with Pool(processes=Constants.NUM_PROCESSES) as pool:
        result = pool.map(create_graph_process,
                          map(lambda df: (df, directory_path, word_to_ixs, lock, save_individual), dataset_pdb_ids))
    call_gc()

    intermediate_time = time.time()
    print(f'Create graphs finished in {intermediate_time - start_time:.1f}s')

    pool.close()

    if save_individual:
        save_feat_word_to_ixs(Constants.GENERAL_WORD_TO_IDX_PATH, word_to_ixs)
        new_pdb_errors = []

        for pdb_id, graph in result:
            if graph is None:
                new_pdb_errors.append(pdb_id)
        print(f'Number of errors {len(new_pdb_errors)}')
        with open(Constants.PDB_ERROR_LIST, 'a') as f:
            for pdb in new_pdb_errors:
                f.write(pdb + '\n')
        return
    else:
        dataset = []
        dataset_pdb_ids = []
        for pdb_id, graph in result:
            if graph is None:
                print(f'{pdb_id} is None')
                continue
            dataset.append(graph)
            dataset_pdb_ids.append(pdb_id)

        print(f'Dataset created in {(time.time() - start_time):.1f}s')

        return dataset, dataset_pdb_ids, word_to_ixs, (None, None)


def save_dataset(dataset, dataset_pdb_ids, word_to_ixs, mean, std, filename=None, limit=None, individual=True):
    if filename is None:
        filename = file_name(limit=limit)

    if individual:
        for graph, fn in zip(dataset, dataset_pdb_ids):
            pdb_id = Constants.filename_to_pdb_id(fn)
            pf = os.path.join(Constants.SAVED_GRAPH_PATH, pdb_id + Constants.GRAPH_EXTENSION)
            save_graphs(pf, [graph])
        filename_wti = Constants.GENERAL_WORD_TO_IDX_PATH
    else:
        save_graphs(filename, dataset)

        fn_no_extension = os.path.splitext(filename)[0]
        filename_df = fn_no_extension + '_filenames.json'
        with open(filename_df, 'w') as f:
            # store the data as binary data stream
            json.dump(dataset_pdb_ids, f)

        filename_standardize = fn_no_extension + '_standardize.npy'
        with open(filename_standardize, 'wb') as f:
            torch.save(mean, f)
            torch.save(std, f)

        filename_wti = fn_no_extension + '_word_to_ix'
    save_feat_word_to_ixs(filename_wti, word_to_ixs)


def load_individuals(pdbs):
    filenames = listdir(Constants.SAVED_GRAPH_PATH)
    dataset = []
    dataset_pdb_ids = []
    for fn in filenames:
        pdb_id = Constants.filename_to_pdb_id(fn)
        if pdb_id in pdbs:
            graph = load_graphs(os.path.join(Constants.SAVED_GRAPH_PATH, fn))
            graph = graph[0][0]
            dataset.append(graph)
            dataset_pdb_ids.append(pdb_id)

    word_to_ixs = load_feat_word_to_ixs(Constants.GENERAL_WORD_TO_IDX_PATH)
    return dataset, dataset_pdb_ids, word_to_ixs


def load_dataset(filename=None, limit=None, individual=True):
    if filename is None:
        filename = file_name(limit=limit)

    fn_no_extension = os.path.splitext(filename)[0]
    if individual:
        filenames = listdir(Constants.SAVED_GRAPH_PATH)
        i = 0
        dataset = []
        dataset_pdb_ids = []
        for fn in filenames:
            graph = load_graphs(os.path.join(Constants.SAVED_GRAPH_PATH, fn))
            graph = graph[0][0]
            dataset.append(graph)
            dataset_pdb_ids.append(Constants.filename_to_pdb_id(fn))
            i += 1
            if limit is not None and i >= limit:
                break
        mean = None
        std = None
    else:
        dataset = load_graphs(filename)
        dataset = dataset[0]

        filename_df = fn_no_extension + '_filenames.json'
        with open(filename_df, 'r') as f:
            # read the data as binary data stream
            dataset_pdb_ids = json.load(f)

        filename_standardize = fn_no_extension + '_standardize.npy'
        with open(filename_standardize, 'rb') as f:
            mean = torch.load(f)
            std = torch.load(f)

    if individual:
        word_to_ixs = load_feat_word_to_ixs(Constants.GENERAL_WORD_TO_IDX_PATH)
    else:
        filename_wti = fn_no_extension + '_word_to_ix'
        word_to_ixs = load_feat_word_to_ixs(filename_wti)
    return dataset, dataset_pdb_ids, word_to_ixs, (mean, std)


def get_dataset(load_filename=None, directory_path=Constants.PDB_PATH, limit=None, individual=True):
    if load_filename is None:
        load_filename = file_name(limit=limit)
    try:
        start_time = time.time()
        dataset, dataset_pdb_ids, word_to_ixs, norm = load_dataset(load_filename, limit=limit, individual=individual)
        print(f'Dataset loaded in {(time.time() - start_time):.1f}s')
    except Exception as e:
        print(f'Load from file {load_filename} didn\'t succeed, now creating new dataset {e}')
        dataset, dataset_pdb_ids, word_to_ixs, norm = update_dataset(directory_path, limit)
    return dataset, dataset_pdb_ids, word_to_ixs, norm


def file_name(path_with_default_name=Constants.SAVED_GRAPHS_PATH_DEFAULT_FILE, limit=None,
              only_res=Constants.GET_ONLY_CA_ATOMS):
    return path_with_default_name + '_' + str(limit) + '_' + (
        'res_only' if only_res else 'all_atoms') + Constants.GRAPH_EXTENSION


# Deprecated
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
