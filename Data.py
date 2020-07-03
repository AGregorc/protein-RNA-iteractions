import json
import os
import warnings
import torch

from Bio.PDB import PDBParser
from dgl.data import save_graphs, load_graphs

import Constants
import Preprocess


def create_graph_sample(model_structure):
    # protein_atoms, atom_features, labels = Preprocess.get_atoms_features_and_labels(structure)
    protein_chains = Preprocess.label_protein_rna_interactions(model_structure)
    Preprocess.generate_node_features(model_structure)
    protein_atoms = Preprocess.get_atoms_list(protein_chains)

    pairs = Preprocess.find_pairs(protein_atoms)

    ##############################################################################
    atoms_with_edge = set()
    for a1, a2 in pairs:
        atoms_with_edge.add(a1)
        atoms_with_edge.add(a2)
    filtered_atoms = list(filter(lambda atom: atom in atoms_with_edge, protein_atoms))
    removed = len(protein_atoms) - len(filtered_atoms)
    if removed > 0:
        # protein_atoms = filtered_atoms
        num_atoms = len(filtered_atoms)
        percent = removed * 100 / (removed + num_atoms)
        print(f'Number of atoms without edge: {removed} ({percent:.1f}%)')
    ##############################################################################

    atom_features, labels = Preprocess.get_atoms_features_and_labels(protein_atoms)
    # if plot:
    #     plot_graph(pairs=pairs, atoms=protein_atoms, atom_color_func=Preprocess.get_labeled_color)

    G = Preprocess.create_dgl_graph(pairs, len(protein_atoms), set_edge_features=True, node_features=atom_features,
                                    labels=labels)
    assert G.number_of_nodes() == len(protein_atoms)

    #     return Sample(graph=G, atoms=protein_atoms, pairs=pairs, labels=labels)
    return G, protein_atoms, pairs, labels


def my_pdb_parser(filename, directory_path=Constants.PDB_PATH):
    parser = PDBParser()
    with warnings.catch_warnings(record=True):
        structure = parser.get_structure(os.path.splitext(filename)[0], os.path.join(directory_path, filename))
    model = structure[0]
    # print(structure)
    # chains = list(model.get_chains())
    return create_graph_sample(model)


def create_dataset(directory_path=Constants.PDB_PATH, limit=None):
    # directory = os.fsencode(PDB_PATH)
    directory = os.fsencode(directory_path)

    dataset = []
    dataset_filenames = []
    error_count = 0

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".pdb"):
            try:
                G, atoms, pairs, labels = my_pdb_parser(filename, directory_path)

            except Exception as e:
                error_count += 1
                print(f'Error during parsing file {filename}')
                continue
            print(f'File {filename} added')
            dataset.append(G)
            dataset_filenames.append(filename)
        if limit is not None and len(dataset) >= limit:
            break
    return dataset, dataset_filenames


def save_dataset(dataset, dataset_filenames, filename=Constants.SAVED_GRAPHS_PATH_DEFAULT_FILE):
    save_graphs(filename, dataset)

    fn_no_extension = filename.split('.')[0]
    filename_df = fn_no_extension + '_filenames.json'
    with open(filename_df, 'w') as f:
        # store the data as binary data stream
        json.dump(dataset_filenames, f)

    filename_wti = fn_no_extension + '_word_to_ix'
    Preprocess.save_feat_word_to_ixs(filename_wti)


def load_dataset(filename=Constants.SAVED_GRAPHS_PATH_DEFAULT_FILE):
    dataset = load_graphs(filename)
    dataset = dataset[0]

    fn_no_extension = filename.split('.')[0]
    filename_df = fn_no_extension + '_filenames.json'
    with open(filename_df, 'r') as f:
        # read the data as binary data stream
        dataset_filenames = json.load(f)

    filename_wti = fn_no_extension + '_word_to_ix'
    Preprocess.load_feat_word_to_ixs(filename_wti)
    return dataset, dataset_filenames


def get_dataset(load_filename=Constants.SAVED_GRAPHS_PATH_DEFAULT_FILE, directory_path=Constants.PDB_PATH, limit=None):
    try:
        dataset, dataset_filenames = load_dataset(load_filename)
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
