import os
import warnings

from Bio.PDB import PDBParser

import Preprocess
from Constants import PDB_PATH
from PlotMPL import plot_graph


def create_graph_sample(structure, plot=False):
    protein_atoms, atom_features, labels = Preprocess.get_atoms_features_and_labels(structure)

    pairs = Preprocess.find_pairs(protein_atoms, do_plot=False)
    if plot:
        plot_graph(pairs=pairs, atoms=protein_atoms, atom_color_func=Preprocess.get_labeled_color)

    G = Preprocess.create_dgl_graph(pairs, set_edge_features=True, node_features=atom_features, labels=labels)
    assert G.number_of_nodes() == len(protein_atoms)

    #     return Sample(graph=G, atoms=protein_atoms, pairs=pairs, labels=labels)
    return G, protein_atoms, pairs, labels


def my_pdb_parser(filename, do_plot=False):
    parser = PDBParser()
    structure = parser.get_structure(os.path.splitext(filename)[0], os.path.join(PDB_PATH, filename))
    model = structure[0]
    # print(structure)
    # chains = list(model.get_chains())
    return create_graph_sample(model, plot=do_plot)


def create_dataset(directory_path, limit=None):
    # directory = os.fsencode(PDB_PATH)
    directory = os.fsencode(directory_path)

    dataset = []
    dataset_filenames = []
    error_count = 0

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".pdb"):
            with warnings.catch_warnings(record=True):
                try:
                    G, atoms, pairs, labels = my_pdb_parser(filename, do_plot=False)
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
