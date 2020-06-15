import os
import warnings

from Bio.PDB import PDBParser

import Preprocess
from Constants import PDB_PATH


def create_graph_sample(structure):
    # protein_atoms, atom_features, labels = Preprocess.get_atoms_features_and_labels(structure)
    protein_chains = Preprocess.label_protein_rna_interactions(structure)
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
        percent = removed*100/(removed+num_atoms)
        print(f'Number of atoms without edge: {removed} ({percent:.1f}%)')
    ##############################################################################

    atom_features, labels = Preprocess.get_atoms_features_and_labels(protein_atoms)
    # if plot:
    #     plot_graph(pairs=pairs, atoms=protein_atoms, atom_color_func=Preprocess.get_labeled_color)

    G = Preprocess.create_dgl_graph(pairs, len(protein_atoms), set_edge_features=True, node_features=atom_features, labels=labels)
    assert G.number_of_nodes() == len(protein_atoms)

    #     return Sample(graph=G, atoms=protein_atoms, pairs=pairs, labels=labels)
    return G, protein_atoms, pairs, labels


def my_pdb_parser(filename):
    parser = PDBParser()
    with warnings.catch_warnings(record=True):
        structure = parser.get_structure(os.path.splitext(filename)[0], os.path.join(PDB_PATH, filename))
    model = structure[0]
    # print(structure)
    # chains = list(model.get_chains())
    return create_graph_sample(model)


def create_dataset(directory_path, limit=None):
    # directory = os.fsencode(PDB_PATH)
    directory = os.fsencode(directory_path)

    dataset = []
    dataset_filenames = []
    error_count = 0

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".pdb"):
            try:
                G, atoms, pairs, labels = my_pdb_parser(filename)
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
