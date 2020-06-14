import dgl
import torch
from Bio.PDB import PPBuilder, NeighborSearch

import Constants
import numpy as np

from PlotMPL import plot_graph


def find_pairs(atoms, distance=1.7, level='A', do_plot=False, do_print=False):
    ns = NeighborSearch(atoms)
    pairs = ns.search_all(distance, level=level)
    if do_print:
        print('Number of pairs:', len(pairs))

    if do_plot:
        plot_graph(pairs)
    return pairs


def get_atoms_list(structure_list):
    atoms = []
    if isinstance(structure_list, list):
        for structure in structure_list:
            atoms = atoms + list(structure.get_atoms())
    else:
        atoms = list(structure_list.get_atoms())
    return atoms


def is_protein(chain):
    ppb = PPBuilder()
    for pp in ppb.build_peptides(chain):
        if len(pp.get_sequence()) > 0:
            return True
    return False


def get_protein_chains(structure):
    protein_chains = []
    for chain in structure.get_chains():
        if is_protein(chain):
            protein_chains.append(chain)
    #     print(protein_chains)
    return protein_chains


def label_protein_rna_interactions(structure):
    protein_chains = get_protein_chains(structure)

    pairs = find_pairs(list(structure.get_atoms()), distance=Constants.LABEL_ATOM_DISTANCE)
    for pair in pairs:
        a1, a2 = pair
        c1 = a1.get_parent().get_parent()
        c2 = a2.get_parent().get_parent()
        if (c1 in protein_chains) != (c2 in protein_chains):
            setattr(a1, Constants.LABEL_ATTRIBUTE_NAME, True)
            setattr(a2, Constants.LABEL_ATTRIBUTE_NAME, True)
    return protein_chains


def is_labeled_positive(atom):
    return hasattr(atom, Constants.LABEL_ATTRIBUTE_NAME) and (getattr(atom, Constants.LABEL_ATTRIBUTE_NAME) is True)


def get_labeled_color(atom):
    if is_labeled_positive(atom):
        return Constants.LABEL_POSITIVE_COLOR
    return Constants.LABEL_NEGATIVE_COLOR


def node_features(atom):  # All nodes must have the same features order
    # global NODE_FEATURES_NUM

    features = [
        atom.mass,
        atom.bfactor,
        atom.occupancy,
        atom.element,
        atom.fullname,
        atom.get_parent().resname
    ]

    # strings:
    Constants.NODE_FEATURES_NUM = len(features)
    return features


def get_atoms_features_and_labels(structure):
    protein_chains = label_protein_rna_interactions(structure)
    protein_atoms = get_atoms_list(protein_chains)

    features = []
    labels = np.zeros(len(protein_atoms))
    for idx, atom in enumerate(protein_atoms):

        setattr(atom, Constants.ATOM_DGL_ID, idx)

        features.append(node_features(atom))

        label = 0
        if is_labeled_positive(atom):
            label = 1
        labels[idx] = label

    #     print(sum(labels), len(labels), sum(labels) / len(labels))
    return protein_atoms, features, torch.from_numpy(labels).to(dtype=torch.int64)


def get_dgl_id(atom):
    return getattr(atom, Constants.ATOM_DGL_ID)


def get_edge_features(a, b):  # TODO: EDGE_FEATURE_NUM is very scary :/

    result = np.zeros(Constants.EDGE_FEATURE_NUM)  # TODO: mera podobnosti, ki je invariantna na translacijo in rotacijo proteina
    # Predlog: razdalja do centra proteina/ogljika alpha oz podobno
    vec = b.get_coord() - a.get_coord()
    norm = np.linalg.norm(vec)
    result[:3] = vec / norm
    result[3] = norm
    Constants.EDGE_FEATURE_NUM = len(result)
    return result


def change_direction_features(np_array):
    result = -np_array
    result[:, 3] = np.abs(result[:, 3])
    return result


node_feat_word_to_ixs = {}
node_feat_wti_lens = {}


def transform_node_features(features_list):
    global node_feat_word_to_ixs
    result = np.zeros((len(features_list), len(features_list[0])))

    for col, feat in enumerate(features_list[0]):
        if isinstance(feat, str):
            if col not in node_feat_word_to_ixs:
                # we have to find columns with strings then.
                node_feat_word_to_ixs[col] = {}  # init word to ix for each column with strings
                node_feat_wti_lens[col] = 0
        else:
            result[:, col] = [feat[col] for feat in features_list]

    for col in node_feat_word_to_ixs.keys():
        for j, feat in enumerate(features_list):
            word = feat[col]
            if word not in node_feat_word_to_ixs[col]:
                node_feat_word_to_ixs[col][word] = node_feat_wti_lens[col]
                node_feat_wti_lens[col] += 1
            result[j, col] = node_feat_word_to_ixs[col][word]

    return torch.from_numpy(result).to(dtype=torch.float32)


def create_dgl_graph(pairs, set_edge_features=False, node_features=None, labels=None):
    src = []
    dst = []
    edge_features = None
    if set_edge_features:
        edge_features = np.zeros((len(pairs), Constants.EDGE_FEATURE_NUM))
    for idx, (a, b) in enumerate(pairs):
        src.append(get_dgl_id(a))
        dst.append(get_dgl_id(b))
        if set_edge_features:
            edge_features[idx] = get_edge_features(a, b)

    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])

    # Construct a DGLGraph
    G = dgl.DGLGraph((u, v))

    if set_edge_features:
        edge_features = np.concatenate((edge_features, change_direction_features(edge_features)), axis=0)
        G.edata[Constants.EDGE_FEATURE_NAME] = torch.from_numpy(edge_features).to(dtype=torch.float32)

    if node_features:
        G.ndata[Constants.NODE_FEATURES_NAME] = transform_node_features(node_features)

    if labels is not None:
        G.ndata[Constants.LABEL_NODE_NAME] = labels
    return G
