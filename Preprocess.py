import json
import math
from collections import deque

import dgl
import torch
from Bio.PDB import PPBuilder, NeighborSearch, Chain, get_surface, is_aa
from Bio.PDB.ResidueDepth import residue_depth, ca_depth, min_dist

import Constants
import numpy as np

from groups import a2gs


def create_graph_sample(model_structure):
    # protein_atoms, atom_features, labels = get_atoms_features_and_labels(structure)
    protein_chains = label_protein_rna_interactions(model_structure)
    surface = get_surface(protein_chains)
    protein_atoms = get_atoms_list(protein_chains)
    generate_node_features(protein_chains, surface)

    pairs = find_pairs(protein_atoms)

    ##############################################################################
    # atoms_with_edge = set()
    # for a1, a2 in pairs:
    #     atoms_with_edge.add(a1)
    #     atoms_with_edge.add(a2)
    # filtered_atoms = list(filter(lambda atom: atom in atoms_with_edge, protein_atoms))
    # removed = len(protein_atoms) - len(filtered_atoms)
    # if removed > 0:
    #     # protein_atoms = filtered_atoms
    #     num_atoms = len(filtered_atoms)
    #     percent = removed * 100 / (removed + num_atoms)
    #     print(f'Number of atoms without edge: {removed} ({percent:.1f}%)')
    ##############################################################################

    atom_features, labels = get_atoms_features_and_labels(protein_atoms, surface)
    # if plot:
    #     plot_graph(pairs=pairs, atoms=protein_atoms, atom_color_func=get_labeled_color)

    G = create_dgl_graph(pairs, len(protein_atoms), set_edge_features=True, node_features=atom_features,
                         labels=labels)
    assert G.number_of_nodes() == len(protein_atoms)

    #     return Sample(graph=G, atoms=protein_atoms, pairs=pairs, labels=labels)
    return G, protein_atoms, pairs, labels


def find_pairs(atoms, distance=1.7, level='A', do_print=False):
    """
        Find pairs of atoms/residues/chains that are closer than specified distance.
        Pairs could represent edges in a graph of atoms.

    :param atoms: list of atoms of a structure
    :param distance: min distance to pair two atoms
    :param level: which entity pairs to return, atoms/residues or chains (A, R, C)
    :param do_print: boolean
    :return: pairs
    """
    ns = NeighborSearch(atoms)
    pairs = ns.search_all(distance, level=level)
    if do_print:
        print('Number of pairs:', len(pairs))

    # if do_plot:
    #     plot_graph(pairs)
    return pairs


def get_atoms_list(protein_chains, only_ca=Constants.GET_ONLY_CA_ATOMS):
    """

    :param protein_chains: list of Bio structures or a Bio structure
    :param only_ca: if true it returns just CA atoms
    :return: list of all atoms inside structure_list
    """
    atoms = []
    for chain in protein_chains:
        chain = filter_chain(chain)
        if only_ca:
            for residue in chain:
                atoms.append(residue['CA'])
        else:
            atoms = atoms + list(chain.get_atoms())
    return atoms


def filter_chain(chain):
    non_aas = []
    for idx, residue in enumerate(chain):
        if not is_aa(residue):
            non_aas.append(residue.id)
    for non_aa in non_aas:
        chain.__delitem__(non_aa)
    return chain


def is_protein(chain):
    """
        Check if chain is a protein.

    :param chain:
    :return:
    """
    ppb = PPBuilder()
    for pp in ppb.build_peptides(chain):
        if len(pp.get_sequence()) > 0:
            return True
    return False


def get_protein_chains(structure):
    """
        Find protein chains

    :param structure: Bio structure
    :return: list of protein chains
    """
    protein_chains = []
    for chain in structure.get_chains():
        if is_protein(chain):
            protein_chains.append(chain)
    #     print(protein_chains)
    return protein_chains


def label_protein_rna_interactions(structure):
    """
        Find all protein-RNA atom interactions.
        To do so, we find all pairs where one atom is from protein molecule and the other is from RNA.
        The min pairing distance is defined in Constants.LABEL_ATOM_DISTANCE.
        If the atom is in interaction we set a new attribute (Constants.LABEL_ATTRIBUTE_NAME) and assign it as True.

    :param structure: Bio structure
    :return: list of protein chains
    """
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
    """
        Check if atom is labeled as positive (in interaction).
        To see when atom is in interaction, have a look at label_protein_rna_interactions function.

    :param atom: Bio atom
    :return: boolean
    """
    return hasattr(atom, Constants.LABEL_ATTRIBUTE_NAME) and (getattr(atom, Constants.LABEL_ATTRIBUTE_NAME) is True)


def get_labeled_color(atom):
    """
        Function made for PlotMPL functions

    :param atom: Bio atom
    :return: labeled color
    """
    if is_labeled_positive(atom):
        return Constants.LABEL_POSITIVE_COLOR
    return Constants.LABEL_NEGATIVE_COLOR


def generate_node_features(protein_chains, surface, only_ca=Constants.GET_ONLY_CA_ATOMS):
    for chain in protein_chains:
        residue_generator = chain.get_residues()

        last_n_residues = deque([None, next(residue_generator), next(residue_generator, None)])
        while last_n_residues[1] is not None:
            prev_res = last_n_residues.popleft()
            prev_res_name = Constants.EMPTY_STR_FEATURE
            if prev_res is not None:
                prev_res_name = prev_res.resname
            res = last_n_residues[0]

            next_res = last_n_residues[1]
            next_res_name = Constants.EMPTY_STR_FEATURE
            if next_res is not None:
                next_res_name = next_res.resname

            res_d = residue_depth(res, surface)
            ca_d = ca_depth(res, surface)
            if res_d is None:
                res_d = 5.0
                print("Nan values!!!")

            if ca_d is None:
                ca_d = 5.0
                print("Nan values!!!")

            for atom in res.get_atoms():
                if only_ca:
                    atom = res['CA']

                atom_d = min_dist(atom.get_coord(), surface)
                if atom_d is None:
                    atom_d = 5.0
                    print(f"Nan valuess!! {atom_d}, {atom}")
                setattr(atom, Constants.NODE_APPENDED_FEATURES['prev_res_name'], prev_res_name)
                setattr(atom, Constants.NODE_APPENDED_FEATURES['next_res_name'], next_res_name)
                setattr(atom, Constants.NODE_APPENDED_FEATURES['residue_depth'], res_d)
                setattr(atom, Constants.NODE_APPENDED_FEATURES['ca_depth'], ca_d)
                setattr(atom, Constants.NODE_APPENDED_FEATURES['atom_depth'], atom_d)

                if only_ca:
                    break
            last_n_residues.append(next(residue_generator, None))


def node_features(atom):
    """
        Assign features to a atom.
        All atoms must have the same number of features and the same order.

    :param atom: Bio atom
    :return: list of features
    """
    # global NODE_FEATURES_NUM
    aa = atom.get_parent().resname.upper()

    features = [
        atom.mass,
        # atom.bfactor,
        atom.occupancy,
        atom.element,  # string
        atom.fullname,  # string
        aa,  # string
        # getattr(atom, Constants.NODE_APPENDED_FEATURES['prev_res_name'], Constants.EMPTY_STR_FEATURE), # string
        # getattr(atom, Constants.NODE_APPENDED_FEATURES['next_res_name'], Constants.EMPTY_STR_FEATURE)  # string
    ]

    for feature_name in Constants.NODE_APPENDED_FEATURES:
        features.append(getattr(atom, Constants.NODE_APPENDED_FEATURES[feature_name]))

    for group_feature in Constants.NODE_GROUP_FEATURES:
        if group_feature in a2gs[aa]:
            features.append(1)
        else:
            features.append(0)

    Constants.NODE_FEATURES_NUM = len(features)
    return features


def get_atoms_features_and_labels(protein_atoms, surface):
    """
        Create node features and label them.
        Since node features have numerical and string types we return list of features.
        The labels are returned as torch tensor.

        IMPORTANT: Here we also set a unique dgl id to each atom!!

    :param protein_atoms: list of Bio atoms
    :return: features, labels
    """
    features = []
    labels = np.zeros(len(protein_atoms))
    for idx, atom in enumerate(protein_atoms):

        setattr(atom, Constants.ATOM_DGL_ID, idx)

        features.append(node_features(atom))

        label = Constants.LABEL_NEGATIVE
        if is_labeled_positive(atom):
            label = Constants.LABEL_POSITIVE
        labels[idx] = label

    #     print(sum(labels), len(labels), sum(labels) / len(labels))
    return features, torch.from_numpy(labels).to(dtype=torch.int64)


def get_dgl_id(atom):
    """
        Each atom must have unique dgl (graph) id.
        It is stored as attribute in Bio atom class

    :param atom: Bio atom
    :return: dgl_id
    """
    return getattr(atom, Constants.ATOM_DGL_ID)


def get_edge_features(atom1, atom2):
    """
        Assign features to a edge.
        All atoms must have the same number of features and the same order.
        Atom1 and atom2 are here interpreted as pairs (with edge).

    :param atom1: Bio atom
    :param atom2: Bio atom
    :return: edge features
    """
    result = np.zeros(Constants.EDGE_FEATURE_NUM)
    # TODO: mera podobnosti, ki je invariantna na translacijo in rotacijo proteina
    # Predlog: razdalja do centra proteina/ogljika alpha oz podobno
    vec = atom2.get_coord() - atom1.get_coord()
    norm = np.linalg.norm(vec)
    result[:3] = vec / norm
    result[3] = norm
    Constants.EDGE_FEATURE_NUM = len(result)
    return result


def change_direction_features(np_array):
    """
        This function made for edge features since it contains vector that points from atom1 to atom2.
        Here we flip the direction to point from atom2 to atom1.

    :param np_array: edge features
    :return: edge features
    """
    result = -np_array
    result[:, 3] = np.abs(result[:, 3])
    return result


node_feat_word_to_ixs = {}
node_feat_wti_lens = {}


def save_feat_word_to_ixs(filename):
    with open(filename + '.json', 'w') as fp:
        json.dump(node_feat_word_to_ixs, fp)


def load_feat_word_to_ixs(filename):
    global node_feat_word_to_ixs, node_feat_wti_lens
    with open(filename + '.json', 'r') as fp:
        node_feat_word_to_ixs = {int(k): v for k, v in json.load(fp).items()}

    node_feat_wti_lens = {}
    for col in node_feat_word_to_ixs:
        col = int(col)
        node_feat_wti_lens[col] = len(node_feat_word_to_ixs[col])


def get_feat_word_to_ixs():
    return node_feat_word_to_ixs


def get_feat_wti_lens():
    return node_feat_wti_lens


def transform_node_features(features_list):
    """
        As we know from node_features function, node features contain also string elements.
        Here we transform string features to (one-hot) indexes that are suitable for Embedding layers.

    :param features_list: list of all node features of a graph
    :return: torch tensor transformed features
    """
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
        col = int(col)
        for j, feat in enumerate(features_list):
            word = feat[col]
            if word not in node_feat_word_to_ixs[col]:
                node_feat_word_to_ixs[col][word] = node_feat_wti_lens[col]
                node_feat_wti_lens[col] += 1
            result[j, col] = node_feat_word_to_ixs[col][word]

    return torch.from_numpy(result).to(dtype=torch.float32)


def create_dgl_graph(pairs, num_nodes, set_edge_features=False, node_features=None, labels=None):
    """
        Our main preprocess function.

    :param pairs: pairs of atoms
    :param num_nodes: sum of all atoms in a graph
    :param set_edge_features: boolean
    :param node_features: features from get_atoms_features_and_labels function
    :param labels: labels from get_atoms_features_and_labels function
    :return: dgl graph
    """
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

    # u = np.concatenate([src, dst])
    # v = np.concatenate([dst, src])
    #
    # # Construct a DGLGraph
    # G = dgl.DGLGraph((u, v))
    G = dgl.DGLGraph()
    G.add_nodes(num_nodes)
    G.add_edges(src, dst)
    G.add_edges(dst, src)

    if set_edge_features:
        edge_features = np.concatenate((edge_features, change_direction_features(edge_features)), axis=0)
        G.edata[Constants.EDGE_FEATURE_NAME] = torch.from_numpy(edge_features).to(dtype=torch.float32)

    if node_features:
        G.ndata[Constants.NODE_FEATURES_NAME] = transform_node_features(node_features)

    if labels is not None:
        G.ndata[Constants.LABEL_NODE_NAME] = labels
    return G
