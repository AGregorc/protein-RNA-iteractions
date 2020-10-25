import json
from collections import deque

import dgl
import numpy as np
import torch
from Bio.PDB import PPBuilder, NeighborSearch, get_surface, is_aa, calc_angle, Vector, make_dssp_dict
from Bio.PDB.ResidueDepth import residue_depth

import Constants
from groups import a2gs


def create_graph_sample(model_structure, word_to_ixs, lock):
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

    atom_features, labels = get_atom_features_and_labels(protein_atoms)
    # if plot:
    #     plot_graph(pairs=pairs, atoms=protein_atoms, atom_color_func=get_labeled_color)

    G = create_dgl_graph(pairs, word_to_ixs, lock, len(protein_atoms), set_edge_features=True, node_features=atom_features, labels=labels)
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
        if not is_aa(residue, standard=True):
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


def min_dist(coord, surface):
    """Return minimum distance between coord and surface."""
    d = surface - coord
    d2 = np.sum(d * d, 1)
    idx = np.argmin(d2)
    return np.sqrt(d2[idx]), idx


def generate_node_features(protein_chains, surface, only_ca=Constants.GET_ONLY_CA_ATOMS):
    pdb_id = protein_chains[0].get_parent().full_id[0]
    dssp = make_dssp_dict(Constants.DSSP_PATH + pdb_id + '.dssp')
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

            key = res.full_id[2:]
            if key not in dssp[0]:
                key = (key[0], (' ', key[1][1], ' '))
                if key not in dssp[0]:
                    print('wtffffffffff, dssp key not found')
            dssp_features = dssp[0][key]

            is_cb = 'CB' in res
            cb_ca_surf_angle = 0
            ca_cb_surf_angle = 0

            ca_atom = res['CA']
            ca_d, ca_surf_idx = min_dist(ca_atom.get_coord(), surface)
            ca_vec = ca_atom.get_vector()
            if not is_cb:
                # print('there is no CB ..... :(((((((')
                pass
            else:
                cb_vec = res['CB'].get_vector()
                cb_d, cb_surf_idx = min_dist(res['CB'].get_coord(), surface)
                cb_ca_surf_angle = calc_angle(cb_vec, ca_vec, Vector(surface[ca_surf_idx]))
                ca_cb_surf_angle = calc_angle(ca_vec, cb_vec, Vector(surface[cb_surf_idx]))

            res_d = residue_depth(res, surface)
            if res_d is None:
                res_d = 5.0
                print("Nan values!!!")

            if ca_d is None:
                ca_d = 5.0
                print("Nan values!!!")

            for atom in res.get_atoms():
                if only_ca:
                    atom = ca_atom

                atom_d, s_idx = min_dist(atom.get_coord(), surface)
                d = atom.get_coord() - ca_atom.get_coord()
                ca_atom_dist = np.sqrt(np.sum(d*d))
                atom_ca_surf_angle = 0
                ca_atom_surf_angle = 0
                if not np.array_equal(atom.get_coord(), ca_atom.get_coord()):
                    atom_ca_surf_angle = calc_angle(atom.get_vector(), ca_vec, Vector(surface[s_idx]))
                    ca_atom_surf_angle = calc_angle(ca_vec, atom.get_vector(), Vector(surface[s_idx]))

                if atom_d is None:
                    atom_d = 5.0
                    print(f"Nan valuess!! {atom_d}, {atom}")
                setattr(atom, Constants.NODE_APPENDED_FEATURES['prev_res_name'], prev_res_name)
                setattr(atom, Constants.NODE_APPENDED_FEATURES['next_res_name'], next_res_name)
                setattr(atom, Constants.NODE_APPENDED_FEATURES['residue_depth'], res_d)
                setattr(atom, Constants.NODE_APPENDED_FEATURES['atom_depth'], atom_d)
                setattr(atom, Constants.NODE_APPENDED_FEATURES['ca_depth'], ca_d)
                setattr(atom, Constants.NODE_APPENDED_FEATURES['ca_atom_dist'], ca_atom_dist)
                setattr(atom, Constants.NODE_APPENDED_FEATURES['cb_ca_surf_angle'], cb_ca_surf_angle)
                setattr(atom, Constants.NODE_APPENDED_FEATURES['ca_cb_surf_angle'], ca_cb_surf_angle)
                setattr(atom, Constants.NODE_APPENDED_FEATURES['atom_ca_surf_angle'], atom_ca_surf_angle)
                setattr(atom, Constants.NODE_APPENDED_FEATURES['ca_atom_surf_angle'], ca_atom_surf_angle)
                setattr(atom, Constants.DSSP_FEATURES_NAME, dssp_features)

                if only_ca:
                    break
            last_n_residues.append(next(residue_generator, None))


def get_node_features(atom):
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

    dssp_list = getattr(atom, Constants.DSSP_FEATURES_NAME, [])
    for f in dssp_list:
        features.append(f)

    Constants.NODE_FEATURES_NUM = len(features)
    return features


def get_atom_features_and_labels(protein_atoms):
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

        features.append(get_node_features(atom))

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


# node_feat_word_to_ixs = {}


def save_feat_word_to_ixs(filename, node_feat_word_to_ixs):
    with open(filename + '.json', 'w') as fp:
        json.dump(node_feat_word_to_ixs.copy(), fp)


def load_feat_word_to_ixs(filename):
    word_to_ixs = {}
    with open(filename + '.json', 'r') as fp:
        word_to_ixs = {int(k): v for k, v in json.load(fp).items()}
    return word_to_ixs


# def get_feat_word_to_ixs():
#     return node_feat_word_to_ixs


def transform_node_features(features_list, node_feat_word_to_ixs, lock):
    """
        As we know from node_features function, node features contain also string elements.
        Here we transform string features to (one-hot) indexes that are suitable for Embedding layers.

    :param node_feat_word_to_ixs:
    :param lock:
    :param features_list: list of all node features of a graph
    :return: torch tensor transformed features
    """
    result = np.zeros((len(features_list), len(features_list[0])))

    # lock.acquire()
    dict_copy = [*node_feat_word_to_ixs.keys()]
    # lock.release()
    for col, feat in enumerate(features_list[0]):
        if isinstance(feat, str):
            if col not in dict_copy:
                # we have to find columns with strings then.
                dict_copy.append(col)  # add column to a copy list too
                # lock.acquire()
                node_feat_word_to_ixs[col] = {}  # init word to ix for each column with strings
                # lock.release()
        else:
            result[:, col] = [feat[col] for feat in features_list]

    for col in dict_copy:
        col = int(col)
        for j, feat in enumerate(features_list):
            word = feat[col]
            if lock:
                lock.acquire()
            # with lock:
            if word not in node_feat_word_to_ixs[col]:
                d = node_feat_word_to_ixs[col]
                d[word] = len(node_feat_word_to_ixs[col])
                node_feat_word_to_ixs[col] = d
            result[j, col] = node_feat_word_to_ixs[col][word]
            if lock:
                lock.release()

    return torch.from_numpy(result).to(dtype=torch.float32)


def create_dgl_graph(pairs, node_feat_word_to_ixs, lock, num_nodes, set_edge_features=False, node_features=None, labels=None):
    """
        Our main preprocess function.

    :param lock: multiprocessing lock
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
        G.ndata[Constants.NODE_FEATURES_NAME] = transform_node_features(node_features, node_feat_word_to_ixs, lock)

    if labels is not None:
        G.ndata[Constants.LABEL_NODE_NAME] = labels
    return G
