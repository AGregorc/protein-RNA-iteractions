import json
import os
import time
from collections import deque
from io import StringIO

import dgl
import numpy as np
import torch
from Bio.PDB import PPBuilder, NeighborSearch, get_surface, is_aa, calc_angle, Vector, make_dssp_dict
from Bio.PDB.DSSP import _make_dssp_dict

import Constants
from Data.groups import a2gs


def create_graph_sample(model_structure, word_to_ixs, lock, save=False):
    # start = time.time()
    protein_chains = label_protein_rna_interactions(model_structure)
    # end = time.time()
    # print(f'label_protein_rna_interactions: {end - start}')
    # start = end
    surface = get_surface(protein_chains, MSMS='msms')
    # end = time.time()
    # print(f'get_surface: {end - start}')
    # start = end
    if len(surface) == 0:
        raise Exception(f'Len of surface for model {model_structure.full_id[0]} is 0')

    protein_atoms = get_atoms_list(protein_chains)
    # end = time.time()
    # print(f'get_atoms_list: {end - start}')
    # start = time.time()

    pairs, ns_object = find_pairs(protein_atoms)
    # end = time.time()
    # print(f'find_pairs: {end - start}')
    # start = end

    generate_node_features(protein_chains, surface, ns_object)
    # end = time.time()
    # print(f'generate_node_features: {end - start}')
    # start = end
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

    atom_features, labels, positions = get_atom_features_and_labels(protein_atoms)

    # end = time.time()
    # print(f'get_atom_features_and_labels: {end - start}')
    # start = end
    # if plot:
    #     plot_graph(pairs=pairs, atoms=protein_atoms, atom_color_func=get_labeled_color)

    G = create_dgl_graph(pairs, word_to_ixs, lock, len(protein_atoms), set_edge_features=False,
                         node_features=atom_features, labels=labels, coordinates=positions, save=save)
    assert G.number_of_nodes() == len(protein_atoms)

    # end = time.time()
    # print(f'create_dgl_graph: {end - start}')

    return G, protein_atoms, pairs, labels


def find_pairs(atoms, distance=Constants.ATOM_ATOM_DISTANCE, level='A', do_print=False):
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

    return pairs, ns


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


def label_protein_rna_interactions(structure, only_ca=Constants.GET_ONLY_CA_ATOMS):
    """
        Find all protein-RNA atom interactions.
        To do so, we find all pairs where one atom is from protein molecule and the other is from RNA.
        The min pairing distance is defined in Constants.LABEL_ATOM_DISTANCE.
        If the atom is in interaction we set a new attribute (Constants.LABEL_ATTRIBUTE_NAME) and assign it as True.

    :param only_ca:
    :param structure: Bio structure
    :return: list of protein chains
    """
    protein_chains = get_protein_chains(structure)

    pairs, _ = find_pairs(list(structure.get_atoms()), distance=Constants.LABEL_ATOM_DISTANCE)
    for pair in pairs:
        a1, a2 = pair
        c1 = a1.get_parent().get_parent()
        c2 = a2.get_parent().get_parent()
        if (c1 in protein_chains) != (c2 in protein_chains):
            if only_ca:
                res1 = a1.get_parent()
                if 'CA' in res1:
                    setattr(res1['CA'], Constants.LABEL_ATTRIBUTE_NAME, True)
                res2 = a2.get_parent()
                if 'CA' in res2:
                    setattr(res2['CA'], Constants.LABEL_ATTRIBUTE_NAME, True)
            else:
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


def residue_depth(residue, surface):
    dist_list = []
    distance = 0
    for atom in residue.get_atoms():
        coord = atom.get_coord()
        d, idx = min_dist(coord, surface)
        distance += d
        dist_list.append((d, idx))
    return distance / len(dist_list), dist_list


def num_of_atoms_above_plane(n, x, atoms):
    if len(atoms) == 0:
        return 0
    d = np.dot(n, x)
    st = np.sum(np.dot(np.array(list(map(lambda m: m.get_coord(), atoms))), n) > d)
    return st


def generate_node_features(protein_chains, surface, ns: NeighborSearch, only_ca=Constants.GET_ONLY_CA_ATOMS):
    pdb_id = protein_chains[0].get_parent().full_id[0]
    pdb_id = pdb_id[-4:]

    with Constants.open_data_file(Constants.DSSP_PATH,  pdb_id + '.dssp') as f:
        dssp_text = f.read()
    dssp = _make_dssp_dict(StringIO(dssp_text))
    get_residues_t = dssp_key_t = min_dist_t = residue_depth_t = atom_d_t = settattr_t = 0

    for chain in protein_chains:
        start = time.time()
        residue_generator = chain.get_residues()
        get_residues_t += time.time() - start

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

            start = time.time()
            is_key = True
            key = res.full_id[2:]
            if key not in dssp[0]:
                key = (key[0], (' ', key[1][1], ' '))
                if key not in dssp[0]:
                    for dssp_key in dssp[0]:
                        if dssp_key[0] == key[0] and dssp_key[1][1] == key[1][1]:
                            key = dssp_key
                            break

                    if key not in dssp[0]:
                        is_key = False
                        # raise Exception(f'DSSP key not found for {key}, model {res.full_id[0]}')
            if is_key:
                dssp_features = dssp[0][key]
            else:
                dssp_features = ('', '-', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            dssp_key_t += time.time() - start

            start = time.time()
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
            min_dist_t += time.time() - start

            start = time.time()
            res_d, dist_list = residue_depth(res, surface)
            if res_d is None:
                res_d = 5.0
                print("Nan values!!!")

            if ca_d is None:
                ca_d = 5.0
                print("Nan values!!!")
            residue_depth_t += time.time() - start

            for idx, atom in enumerate(res.get_atoms()):
                if only_ca:
                    atom = ca_atom

                start = time.time()
                atom_d, s_idx = dist_list[idx]
                atom_coord = atom.get_coord()
                ca_atom_coord = ca_atom.get_coord()

                d = atom_coord - ca_atom_coord
                ca_atom_dist = np.sqrt(np.sum(d * d))
                atom_ca_surf_angle = 0
                ca_atom_surf_angle = 0
                if not np.array_equal(atom_coord, ca_atom_coord):
                    atom_ca_surf_angle = calc_angle(atom.get_vector(), ca_vec, Vector(surface[s_idx]))
                    ca_atom_surf_angle = calc_angle(ca_vec, atom.get_vector(), Vector(surface[s_idx]))

                if atom_d is None:
                    atom_d = 5.0
                    print(f"Nan valuess!! {atom_d}, {atom}")
                atom_d_t += time.time() - start

                start = time.time()
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
                settattr_t += time.time() - start

                cumsum_main = 0
                cumsum_plane = 0

                cumsum_atom_main = [0]*len(Constants.NEIGHBOUR_SUM_RADIUS_ATOMS)
                cumsum_atom_plane = [0]*len(Constants.NEIGHBOUR_SUM_RADIUS_ATOMS)
                for num, radius in enumerate(Constants.NEIGHBOUR_SUM_RADIUS):
                    atoms = ns.search(atom_coord, radius)
                    setattr(atom,
                            Constants.NODE_APPENDED_FEATURES[Constants.neighbour_sum_radius_name(num)],
                            len(atoms) - cumsum_main)

                    num_above_plane = num_of_atoms_above_plane(surface[s_idx] - atom_coord, atom_coord, atoms)
                    setattr(atom,
                            Constants.NODE_APPENDED_FEATURES[Constants.neighbour_sum_above_plane_radius_name(num)],
                            num_above_plane - cumsum_plane)
                    cumsum_main += len(atoms)
                    cumsum_plane += num_above_plane

                    for i, atom_element in enumerate(Constants.NEIGHBOUR_SUM_RADIUS_ATOMS):
                        atoms_one_element = list(filter(lambda a: a.element.upper() == atom_element.upper(), atoms))
                        setattr(atom,
                                Constants.NODE_APPENDED_FEATURES[Constants.neighbour_sum_radius_name(num, atom_element)],
                                len(atoms_one_element) - cumsum_atom_main[i])

                        num_above_plane = num_of_atoms_above_plane(surface[s_idx] - atom_coord,
                                                                   atom_coord,
                                                                   atoms_one_element)
                        setattr(atom,
                                Constants.NODE_APPENDED_FEATURES[Constants.neighbour_sum_above_plane_radius_name(num, atom_element)],
                                num_above_plane - cumsum_atom_plane[i])
                        cumsum_atom_main[i] += len(atoms_one_element)
                        cumsum_atom_plane[i] += num_above_plane
                if only_ca:
                    break
            last_n_residues.append(next(residue_generator, None))

    # print(f'Times: get_residues_t: {get_residues_t:.2f}, dssp_key_t: {dssp_key_t:.2f}, min_dist_t: {min_dist_t:.2f}, '
    #       f'residue_depth_t: {residue_depth_t:.2f}, atom_d_t: {atom_d_t:.2f}, settattr_t: {settattr_t:.2f}')


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
        atom.occupancy,
        atom.element,  # string
        atom.fullname,  # string
        aa,  # string
    ]

    for feature_name in Constants.NODE_APPENDED_FEATURES:
        features.append(getattr(atom, Constants.NODE_APPENDED_FEATURES[feature_name]))

    for group_feature in Constants.NODE_GROUP_FEATURES:
        if group_feature in a2gs[aa]:
            features.append(1)
        else:
            features.append(0)

    dssp_list = getattr(atom, Constants.DSSP_FEATURES_NAME, [])
    for i, f in enumerate(dssp_list):
        if i == 5:
            continue
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
    positions = np.zeros((len(protein_atoms), 3))
    labels = np.zeros(len(protein_atoms))
    for idx, atom in enumerate(protein_atoms):

        setattr(atom, Constants.ATOM_DGL_ID, idx)

        features.append(get_node_features(atom))
        positions[idx, :] = atom.get_coord()

        label = Constants.LABEL_NEGATIVE
        if is_labeled_positive(atom):
            label = Constants.LABEL_POSITIVE
        labels[idx] = label

    #     print(sum(labels), len(labels), sum(labels) / len(labels))
    return features, torch.from_numpy(labels).to(dtype=torch.long), torch.from_numpy(positions)


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
    fn = filename + '.json'
    if os.path.exists(fn):
        with open(fn, 'r') as fp:
            word_to_ixs = {int(k): v for k, v in json.load(fp).items()}
    return word_to_ixs


# def get_feat_word_to_ixs():
#     return node_feat_word_to_ixs


def transform_node_features(features_list, node_feat_word_to_ixs, lock, save=False):
    """
        As we know from node_features function, node features contain also string elements.
        Here we transform string features to (one-hot) indexes that are suitable for Embedding layers.

    :param features_list: list of all node features of a graph
    :param node_feat_word_to_ixs:
    :param lock:
    :param save:
    :return: torch tensor transformed features
    """
    result = np.zeros((len(features_list), len(features_list[0])))
    start_f = time.time()
    change = False
    # lock.acquire()
    dict_copy = [*node_feat_word_to_ixs.keys()]
    # lock.release()
    for col, feat in enumerate(features_list[0]):
        if isinstance(feat, str):
            if col not in dict_copy:
                # we have to find columns with strings then.
                dict_copy.append(col)  # add column to a copy list too
                if lock:
                    lock.acquire()
                node_feat_word_to_ixs[col] = {'': 0}  # init word to ix for each column with strings
                # value '' is default value
                if lock:
                    lock.release()
                change = True
        else:
            result[:, col] = [feat[col] for feat in features_list]

    # end = time.time()
    # print(f'check cols: {end - start_f:.2f}')
    # new_dict_t = assert_result_t = 0

    for col in dict_copy:
        col = int(col)
        col_word_to_ixs = node_feat_word_to_ixs[col]
        for j, feat in enumerate(features_list):
            word = feat[col]
            # with lock:
            # start = time.time()
            if word not in col_word_to_ixs:
                if save:
                    if lock:
                        lock.acquire()
                    d = col_word_to_ixs
                    d[word] = len(col_word_to_ixs)

                    node_feat_word_to_ixs[col] = d
                    if lock:
                        lock.release()

                    col_word_to_ixs = node_feat_word_to_ixs[col]
                    change = True
                else:
                    word = ''

            # new_dict_t += time.time() - start
            # start = time.time()
            result[j, col] = col_word_to_ixs[word]

            # assert_result_t += time.time() - start

    if save and change:
        if lock:
            lock.acquire()
        save_feat_word_to_ixs(Constants.GENERAL_WORD_TO_IDX_PATH, node_feat_word_to_ixs)
        if lock:
            lock.release()

    # end = time.time()
    # print(f'fill string features: {end - start_f:.2f}, new_dict_t: {new_dict_t:.2f}, assert_result_t: {assert_result_t:.2f}')
    return torch.from_numpy(result).to(dtype=torch.float32)


def create_dgl_graph(pairs, node_feat_word_to_ixs, lock, num_nodes, set_edge_features=False, node_features=None,
                     labels=None, coordinates=None, save=False):
    """
        Our main preprocess function.

    :param coordinates: TODO
    :param node_feat_word_to_ixs: dictionary of word to index dictionaries (for each string feature column)
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
        G.ndata[Constants.NODE_FEATURES_NAME] = transform_node_features(node_features, node_feat_word_to_ixs, lock, save)

    if coordinates is not None:
        G.ndata[Constants.COORDINATES_GRAPH_NAME] = coordinates

    if labels is not None:
        G.ndata[Constants.LABEL_NODE_NAME] = labels
    return G
