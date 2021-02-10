import os
from Data import groups

ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_PATH, '..', 'data')
SAVED_GRAPHS_PATH = os.path.join(DATA_PATH, 'preprocessed_data')
SAVED_GRAPHS_PATH_DEFAULT_FILE = os.path.join(SAVED_GRAPHS_PATH, 'graph_data')
SAVED_GRAPH_PATH = os.path.join(SAVED_GRAPHS_PATH, 'pdb_ids')
GRAPH_EXTENSION = '.bin'
PDB_PATH = os.path.join(DATA_PATH, 'pdbs')
DSSP_PATH = os.path.join(DATA_PATH, 'dssp')
MODELS_PATH = os.path.join(DATA_PATH, 'models')
NUM_PROCESSES = 12

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
if not os.path.exists(SAVED_GRAPHS_PATH):
    os.makedirs(SAVED_GRAPHS_PATH)
if not os.path.exists(SAVED_GRAPH_PATH):
    os.makedirs(SAVED_GRAPH_PATH)
if not os.path.exists(PDB_PATH):
    os.makedirs(PDB_PATH)
if not os.path.exists(DSSP_PATH):
    os.makedirs(DSSP_PATH)
if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)


TRAIN_VAL_TEST_SPLIT_FILE_PATH = os.path.join(DATA_PATH, 'train_val_test_split.json')


def set_model_directory(model_name):
    model_path = os.path.join(MODELS_PATH, model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)


GET_ONLY_CA_ATOMS = False

if GET_ONLY_CA_ATOMS:
    ATOM_ATOM_DISTANCE = 5
else:
    ATOM_ATOM_DISTANCE = 1.7

ATOM_DGL_ID = 'my_dgl_id'
LABEL_ATTRIBUTE_NAME = 'my_label'
LABEL_ATOM_DISTANCE = 4.2

LABEL_NODE_NAME = 'label'
LABEL_POSITIVE_COLOR = 'r'
LABEL_NEGATIVE_COLOR = 'b'
LABEL_POSITIVE = 1
LABEL_NEGATIVE = 0

EMPTY_STR_FEATURE = ' '
NODE_APPENDED_FEATURES = {
    'prev_res_name': 'previous_residue_name',
    'next_res_name': 'next_residue_name',
    'residue_depth': 'residue_depth',
    'atom_depth': 'atom_depth',
    'ca_depth': 'ca_depth',
    'ca_atom_dist': 'ca_atom_dist',
    'cb_ca_surf_angle': 'cb_ca_surf_angle',
    'ca_cb_surf_angle': 'ca_cb_surf_angle',
    'atom_ca_surf_angle': 'atom_ca_surf_angle',
    'ca_atom_surf_angle': 'ca_atom_surf_angle',
}


def neighbour_sum_radius_name(idx, atom=None):
    if atom is None:
        return 'neighbour_sum_radius_' + str(NEIGHBOUR_SUM_RADIUS[idx])
    else:
        return 'neighbour_sum_radius_' + str(atom) + '_' + str(NEIGHBOUR_SUM_RADIUS[idx])


def neighbour_sum_above_plane_radius_name(idx, atom=None):
    if atom is None:
        return 'neighbour_sum_radius_above_plane_' + str(NEIGHBOUR_SUM_RADIUS[idx])
    else:
        return 'neighbour_sum_radius_' + str(atom) + 'above_plane' + str(NEIGHBOUR_SUM_RADIUS[idx])


NEIGHBOUR_SUM_RADIUS = [2, 6, 10]
NEIGHBOUR_SUM_RADIUS_ATOMS = ['C', 'H', 'N', 'O']
for num in range(len(NEIGHBOUR_SUM_RADIUS)):
    name = neighbour_sum_radius_name(num)
    NODE_APPENDED_FEATURES[name] = name
    name_ap = neighbour_sum_above_plane_radius_name(num)
    NODE_APPENDED_FEATURES[name_ap] = name_ap

    for elem in NEIGHBOUR_SUM_RADIUS_ATOMS:
        name = neighbour_sum_radius_name(num, elem)
        NODE_APPENDED_FEATURES[name] = name
        name_ap = neighbour_sum_above_plane_radius_name(num, elem)
        NODE_APPENDED_FEATURES[name_ap] = name_ap

NODE_GROUP_FEATURES = groups.group_list
DSSP_FEATURES_NAME = 'dssp_features'
DSSP_FEATURES_NUM = 13

NODE_FEATURES_NAME = 'features'
NODE_FEATURES_NUM = 5 + len(NODE_APPENDED_FEATURES) + len(NODE_GROUP_FEATURES) + DSSP_FEATURES_NUM
EDGE_FEATURE_NAME = 'relative_position'
# EDGE_FEATURE_NUM = 4
EDGE_FEATURE_NUM = 0  # Ignoring edge features for now

COORDINATES_GRAPH_NAME = 'coordinates'


FEATURE_NAMES = [
    'mass',
    'occupancy',
    'element',
    'fullname',
    'aa'
]
for feature_name in NODE_APPENDED_FEATURES:
    FEATURE_NAMES.append(feature_name)

for group_feature in NODE_GROUP_FEATURES:
    FEATURE_NAMES.append(group_feature)

# dssp features
FEATURE_NAMES.append('dssp_aa')
FEATURE_NAMES.append('dssp_S')
FEATURE_NAMES.append('dssp_acc')
FEATURE_NAMES.append('dssp_phi')
FEATURE_NAMES.append('dssp_psi')
FEATURE_NAMES.append('dssp_nh->o_1.1')
FEATURE_NAMES.append('dssp_nh->o_1.2')
FEATURE_NAMES.append('dssp_o->hn_1.1')
FEATURE_NAMES.append('dssp_o->hn_1.2')
FEATURE_NAMES.append('dssp_nh->o_2.1')
FEATURE_NAMES.append('dssp_nh->o_2.2')
FEATURE_NAMES.append('dssp_o->hn_2.1')
FEATURE_NAMES.append('dssp_o->hn_2.2')

assert len(FEATURE_NAMES) == NODE_FEATURES_NUM
