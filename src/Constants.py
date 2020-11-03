import os
import groups

DATA_PATH = '../data/'
SAVED_GRAPHS_PATH = DATA_PATH + 'preprocessed_data/'
SAVED_GRAPHS_PATH_DEFAULT_FILE = SAVED_GRAPHS_PATH + 'graph_data'
GRAPH_EXTENSION = '.bin'
PDB_PATH = DATA_PATH + 'pdbs/'
DSSP_PATH = DATA_PATH + 'dssp/'
NUM_PROCESSES = 10

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
if not os.path.exists(SAVED_GRAPHS_PATH):
    os.makedirs(SAVED_GRAPHS_PATH)
if not os.path.exists(PDB_PATH):
    os.makedirs(PDB_PATH)
if not os.path.exists(DSSP_PATH):
    os.makedirs(DSSP_PATH)

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


def neighbor_sum_radius_name(idx):
    return 'neighbor_sum_radius_' + str(NEIGHBOR_SUM_RADIUS[idx])


def neighbor_sum_above_plane_radius_name(idx):
    return 'neighbor_sum_radius_above_plane_' + str(NEIGHBOR_SUM_RADIUS[idx])


NEIGHBOR_SUM_RADIUS = [1.5, 2, 4, 6, 8, 10]
for num in range(len(NEIGHBOR_SUM_RADIUS)):
    name = neighbor_sum_radius_name(num)
    NODE_APPENDED_FEATURES[name] = name
    name_ap = neighbor_sum_above_plane_radius_name(num)
    NODE_APPENDED_FEATURES[name_ap] = name_ap

NODE_GROUP_FEATURES = groups.group_list
DSSP_FEATURES_NAME = 'dssp_features'
DSSP_FEATURES_NUM = 13

NODE_FEATURES_NAME = 'features'
NODE_FEATURES_NUM = 5 + len(NODE_APPENDED_FEATURES) + len(NODE_GROUP_FEATURES) + DSSP_FEATURES_NUM
EDGE_FEATURE_NAME = 'relative_position'
EDGE_FEATURE_NUM = 4

COORDINATES_GRAPH_NAME = 'coordinates'
