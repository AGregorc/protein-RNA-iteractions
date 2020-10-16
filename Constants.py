import os
import groups

DATA_PATH = 'data/'
SAVED_GRAPHS_PATH = DATA_PATH + 'preprocessed_data/'
SAVED_GRAPHS_PATH_DEFAULT_FILE = SAVED_GRAPHS_PATH + 'graph_data'
GRAPH_EXTENSION = '.bin'
PDB_PATH = DATA_PATH + 'pdbs/'
NUM_THREADS = 5

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
if not os.path.exists(SAVED_GRAPHS_PATH):
    os.makedirs(SAVED_GRAPHS_PATH)
if not os.path.exists(PDB_PATH):
    os.makedirs(PDB_PATH)

GET_ONLY_CA_ATOMS = False

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

NODE_GROUP_FEATURES = groups.group_list

NODE_FEATURES_NAME = 'features'
NODE_FEATURES_NUM = 5 + len(NODE_APPENDED_FEATURES) + len(NODE_GROUP_FEATURES)
EDGE_FEATURE_NAME = 'relative_position'
EDGE_FEATURE_NUM = 4

