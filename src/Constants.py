import gzip
import os
from Data import groups

DATA_API_URL = 'http://193.2.72.56:5004/'
# DATA_API_URL = 'http://localhost:5004/'

ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_PATH, '..', 'data')
SAVED_GRAPHS_PATH = os.path.join(DATA_PATH, 'preprocessed_data')
SAVED_GRAPHS_PATH_DEFAULT_FILE = os.path.join(SAVED_GRAPHS_PATH, 'graph_data')
SAVED_GRAPH_PATH = os.path.join(SAVED_GRAPHS_PATH, 'pdb_ids')
GRAPH_EXTENSION = '.bin'
PDB_PATH = os.path.join(DATA_PATH, 'pdbs')
DSSP_PATH = os.path.join(DATA_PATH, 'dssp')
MODELS_PATH = os.path.join(DATA_PATH, 'models')
UPDATED_MODELS_PATH = os.path.join(DATA_PATH, 'updated_models')
TMP_PATH = os.path.join(DATA_PATH, 'tmp')
GENERAL_WORD_TO_IDX_PATH = os.path.join(SAVED_GRAPHS_PATH, 'pdb_ids_word_to_ix')
# You can change this parameter
NUM_PROCESSES = int(os.getenv('NUM_PROCESSES', 6))


def makedir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


makedir_if_not_exists(DATA_PATH)
makedir_if_not_exists(SAVED_GRAPHS_PATH)
makedir_if_not_exists(SAVED_GRAPH_PATH)
makedir_if_not_exists(PDB_PATH)
makedir_if_not_exists(DSSP_PATH)
makedir_if_not_exists(MODELS_PATH)
makedir_if_not_exists(UPDATED_MODELS_PATH)
makedir_if_not_exists(TMP_PATH)

TRAIN_VAL_TEST_SPLIT_FILE_PATH = os.path.join(DATA_PATH, 'train_val_test_split.json')
PDB_ERROR_LIST = os.path.join(DATA_PATH, 'pdb_error_list.lst')

# Create file if it doesn't exists
open(TRAIN_VAL_TEST_SPLIT_FILE_PATH, 'a').close()
open(PDB_ERROR_LIST, 'a').close()


# Data file is either .pdb or .dssp and it should be opened with this function
def open_data_file(path, filename, read=True):
    if not filename.endswith(".gz"):
        filename += ".gz"

    mode = "rt"
    # if ".bin" in filename:
    #     mode = "rb"
    if not read:
        mode = "wt"
        # if ".bin" in filename:
        #     mode = "wb"

    fn = os.path.join(path, filename)
    return gzip.open(fn, mode)


# Data file is either .pdb pr .dssp
def data_file_exists(path, filename):
    if not filename.endswith(".gz"):
        filename += ".gz"
    return os.path.exists(os.path.join(path, filename))


def to_pdb_filename(pdb_id):
    pdb_id = filename_to_pdb_id(pdb_id)
    return pdb_id + '.pdb.gz'


def to_dssp_filename(pdb_id):
    pdb_id = filename_to_pdb_id(pdb_id)
    return pdb_id + '.dssp.gz'


def is_pdb(filename):
    return filename.endswith(".pdb") or filename.endswith(".pdb.gz")


def is_dssp(filename):
    return filename.endswith(".dssp") or filename.endswith(".dssp.gz")


def filename_to_pdb_id(filename):
    return filename.split(".")[0]


def set_model_directory(model_name):
    model_path = os.path.join(MODELS_PATH, model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)


GET_ONLY_CA_ATOMS = False

if GET_ONLY_CA_ATOMS:
    ATOM_ATOM_DISTANCE = 5
else:
    ATOM_ATOM_DISTANCE = 1.7

DEFAULT_ATOM_ATOM_DISTANCE = 1.7
# Our default ATOM_ATOM_DISTANCE is 1.7
# if it's not then rename saved graphs folder
if ATOM_ATOM_DISTANCE != DEFAULT_ATOM_ATOM_DISTANCE:
    SAVED_GRAPHS_PATH += '_' + str(ATOM_ATOM_DISTANCE)
    print(SAVED_GRAPHS_PATH)
    SAVED_GRAPH_PATH = os.path.join(SAVED_GRAPHS_PATH, 'pdb_ids')
    GENERAL_WORD_TO_IDX_PATH = os.path.join(SAVED_GRAPHS_PATH, 'pdb_ids_word_to_ix')
    makedir_if_not_exists(SAVED_GRAPHS_PATH)
    makedir_if_not_exists(SAVED_GRAPH_PATH)

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

BEST_MODEL = 'two_branches_small'
DATE_FORMAT = '%d-%m-%Y'
