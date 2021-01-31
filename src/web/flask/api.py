import os
import random
import sys
import time

import torch
from Bio.PDB import PDBParser
from dgl.data import load_graphs
from flask import Flask
from flask_cors import CORS, cross_origin

from Data.Evaluate import predict_percent
from GNN.MyModels import MyModels

# path = os.path.abspath(os.path.dirname(__file__))
# sys.path.append(os.path.join(path, '..', '..'))

import Constants
from Data.Data import get_dataset, my_pdb_parser
from Data.Preprocess import is_labeled_positive, is_protein, get_dgl_id, load_feat_word_to_ixs, get_protein_chains, \
    get_atoms_list

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
# app.config['CORS_HEADERS'] = 'Content-Type'


# limit = 10
# dataset, dataset_filenames, word_to_ixs, standardize = get_dataset(limit=limit)
# del dataset

word_to_ixs = load_feat_word_to_ixs(os.path.join(Constants.SAVED_GRAPHS_PATH,
                                                 'graph_data_1424_all_atoms_word_to_ix'))
dataset_filenames = [os.path.splitext(fn)[0] for fn in os.listdir(Constants.SAVED_GRAPH_PATH)]
parser = PDBParser()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'first_linear_then_more_GraphConvs_then_linear'

my_models = MyModels(word_to_ixs)
net, loss = my_models.load_models(model_name, device)
print(f'Loaded model {model_name} with loss', loss)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/api/list_all_pdbs')
def list_all_pdbs():
    return {
        'all_pdbs': dataset_filenames
    }


@app.route('/api/get_predictions/<pdb_fn>')
def get_predictions(pdb_fn):
    if not pdb_fn.endswith('.pdb'):
        pdb_fn = pdb_fn + '.pdb'
    pdb_id = os.path.splitext(pdb_fn)[0]

    start = time.time()
    print(f'Get predictions for {pdb_fn}')
    graph = load_graphs(os.path.join(Constants.SAVED_GRAPH_PATH, pdb_id + Constants.GRAPH_EXTENSION))
    graph = graph[0][0]

    t = time.time()
    print(f'Preprocessed in {t - start}')

    with open(os.path.join(Constants.PDB_PATH, pdb_fn), 'r') as f:
        pdb_file = f.read()
        f.seek(0)
        bio_model = parser.get_structure(pdb_id, f)[0]
    protein_chains = get_protein_chains(bio_model)
    print(f'Detect proteins in {time.time() - t}')
    t = time.time()

    atom_dict = {}
    predictions = predict_percent(net, [graph], predict_type='y_combine_all_percent')
    # print(predictions)
    atoms = get_atoms_list(protein_chains)
    for dgl_id, atom in enumerate(atoms):
        # atom_dict[atom.serial_number] = min(max(is_labeled_positive(atom) + 0.0 + random.uniform(-0.4, 0.4), 0), 1)
        # dgl_id = get_dgl_id(atom)
        atom_dict[atom.serial_number] = float(predictions[dgl_id])
    print(f'Predict atoms in  {time.time() - t}')
    t = time.time()

    return {
        'protein_chains': [chain.id for chain in protein_chains],
        'predictions': atom_dict,
        'optimal_threshold': 0.5,
        'file': pdb_file,
    }
