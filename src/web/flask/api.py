import os
import random
import sys
import time

import torch
from flask import Flask
from flask_cors import CORS, cross_origin

from Data.Evaluate import predict_percent
from GNN.MyModels import MyModels

path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(path, '..', '..'))

from Constants import PDB_PATH
from Data.Data import get_dataset, my_pdb_parser
from Data.Preprocess import is_labeled_positive, is_protein, get_dgl_id

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
# app.config['CORS_HEADERS'] = 'Content-Type'


limit = 10
dataset, dataset_filenames, word_to_ixs, standardize = get_dataset(limit=limit)
del dataset

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


@app.route('/api/get_predictions/<pdb>')
def get_predictions(pdb):
    if not pdb.endswith('.pdb'):
        pdb = pdb + '.pdb'

    start = time.time()
    print(f'Get predictions for {pdb}')
    graph, atoms, pairs, labels = my_pdb_parser(os.path.join(PDB_PATH, pdb), word_to_ixs=word_to_ixs,
                                                standardize=standardize)
    t = time.time()
    print(f'Preprocessed in {t - start}')
    bio_model = atoms[0].get_parent().get_parent().get_parent()
    proteins = []
    for chain in bio_model.get_chains():
        if is_protein(chain):
            proteins.append(chain.id)
    print(f'Detect proteins in {time.time() - t}')
    t = time.time()

    atom_dict = {}
    predictions = predict_percent(net, [graph], predict_type='y_combine_all_percent')
    # print(predictions)
    for atom in atoms:
        # atom_dict[atom.serial_number] = min(max(is_labeled_positive(atom) + 0.0 + random.uniform(-0.4, 0.4), 0), 1)
        dgl_id = get_dgl_id(atom)
        atom_dict[atom.serial_number] = float(predictions[dgl_id])
    print(f'Predict atoms in  {time.time() - t}')
    t = time.time()

    pdb_file = None
    with open(os.path.join(PDB_PATH, pdb), 'r') as f:
        pdb_file = f.read()
    print(f'Read pdb file in {time.time() - t}')

    return {
        'protein_chains': proteins,
        'predictions': atom_dict,
        'optimal_threshold': 0.5,
        'file': pdb_file,
    }
