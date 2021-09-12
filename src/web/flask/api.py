import os
import random
import sys
import time
import warnings
from os.path import splitext

import torch
from Bio.PDB import PDBParser
from dgl.data import load_graphs
from flask import Flask, send_from_directory, request, jsonify
from flask_basicauth import BasicAuth
from flask_cors import CORS, cross_origin


path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(path, '..', '..'))

import Constants
from Data.Evaluate import predict_percent, print_metrics, _get, smooth_graph
from GNN.MyModels import MyModels, list_models
from Data.Preprocess import is_labeled_positive, is_protein, get_dgl_id, load_feat_word_to_ixs, get_protein_chains, \
    get_atoms_list

NUM_MODELS = 6

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
# app.config['CORS_HEADERS'] = 'Content-Type'

app.config['BASIC_AUTH_USERNAME'] = 'admin'
app.config['BASIC_AUTH_PASSWORD'] = os.getenv('ADMIN_PASS', 'pass')
basic_auth = BasicAuth(app)

parser = PDBParser()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = Constants.BEST_MODEL
predict_type = 'y_combine_all_smooth_percent'

word_to_ixs = load_feat_word_to_ixs(Constants.GENERAL_WORD_TO_IDX_PATH)
my_models = MyModels(word_to_ixs)
net, loss, thresholds = my_models.get_model(model_name, device)
threshold = thresholds[predict_type]
# net, loss, threshold = None, None, None
print(f'Loaded model {model_name} with loss {loss} and threshold {threshold}.')
net_dict = {}


def load_models_dict():
    global net_dict
    net_dict = {}
    models = list_models(Constants.UPDATED_MODELS_PATH, NUM_MODELS)
    for model in models:
        date_prefix = model.split('_')[0]
        net_tmp, _, _ = my_models.get_model(model_name, device, prefix=date_prefix,
                                               path=Constants.UPDATED_MODELS_PATH)
        net_dict[date_prefix] = net_tmp


load_models_dict()


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/api/new_model', methods=['POST'])
@basic_auth.required
def new_model():
    file = request.files['model']
    fn = file.filename
    print(fn)
    if splitext(fn)[1] == '.pt':
        file_path = os.path.join(Constants.UPDATED_MODELS_PATH, fn)
        if not os.path.exists(file_path):
            file.save(file_path)

    load_models_dict()
    return jsonify(success=True)


@app.route('/api/list_models')
def send_list_models():
    models = list_models(Constants.UPDATED_MODELS_PATH, NUM_MODELS)
    models = list(map(lambda m: splitext(m)[0], models))
    return jsonify(models=models)


@app.route('/api/preprocessed_file/<pdb_id>')
def send_preprocessed_pdb(pdb_id):
    if '.' in pdb_id:
        pdb_id = Constants.filename_to_pdb_id(pdb_id)
    return send_from_directory(Constants.SAVED_GRAPH_PATH, pdb_id + Constants.GRAPH_EXTENSION)


@app.route('/api/list_all_pdbs')
def list_all_pdbs():
    dataset_pdb_ids = [Constants.filename_to_pdb_id(fn) for fn in os.listdir(Constants.SAVED_GRAPH_PATH)]
    return jsonify(all_pdbs=dataset_pdb_ids)


@app.route('/api/get_predictions/<pdb_fn>')
def get_predictions(pdb_fn):
    if not pdb_fn.endswith('.pdb'):
        pdb_fn = pdb_fn + '.pdb'
    pdb_id = Constants.filename_to_pdb_id(pdb_fn)

    model_tmp = request.args.get('model')
    if model_tmp is None:
        net_tmp = net
    else:
        date_prefix = model_tmp.split('_')[0]
        if date_prefix in net_dict:
            net_tmp = net_dict[date_prefix]
        else:
            return jsonify(success=False)

    start = time.time()
    print(f'Get predictions for {pdb_fn}')
    try:
        graph = load_graphs(os.path.join(Constants.SAVED_GRAPH_PATH, pdb_id + Constants.GRAPH_EXTENSION))
        graph = graph[0][0]
    except Exception as e:
        print(e, os.path.join(Constants.SAVED_GRAPH_PATH, pdb_id + Constants.GRAPH_EXTENSION))
        return jsonify(success=False)

    t = time.time()
    print(f'Preprocessed in {t - start}')

    with Constants.open_data_file(Constants.PDB_PATH, pdb_fn) as f:
        pdb_file = f.read()
        f.seek(0)
        with warnings.catch_warnings(record=True):
            bio_model = parser.get_structure(pdb_id, f)[0]
    protein_chains = get_protein_chains(bio_model)
    print(f'Detect proteins in {time.time() - t}')
    t = time.time()

    atom_dict = {}
    predictions = predict_percent(net_tmp, [graph], predict_type=predict_type)
    predictions = smooth_graph(graph, predictions)
    # print(predictions)
    atoms = get_atoms_list(protein_chains)
    for dgl_id, (atom, label) in enumerate(zip(atoms, graph.ndata[Constants.LABEL_NODE_NAME])):
        # atom_dict[atom.serial_number] = min(max(is_labeled_positive(atom) + 0.0 + random.uniform(-0.4, 0.4), 0), 1)
        # dgl_id = get_dgl_id(atom)
        # if int(atom.serial_number) != int(serial_number.item()):
        #     print('dafs', dgl_id, int(atom.serial_number), int(serial_number.item()))
        # print(int(serial_number.item()))
        atom_dict[int(atom.serial_number)] = float(predictions[dgl_id])
        # atom_dict[int(serial_number.item())] = float(label.item())
    # print(_get(graph.ndata[Constants.LABEL_NODE_NAME], predictions > 0.30535218, predictions))
    print(f'Predict atoms in  {time.time() - t}')
    t = time.time()

    return {
        'protein_chains': [chain.id for chain in protein_chains],
        'predictions': atom_dict,
        'optimal_threshold': threshold,
        'file': pdb_file,
        'success': True,
    }
