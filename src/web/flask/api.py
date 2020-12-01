import os
import sys

from flask import Flask
from flask_cors import CORS, cross_origin

path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(path, '..', '..'))

from Constants import PDB_PATH
from Data.Data import get_dataset, my_pdb_parser
from Data.Preprocess import is_labeled_positive, is_protein

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
# app.config['CORS_HEADERS'] = 'Content-Type'

limit = 5
dataset, dataset_filenames, word_to_ixs, standardize = get_dataset(limit=limit)
del dataset, dataset_filenames


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/api/get_predictions/<pdb>')
def get_predictions(pdb):
    pdb = pdb + '.pdb'
    _, atoms, pairs, labels = my_pdb_parser(os.path.join(PDB_PATH, pdb), word_to_ixs=word_to_ixs,
                                            standardize=standardize)
    model = atoms[0].get_parent().get_parent().get_parent()
    proteins = []
    for chain in model.get_chains():
        if is_protein(chain):
            proteins.append(chain.id)
    atom_dict = {}
    for atom in atoms:
        atom_dict[atom.serial_number] = is_labeled_positive(atom) + 0

    pdb_file = None
    with open(os.path.join(PDB_PATH, pdb), 'r') as f:
        pdb_file = f.read()
    return {
        'protein_chains': proteins,
        'predictions': atom_dict,
        'file': pdb_file,
    }
