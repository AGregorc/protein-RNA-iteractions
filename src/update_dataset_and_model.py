import os

import Constants
from Data.load_data import load_preprocessed_data

# TODO
if __name__ == '__main__':
    pdbs = []
    with open(os.path.join(Constants.DATA_PATH, 'pdbs.lst')) as f:
        for pdb in f:
            pdbs.append(pdb.strip())

    load_preprocessed_data(pdbs)
