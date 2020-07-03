import os

import requests
from tqdm import tqdm

PDB_DIR = 'pdbs/'
URL_RCSB = 'https://files.rcsb.org/view/'


def load_data(start_pdb=0, limit=None):
    if not os.path.exists(PDB_DIR):
        os.makedirs(PDB_DIR)

    pdbs = []
    with open('pdbs.lst') as f:
        for pdb in f:
            pdbs.append(pdb.strip())

    print(f'Number of .pdbs: {len(pdbs)}')

    i = 0
    if limit is None:
        all_pdbs = pdbs[start_pdb:]
    else:
        all_pdbs = pdbs[start_pdb:start_pdb + limit]
    for pdb in tqdm(all_pdbs):
        # if i % 10 == 0:
        #     print('.', end='')
        filename = f'{pdb}.pdb'
        if not os.path.exists(PDB_DIR + filename):
            try:
                r = requests.get(URL_RCSB + filename, allow_redirects=True, timeout=8)
                open(PDB_DIR + filename, 'wb').write(r.content)
            except TimeoutError as te:
                continue

            i += 1
        # else:
        #     print(f'File {filename} already exists')

    if i > limit:
        print('\nWe reached limit !!! awwww')


if __name__ == '__main__':
    load_data(limit=500)
