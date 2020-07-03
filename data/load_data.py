import os

import requests
from tqdm import tqdm

PDB_DIR = 'pdbs/'


def load_data(start_pdb=139, limit=500):
    pdbs = []
    with open('pdbs.lst') as f:
        for pdb in f:
            pdbs.append(pdb.strip())

    print(f'Number of .pdbs: {len(pdbs)}')
    url = 'https://files.rcsb.org/view/'

    i = 0
    for pdb in tqdm(pdbs[start_pdb:start_pdb + limit]):
        # if i % 10 == 0:
        #     print('.', end='')
        filename = f'{pdb}.pdb'
        if not os.path.exists(PDB_DIR + filename):
            try:
                r = requests.get(url + filename, allow_redirects=True)
                open(PDB_DIR + filename, 'wb').write(r.content)
            except TimeoutError as te:
                continue

            i += 1
        # else:
        #     print(f'File {filename} already exists')

    if i > limit:
        print('\nWe reached limit !!! awwww')


if __name__ == '__main__':
    load_data()
