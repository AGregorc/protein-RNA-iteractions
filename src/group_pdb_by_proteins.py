import Constants
import os
import json
import warnings

from Bio.PDB import PDBParser, PPBuilder

directory = os.fsencode(Constants.PDB_PATH)

idx = 0
# dataset_dict = {}
dataset_filenames = []

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".pdb"):
        # dataset_dict[filename] = idx
        dataset_filenames.append(filename)
        idx += 1

pdb_to_seq = {}

parser = PDBParser()
ppb = PPBuilder()
i = 0
for filename in dataset_filenames:
    with warnings.catch_warnings(record=True):
        with open(os.path.join(Constants.PDB_PATH, filename)) as f:
            structure = parser.get_structure(os.path.splitext(filename)[0], f)
    model = structure[0]
    for pp in ppb.build_peptides(model):
        #print(pp.get_sequence())
        pdb_to_seq[filename] = str(pp.get_sequence())
        break

file_to_ds = {}

with open(Constants.TRAIN_VAL_TEST_SPLIT_FILE_PATH) as file:
    split_d = json.load(file)
    for tr_val_or_test, filenames in split_d.items():
        for fn in filenames:
            file_to_ds[fn] = tr_val_or_test

seq_to_pdbs = {}

for pdb, seq in pdb_to_seq.items():
    pdbs = seq_to_pdbs.get(seq, [])
    pdbs.append(pdb)
    seq_to_pdbs[seq] = pdbs

seq_to_ds = {seq: [file_to_ds[pdb] for pdb in pdbs if pdb in file_to_ds] for seq, pdbs in seq_to_pdbs.items()}

in_more_ds = 0
for seq, ds in seq_to_ds.items():
    if not all(x == ds[0] for x in ds):
        in_more_ds += 1
#         print(seq, ds)

with open(os.path.join(Constants.DATA_PATH, 'seq_to_pdbs.json'), 'w') as fp:
    json.dump(seq_to_pdbs.copy(), fp, indent=2)
