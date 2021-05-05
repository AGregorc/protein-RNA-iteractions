import Constants as C
import os
import json
import warnings

from Bio.PDB import PDBParser, PPBuilder

from Data.utils import get_analysis_pdb_list

directory = os.fsencode(C.PDB_PATH)

idx = 0
# dataset_dict = {}
dataset_pdb_ids = get_analysis_pdb_list()

pdb_to_seq = {}

parser = PDBParser()
ppb = PPBuilder()
i = 0
for pdb_id in dataset_pdb_ids:
    with warnings.catch_warnings(record=True):
        with C.open_data_file(C.PDB_PATH, C.to_pdb_filename(pdb_id)) as f:
            structure = parser.get_structure(pdb_id, f)
    model = structure[0]
    for pp in ppb.build_peptides(model):
        pdb_to_seq[pdb_id] = str(pp.get_sequence())
        break

pdb_to_ds = {}

with open(C.TRAIN_VAL_TEST_SPLIT_FILE_PATH) as file:
    split_d = json.load(file)
    for tr_val_or_test, pdb_ids in split_d.items():
        for pid in pdb_ids:
            pdb_to_ds[pid] = tr_val_or_test

seq_to_pdbs = {}

for pdb, seq in pdb_to_seq.items():
    pdbs = seq_to_pdbs.get(seq, [])
    pdbs.append(pdb)
    seq_to_pdbs[seq] = pdbs

seq_to_ds = {seq: [pdb_to_ds[pdb] for pdb in pdbs if pdb in pdb_to_ds] for seq, pdbs in seq_to_pdbs.items()}

in_more_ds = 0
for seq, ds in seq_to_ds.items():
    if not all(x == ds[0] for x in ds):
        in_more_ds += 1
#         print(seq, ds)

with open(os.path.join(C.DATA_PATH, 'seq_to_pdbs.json'), 'w') as fp:
    json.dump(seq_to_pdbs.copy(), fp, indent=2)
