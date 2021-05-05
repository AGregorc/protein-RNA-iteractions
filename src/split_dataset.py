import json
import os

from sklearn.model_selection import train_test_split, GroupShuffleSplit

import Constants
from Constants import TRAIN_VAL_TEST_SPLIT_FILE_PATH
from Data.Data import get_dataset
from Data.Evaluate import dataset_info


def randomized_stratified_group_split(dataset, dataset_pdb_ids, distributions, groups, max_diff=0.2, max_tries=200):
    assert sum(distributions) == 1
    assert len(distributions) == 3  # train, val, test
    val_te_size = distributions[1] + distributions[2]
    test_size = distributions[2] / val_te_size

    best_diff = 100
    best_tr_ids = None
    best_val_ids = None
    best_te_ids = None
    best_i = -1
    gss = GroupShuffleSplit(n_splits=max_tries, test_size=val_te_size, random_state=77)

    def get_subset(li, ids):
        return [li[d] for d in ids]

    # for i in range(max_tries):
    for i, (train_idx, test_val_idx) in enumerate(gss.split(dataset, groups=groups)):
        train_d = get_subset(dataset, train_idx)
        train_ids = get_subset(dataset_pdb_ids, train_idx)
        test_val_d = get_subset(dataset, test_val_idx)
        test_val_ids = get_subset(dataset_pdb_ids, test_val_idx)

        val_d, test_d, val_ids, test_ids = train_test_split(test_val_d, test_val_ids, shuffle=True, test_size=test_size)

        labels_p = dataset_info(train_d, val_d, test_d, do_print=False)
        trv = abs(labels_p[0] - labels_p[1])
        vte = abs(labels_p[1] - labels_p[2])
        trte = abs(labels_p[0] - labels_p[2])
        tmp_diff = max(trv, vte, trte)
        print(f'{i} curr diff: {tmp_diff}')
        if tmp_diff < max_diff:
            return train_ids, val_ids, test_ids

        if tmp_diff < best_diff:
            best_diff = tmp_diff
            best_tr_ids = train_ids
            best_val_ids = val_ids
            best_te_ids = test_ids
            best_i = i
    print(f'best i: {best_i}, {best_diff}')
    return best_tr_ids, best_val_ids, best_te_ids


def get_train_val_test_data(dataset, dataset_ids, split_file=TRAIN_VAL_TEST_SPLIT_FILE_PATH):
    pdb_to_ds = {}

    with open(split_file) as file:
        split_d = json.load(file)
        for tr_val_or_test, pdb_ids in split_d.items():
            for pid in pdb_ids:
                pdb_to_ds[pid] = tr_val_or_test

    train_d = []
    train_ids = []
    val_d = []
    val_ids = []
    test_d = []
    test_ids = []
    for data, pdb_id in zip(dataset, dataset_ids):
        if pdb_id not in pdb_to_ds:
            continue
        tr_val_or_test = pdb_to_ds[pdb_id]
        tmp_d = train_d
        tmp_ids = train_ids
        if tr_val_or_test == 'validation':
            tmp_d = val_d
            tmp_ids = val_ids
        if tr_val_or_test == 'test':
            tmp_d = test_d
            tmp_ids = test_ids
        tmp_d.append(data)
        tmp_ids.append(pdb_id)

    return train_d, train_ids, val_d, val_ids, test_d, test_ids


def split_and_save(data_limit, pdb_to_group):
    dataset, dataset_pdb_ids, word_to_ixs, standardize = get_dataset(limit=data_limit)
    groups = [pdb_to_group[pid] for pid in dataset_pdb_ids]
    print(groups)
    train_ids, val_ids, test_ids = randomized_stratified_group_split(dataset, dataset_pdb_ids, [0.6, 0.2, 0.2], groups)
    split_dict = {
        'train': train_ids,
        'validation': val_ids,
        'test': test_ids
    }
    with open(TRAIN_VAL_TEST_SPLIT_FILE_PATH, 'w') as f:
        json.dump(split_dict, f, indent=2)

    train_d, train_ids, val_d, val_ids, test_d, test_ids = get_train_val_test_data(dataset, dataset_pdb_ids)
    dataset_info(train_d, val_d, test_d)


if __name__ == '__main__':

    pdb_to_group_dict = {}
    with open(os.path.join(Constants.DATA_PATH, 'seq_to_pdbs.json'), 'r') as fp:
        seq_to_pdbs = json.load(fp)
        for idx, (seq, pdbs) in enumerate(seq_to_pdbs.items()):
            for pdb in pdbs:
                pdb_to_group_dict[pdb] = idx
    print(pdb_to_group_dict)

    limit = 1424
    split_and_save(limit, pdb_to_group_dict)


