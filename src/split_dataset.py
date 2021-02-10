import json

from sklearn.model_selection import train_test_split

from Constants import TRAIN_VAL_TEST_SPLIT_FILE_PATH
from Data.Data import get_dataset
from Data.Evaluate import dataset_info


def randomized_stratified_group_split(dataset, dataset_filenames, distributions, max_diff=0.2, max_tries=100):
    assert sum(distributions) == 1
    assert len(distributions) == 3  # train, val, test
    val_te_size = distributions[1] + distributions[2]
    test_size = distributions[2] / val_te_size

    best_diff = 100
    best_trf = None
    best_valf = None
    best_tef = None
    best_i = -1

    for i in range(max_tries):
        train_d, test_val_d, train_f, test_val_f = train_test_split(dataset, dataset_filenames, shuffle=True,
                                                                    test_size=val_te_size)
        val_d, test_d, val_f, test_f = train_test_split(test_val_d, test_val_f, shuffle=True, test_size=test_size)

        labels_p = dataset_info(train_d, val_d, test_d, do_print=False)
        trv = abs(labels_p[0] - labels_p[1])
        vte = abs(labels_p[1] - labels_p[2])
        trte = abs(labels_p[0] - labels_p[2])
        tmp_diff = max(trv, vte, trte)
        print(f'{i} curr diff: {tmp_diff}')
        if tmp_diff < max_diff:
            return train_f, val_f, test_f

        if tmp_diff < best_diff:
            best_diff = tmp_diff
            best_trf = train_f
            best_valf = val_f
            best_tef = test_f
            best_i = i
    print(f'best i: {best_i}, {best_diff}')
    return best_trf, best_valf, best_tef


def get_train_val_test_data(dataset, dataset_filenames, split_file=TRAIN_VAL_TEST_SPLIT_FILE_PATH):
    file_to_ds = {}

    with open(split_file) as file:
        split_d = json.load(file)
        for tr_val_or_test, filenames in split_d.items():
            for fn in filenames:
                file_to_ds[fn] = tr_val_or_test

    train_d = []
    train_f = []
    val_d = []
    val_f = []
    test_d = []
    test_f = []
    for data, filename in zip(dataset, dataset_filenames):
        if not filename.endswith('.pdb'):
            filename = filename + '.pdb'
        tr_val_or_test = file_to_ds[filename]
        tmp_d = train_d
        tmp_f = train_f
        if tr_val_or_test == 'validation':
            tmp_d = val_d
            tmp_f = val_f
        if tr_val_or_test == 'test':
            tmp_d = test_d
            tmp_f = test_f
        tmp_d.append(data)
        tmp_f.append(filename)

    return train_d, train_f, val_d, val_f, test_d, test_f


def split_and_save(data_limit):
    dataset, dataset_filenames, word_to_ixs, standardize = get_dataset(limit=data_limit)
    train_f, val_f, test_f = randomized_stratified_group_split(dataset, dataset_filenames, [0.6, 0.2, 0.2])
    split_dict = {
        'train': train_f,
        'validation': val_f,
        'test': test_f
    }
    with open(TRAIN_VAL_TEST_SPLIT_FILE_PATH, 'w') as f:
        json.dump(split_dict, f, indent=2)

    train_d, train_f, val_d, val_f, test_d, test_f = get_train_val_test_data(dataset, dataset_filenames)
    dataset_info(train_d, val_d, test_d)


if __name__ == '__main__':
    data_limit = 1424
    split_and_save(data_limit)
