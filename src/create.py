import time

from Data.Data import create_dataset, save_dataset

limit = 5
if __name__ == '__main__':
    start = time.time()
    dataset, dataset_filenames, word_to_ixs, standardize = create_dataset(limit=limit)
    save_dataset(dataset, dataset_filenames, word_to_ixs, *standardize, limit=limit)
    print(f"END: {time.time() - start}")
