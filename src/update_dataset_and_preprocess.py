import time

from Data.Data import update_dataset, save_dataset, get_dataset
from Data.load_data import update_pdbs_list_and_load, ALL_PROTEINS_QUERY, RNA_PROTEIN_QUERY

limit = None
if __name__ == '__main__':
    # Update dataset
    start = time.time()
    success = update_pdbs_list_and_load(query=ALL_PROTEINS_QUERY)
    print(f'Loading pdb and dssp done {time.time() - start}, was successful: {success}')

    # Preprocess
    start = time.time()
    update_dataset(limit=limit, save_individual=True)
    print(f"END: {time.time() - start}")
