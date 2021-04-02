import time

import schedule

from Data.Data import update_dataset
from Data.load_data import update_pdbs_list_and_load, ALL_PROTEINS_QUERY
from Data.utils import is_first_week_of_month, schedule_every_monday_at

# TODO: change to None
LIMIT = 2


def update_and_preprocess(limit=LIMIT):
    if not is_first_week_of_month():
        return

    # Update dataset
    start = time.time()
    success = update_pdbs_list_and_load(query=ALL_PROTEINS_QUERY, limit=limit)
    print(f'Loading pdb and dssp done {time.time() - start}, was successful: {success}')

    # Preprocess
    start = time.time()
    update_dataset(limit=limit, save_individual=True)
    print(f"END: {time.time() - start}")


if __name__ == '__main__':
    print(f'Running update dataset process')
    schedule_every_monday_at(update_and_preprocess, "00:00", True)
