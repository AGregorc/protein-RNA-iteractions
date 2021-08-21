import time

import schedule

from Data.Data import update_dataset
from Data.load_data import update_pdbs_list_and_load, ALL_PROTEINS_QUERY
from Data.utils import is_first_week_of_month, schedule_every_monday_at
from update_model import update_model

# change to number for debugging
LIMIT = None
RUN_AT_START = True
UPDATE_MODEL = True


def update_and_preprocess(limit=LIMIT):
    global RUN_AT_START
    if RUN_AT_START:
        RUN_AT_START = False
    elif not is_first_week_of_month():
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
    if UPDATE_MODEL:
        second_process = lambda: update_model(limit=LIMIT, force_update=True, load_preprocessed=False)
    else:
        second_process = None

    schedule_every_monday_at(update_and_preprocess, "00:00", RUN_AT_START, second_process=second_process)
