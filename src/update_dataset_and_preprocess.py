import time
import schedule
import datetime

from Data.Data import update_dataset, save_dataset, get_dataset
from Data.load_data import update_pdbs_list_and_load, ALL_PROTEINS_QUERY, RNA_PROTEIN_QUERY

# TODO: change to None
LIMIT = 2

def is_first_week_of_month():
    day_of_month = datetime.datetime.now().day
    if day_of_month > 7:
        # not first day of month
        return False
    return True


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
    schedule.every().monday.do(update_and_preprocess)

    print(f'Running update dataset process')
    # Run the job now
    schedule.run_all()

    while True:
        schedule.run_pending()
        time.sleep(1)
