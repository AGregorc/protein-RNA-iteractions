import os
import time
from datetime import datetime

import requests
import torch
from sklearn.model_selection import train_test_split
from torch import nn

import Constants
from Data.Data import load_individuals
from Data.load_data import load_preprocessed_data, RNA_PROTEIN_QUERY, update_pdbs_list_and_load, get_pdb_list
from Data.utils import schedule_every_monday_at, is_first_week_of_month
from GNN.MyModels import MyModels, get_model_filename
from GNN.run_ignite import run

# change to number for debugging
LIMIT = None
PASS = os.getenv('ADMIN_PASS', 'pass')
RUN_AT_START = True


def update_model(limit=LIMIT, force_update=False, load_preprocessed=True):
    global RUN_AT_START
    if force_update:
        # Move forward and update model
        pass
    elif RUN_AT_START:
        RUN_AT_START = False
    elif not is_first_week_of_month():
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    now = datetime.now()

    # Update dataset
    start = time.time()
    pdb_list = get_pdb_list(query=RNA_PROTEIN_QUERY, limit=limit)
    print(f'Get pdb list done {time.time() - start}')

    if load_preprocessed:
        # First load preprocessed data from api then get the dataset
        load_preprocessed_data(pdb_list)

    dataset, dataset_pdb_ids, word_to_ixs = load_individuals(pdb_list)
    print(f'len {len(dataset)}, {len(pdb_list)}, {limit}')
    train_d, val_d, train_ids, val_ids = train_test_split(dataset, dataset_pdb_ids, shuffle=True, test_size=0.5)

    net = MyModels(word_to_ixs).my_models[Constants.BEST_MODEL]
    date_prefix = now.strftime(Constants.DATE_FORMAT)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 7.54], device=device))
    run(net, train_d, val_d,
        device=device,
        criterion=criterion,
        batch_size=10,
        epochs=2,
        path=Constants.UPDATED_MODELS_PATH,
        model_name_prefix=date_prefix)

    del train_d, val_d

    model_fn, loss = get_model_filename(Constants.UPDATED_MODELS_PATH, date_prefix)
    with open(os.path.join(Constants.UPDATED_MODELS_PATH, model_fn), 'rb') as f:
        try:
            resp = requests.post(Constants.DATA_API_URL + 'api/new_model', files={'model': f}, auth=('admin', PASS))
            if resp.status_code < 300:
                print('Successfully sent model to API')
            else:
                print('The model was not sent to API {resp}')
        except:
            print("Failed to establish a connection with API")


if __name__ == '__main__':
    print(f'Running update dataset process')
    schedule_every_monday_at(update_model, "01:00", RUN_AT_START)
