import copy
import os
import time
from datetime import datetime, date

import torch
from schedule import Scheduler
from torch import nn

from Constants import DATA_PATH
from Data.Data import load_individuals
from Data.Evaluate import dataset_info, calculate_metrics, feature_importance
from Data.PlotMPL import plot_training_history
from GNN.MyModels import MyModels
from GNN.run_ignite import run
from split_dataset import get_train_val_test_data


def get_analysis_pdb_list(start_pdb=0, limit=None):
    if start_pdb is None:
        start_pdb = 0
    pdbs = []
    with open(os.path.join(DATA_PATH, 'pdbs.lst')) as f:
        i = 0
        st = 0
        for pdb in f:
            if i >= start_pdb:
                pdbs.append(pdb.strip())
                st += 1
            i += 1
            if limit is not None and st >= limit:
                break
    return pdbs


def data(limit=1424, test=False):
    pdbs = get_analysis_pdb_list(limit)

    dataset, dataset_pdb_ids, word_to_ixs = load_individuals(pdbs)

    train_d, train_ids, val_d, val_ids, test_d, test_ids = get_train_val_test_data(dataset, dataset_pdb_ids)
    dataset_info(train_d, val_d, test_d)

    del dataset, dataset_pdb_ids
    if test:
        return train_d, train_ids, test_d, test_ids, word_to_ixs
    else:
        return train_d, train_ids, val_d, val_ids, word_to_ixs


def tune_hyperparameter(word_to_ixs, model_name, train_d, val_d, device, weights=None):
    if weights is None:
        weights = [1.0, 2.0, 3.0, 5.0, 6.0, 7.54, 9.0]

    net_original = MyModels(word_to_ixs).my_models[model_name]
    print(net_original, model_name)
    best_auc = 0
    best_hyperparam = None

    for weight in weights:
        print(f'Training with weight: {weight}')
        net = MyModels(word_to_ixs).my_models[model_name]
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, weight], device=device))
        run(net, copy.deepcopy(train_d), copy.deepcopy(val_d),
            device=device,
            criterion=criterion,
            batch_size=10,
            epochs=500,
            model_name=model_name,
            model_name_prefix='w_' + str(weight))

        thresholds, auc = calculate_metrics(val_d, net, print_model_name=model_name, do_plot=False, save=False)
        curr_auc = auc['y_combine_all_smooth_percent']
        # my_models.save_thresholds(model_name, thresholds)
        if curr_auc >= best_auc:
            best_hyperparam = weight
            best_auc = curr_auc
    print('Best weight: ', best_hyperparam)
    return best_hyperparam


def train_load_model(my_models, model_name, do_train, train_d, val_d, device, calc_metrics=True, calc_feat_i=False):
    net = my_models.my_models[model_name]

    print(net, model_name)
    if do_train:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 6.0], device=device))
        training_h, validation_h, whole_training_h = run(net, train_d, val_d,
                                                         device=device,
                                                         criterion=criterion,
                                                         batch_size=10,
                                                         epochs=10000,
                                                         model_name=model_name)
        print(f'Run for {model_name} is done\n\n')

        plot_training_history(training_h, validation_h, model_name=model_name, save=True)
        thresholds = None
    else:
        net, loss, thresholds = my_models.get_model(model_name, device)
    if calc_metrics:
        thresholds, auc = calculate_metrics(val_d, net, print_model_name=model_name, do_plot=True, save=True, thresholds=thresholds)
        my_models.save_thresholds(model_name, thresholds)
    if calc_feat_i:
        feature_importance(net, val_d, model_name, save=True)
    return net


def is_first_week_of_month():
    day_of_month = datetime.now().day
    if day_of_month > 7:
        # not first day of month
        return False
    return True


def is_midnight():
    now = datetime.now()
    seconds_since_midnight = (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    return seconds_since_midnight <= 1


def schedule_every_monday_at(process, str_time, run_at_start=True, second_process=None):
    scheduler1 = Scheduler()
    scheduler1.every().monday.at(str_time).do(process)
    if second_process is not None:
        scheduler1.every().friday.at(str_time).do(second_process)

    if run_at_start:
        # Run the job now
        scheduler1.run_all()

    while True:
        scheduler1.run_pending()
        if is_midnight():
            print(f"Scheduler is working, today is {date.today()}")
        time.sleep(1)
