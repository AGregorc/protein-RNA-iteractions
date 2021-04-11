import torch

from Constants import NODE_FEATURES_NUM, GENERAL_WORD_TO_IDX_PATH
from Data.Evaluate import majority_model_metrics
from Data.PlotMPL import visualize_model, plot_from_file
from Data.Preprocess import load_feat_word_to_ixs
from Data.utils import data, train_load_model, tune_hyperparameter
from GNN.MyModels import MyModels


def main():
    class WhatUWannaDoNow:
        TRAIN = 0
        TUNE_HYPERPARAMS = 1
        VISUALIZE_MODELS = 2
        VISUALIZE_METRICS = 3
        FEATURE_IMPORTANCE = 4

    data_limit = None
    model_names = [
                   'two_branches_small',
                   'two_branches']
    # model_names = 'two_branches_small'
    what_to_do = WhatUWannaDoNow.VISUALIZE_METRICS
    metrics = True

    train_d, train_f, val_d, val_f, word_to_ixs = data(limit=data_limit)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Number of features {NODE_FEATURES_NUM}, device {device}')

    models = MyModels(word_to_ixs, ignore_columns=[])
    if model_names == 'all':
        model_names = list(models.my_models)
    elif not isinstance(model_names, list):
        model_names = [model_names]

    for model_name in model_names:
        if what_to_do == WhatUWannaDoNow.TRAIN:
            train_load_model(models, model_name, True, train_d, val_d, device, metrics, False)
        if what_to_do == WhatUWannaDoNow.VISUALIZE_METRICS:
            train_load_model(models, model_name, False, train_d, val_d, device, True, False)
        if what_to_do == WhatUWannaDoNow.TUNE_HYPERPARAMS:
            tune_hyperparameter(word_to_ixs, model_name, train_d, val_d, device)
        if what_to_do == WhatUWannaDoNow.VISUALIZE_MODELS:
            visualize_model(models.my_models[model_name], model_name, val_d[0])
        if what_to_do == WhatUWannaDoNow.FEATURE_IMPORTANCE:
            train_load_model(models, model_name, False, train_d, val_d, device, metrics, True)

    if metrics:
        majority_model_metrics(val_d, save=True)

    # use_new_window()
    # #
    # plot_from_file(train_f[0],
    #                lambda atom: LABEL_POSITIVE_COLOR if is_labeled_positive(atom) else LABEL_NEGATIVE_COLOR,
    #                word_to_ixs, standardize=standardize)
    # plot_predicted(train_f[0], net, word_to_ixs, standardize=standardize)


if __name__ == '__main__':
    main()
    # plot_from_file('2ru2.pdb', lambda atom: None, load_feat_word_to_ixs(GENERAL_WORD_TO_IDX_PATH))
