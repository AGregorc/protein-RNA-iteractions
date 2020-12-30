import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from Constants import NODE_FEATURES_NUM
from Data.Data import get_dataset
from Data.Evaluate import calculate_metrics, majority_model_metrics
from Data.PlotMPL import plot_training_history
from GNN.MyModels import MyModels
from GNN.run_ignite import run


def main():
    limit = 5
    dataset, dataset_filenames, word_to_ixs, standardize = get_dataset(limit=limit)
    train_d, test_val_d, train_f, test_val_f = train_test_split(dataset, dataset_filenames, shuffle=False, test_size=0.4)
    val_d, test_d, val_f, test_f = train_test_split(test_val_d, test_val_f, shuffle=False, test_size=0.5)

    del dataset, dataset_filenames, test_val_d, test_val_f
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Number of features', NODE_FEATURES_NUM)

    models = MyModels(word_to_ixs)
    for model_name, net in models.my_models.items():
        # model_name = 'just_linear'
        # net = models.my_models[model_name]
        print(net)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.6], device=device))

        training_h, validation_h, whole_training_h = run(net, train_d, val_d,
                                                         criterion=criterion,
                                                         batch_size=10, epochs=1000,
                                                         model_name=model_name)
        print(f'Run for {model_name} is done\n\n')
        plot_training_history(training_h, validation_h, model_name=model_name, save=True)


if __name__ == '__main__':
    main()
