import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, mean_squared_error, roc_curve, \
    auc
import matplotlib.pyplot as plt
import numpy as np

from Constants import LABEL_NODE_NAME


def calculate_metrics(dataset: list, model, print_model_name: str):
    """
        Final metrics to compare different models
    :param dataset: list of graphs
    :param model: model
    :param print_model_name: to print model name
    :return: accuracy, precision, recall, f1 scores
    """
    y_true = dataset[0].ndata[LABEL_NODE_NAME]
    for graph in dataset[1:]:
        y_true = torch.cat((y_true, graph.ndata[LABEL_NODE_NAME]), dim=0)
    y_true = y_true.cpu()
    logits = model.predict(dataset)
    y_interaction_score = logits[:, 1]
    y_pred = logits.argmax(dim=1)

    y_pred[logits[:, 1].argmax(dim=0)] = 1  # at least the most probable atom should be in interaction

    fpr, tpr, thresholds = roc_curve(y_true, y_interaction_score, pos_label=1)
    area_under_curve = auc(fpr, tpr)

    confusion_mtx, f1, precision, recall, rmse = _get(y_true, y_pred)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    for i in range(2):
        if print_model_name is not None:
            print('Measures for {}:'.format(print_model_name))
            print('Confusion matrix:')
            print(confusion_mtx)
            print('F1 score:')
            print(f1)
            print('Precision:')
            print(precision)
            print('Recall:')
            print(recall)
            print('RMSE:')
            print(rmse)
            print('Optimal threshold: ')
            print(optimal_threshold)
            plot_roc(fpr, tpr, area_under_curve)
        y_pred = y_interaction_score > optimal_threshold
        print('And now when predicted is from optimal threshold of only one node')
        confusion_mtx, f1, precision, recall, rmse = _get(y_true, y_pred)

    return confusion_mtx, f1, precision, recall, rmse, optimal_threshold


def _get(y_true, y_pred):
    confusion_mtx = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return confusion_mtx, f1, precision, recall, rmse


def plot_roc(fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
