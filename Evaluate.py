import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, mean_squared_error

from Constants import LABEL_NODE_NAME
from GNN import GeneralModel


def calculate_metrics(dataset: list, model: GeneralModel, print_model_name: str):
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
    y_pred = logits.argmax(dim=1)

    y_pred[logits[:, 1].argmax(dim=0)] = 1  # at least the most probable atom should be in interaction

    confusion_mtx = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

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

    return confusion_mtx, f1, precision, recall, rmse
