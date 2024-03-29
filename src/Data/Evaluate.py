import os

import dgl
import torch
from captum.attr import IntegratedGradients, DeepLift
from dgl import DGLGraph
import dgl.function as fn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, mean_squared_error, roc_curve, \
    auc
import matplotlib.pyplot as plt
import numpy as np

import Constants
from Constants import LABEL_NODE_NAME


def predict(net, dataset):
    net.eval()
    result = torch.empty(0, 2)
    device = next(net.parameters()).device

    with torch.no_grad():
        for g in dataset:
            g = g.to(device)
            logits = net(g)
            result = torch.cat((result, logits.cpu()), 0)
    return result


def smooth_graph(g: DGLGraph, node_data, p=0.3):
    node_name = 'smooth'
    result_name = 'result'
    g.ndata[node_name] = node_data

    def reducer(nodes):
        min_val, _ = torch.min(nodes.mailbox['m'], dim=1)
        # print(torch.min(nodes.mailbox['m'], dim=1))
        return {
            result_name:
                (nodes.data[node_name] + min_val + nodes.mailbox['m'].sum(1)) / (nodes.mailbox['m'].shape[1] + 2)
        }

    g.update_all(fn.copy_u(node_name, 'm'), reducer)
    return node_data * (1 - p) + g.ndata[result_name] * p


def smooth_dataset(graphs, node_data, p=0.3):
    result = torch.empty(node_data.shape)
    curr_idx = 0

    for g in graphs:
        to_idx = curr_idx + g.number_of_nodes()
        result[curr_idx:to_idx] = smooth_graph(g, node_data[curr_idx:to_idx], p)
        curr_idx = to_idx

    return result


def predict_percent(model, dataset, predict_type=None):
    model.eval()
    output = predict(model, dataset)

    y_pred = output.argmax(dim=1)
    y_pred[output[:, 1].argmax(dim=0)] = 1  # at least the most probable atom should be in interaction

    # Calculate confidence probabilities in a different way
    is_not_p = torch.sigmoid(output[:, 0])  # Is not in interaction
    is_in_p = torch.sigmoid(output[:, 1])  # Is in interaction

    y_interaction_percent = is_in_p
    y_reverse_interaction_percent = 1 - is_not_p

    c = torch.stack((y_reverse_interaction_percent, y_interaction_percent), dim=1)
    y_pick_major_percent = torch.gather(c, 1, torch.unsqueeze(y_pred, 1)).squeeze()
    y_combine_percent = (y_interaction_percent + y_reverse_interaction_percent) / 2

    y_combine_all_percent = (y_interaction_percent + y_reverse_interaction_percent + y_pick_major_percent) / 3
    y_combine_all_smooth_percent = smooth_dataset(dataset, y_combine_all_percent)

    all_predictions = {
        'y_pred': y_pred,
        'y_interaction_percent': y_interaction_percent,
        'y_reverse_interaction_percent': y_reverse_interaction_percent,
        'y_combine_percent': y_combine_percent,
        'y_pick_major_percent': y_pick_major_percent,
        'y_combine_all_percent': y_combine_all_percent,
        'y_combine_all_smooth_percent': y_combine_all_smooth_percent,
    }
    if predict_type is None:
        return all_predictions
    return all_predictions[predict_type]


def majority_model_metrics(dataset: list, do_plot=True, save=False):
    model_name = 'majority_classifier'
    if save:
        Constants.set_model_directory(model_name)
    y_true = dataset[0].ndata[LABEL_NODE_NAME].cpu()
    for graph in dataset[1:]:
        y_true = torch.cat((y_true, graph.ndata[LABEL_NODE_NAME].cpu()), dim=0)
    y_true = y_true.cpu()
    y_pred = torch.zeros_like(y_true)

    all_predictions = {
        'y_pred': y_pred,
    }
    print_metrics(y_true, all_predictions, model_name, do_plot, save)


def calculate_metrics(dataset: list, model, print_model_name: str, do_plot=True, save=False, thresholds=None):
    """
        Final metrics to compare different models
    :param dataset: list of graphs
    :param model: model
    :param print_model_name: to print model name
    :param do_plot: if true it plots roc and other visualizations
    :return: accuracy, precision, recall, f1 scores
    """
    if save:
        Constants.set_model_directory(print_model_name)

    y_true = _get_y_true(dataset)
    all_predictions = predict_percent(model, dataset, predict_type=None)
    return print_metrics(y_true, all_predictions, print_model_name, do_plot, save, threshold_dict=thresholds)


def print_metrics(y_true, all_predictions, print_model_name: str, do_plot=True, save=False, threshold_dict=None):
    string_out = []
    if threshold_dict is None:
        all_thresholds = {}
    else:
        all_thresholds = threshold_dict
    all_aucs = {}

    for name, predictions in all_predictions.items():
        fpr, tpr, thresholds = roc_curve(y_true, predictions, pos_label=1)
        if str(name) not in all_thresholds:
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = float(thresholds[optimal_idx])
            if optimal_threshold <= 0:
                optimal_threshold = 0.01
            if optimal_threshold >= 1:
                optimal_threshold = 0.99
            all_thresholds[str(name)] = float(optimal_threshold)
        else:
            optimal_threshold = all_thresholds[str(name)]
            optimal_idx = len(thresholds)
            for idx, th in enumerate(thresholds):
                if float(th) <= optimal_threshold:
                    optimal_idx = idx
                    break
        area_under_curve = auc(fpr, tpr)
        all_aucs[str(name)] = area_under_curve

        y_predicted = predictions > optimal_threshold
        confusion_mtx, f1, precision, recall, rmse = _get(y_true, y_predicted, predictions)

        if do_plot:
            plot_positive_hist(y_true, predictions, save=save, model_name=print_model_name, appendix=name)
            plot_negative_hist(y_true, predictions, save=save, model_name=print_model_name, appendix=name)

        if print_model_name is not None:
            string_out.append(f'Measures for {print_model_name} predicted as {name}:')
            string_out.append('Confusion matrix:')
            string_out.append(str(confusion_mtx))
            string_out.append('F1 score:')
            string_out.append(str(f1))
            string_out.append('Precision:')
            string_out.append(str(precision))
            string_out.append('Recall:')
            string_out.append(str(recall))
            string_out.append('RMSE:')
            string_out.append(str(rmse))
            string_out.append('AUC:')
            string_out.append(str(area_under_curve))
            string_out.append('Optimal threshold: ')
            string_out.append(str(optimal_threshold))
            if do_plot:
                plot_roc(fpr, tpr, area_under_curve, optimal_idx, model_name=print_model_name, save=save, appendix=name)
        string_out.append('\n\n')

    if print_model_name is not None:
        str_result = '\n'.join(string_out)
        if save:
            with open(os.path.join(Constants.MODELS_PATH, print_model_name, 'measures.txt'), 'w') as f:
                f.write(str_result)
        print(str_result)

    return all_thresholds, all_aucs


def _get(y_true, y_pred, percentages):
    # y_pred_bin = y_pred
    # if threshold:
    #     y_pred_bin = y_pred > threshold
    confusion_mtx = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    rmse = mean_squared_error(y_true, percentages) ** 0.5
    return confusion_mtx, f1, precision, recall, rmse


def _get_y_true(dataset):
    y_true = dataset[0].ndata[LABEL_NODE_NAME].cpu()
    for graph in dataset[1:]:
        y_true = torch.cat((y_true, graph.ndata[LABEL_NODE_NAME].cpu()), dim=0)
    y_true = y_true.cpu()
    return y_true


def plot_roc(fpr, tpr, roc_auc, threshold_idx, save=False, model_name='', appendix=''):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.plot(fpr[threshold_idx], tpr[threshold_idx], marker='o', color='black')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for ' + model_name + ' ' + appendix)
    plt.legend(loc="lower right")
    if save:
        plt.savefig(os.path.join(Constants.MODELS_PATH, model_name, appendix + '_ROC.png'))
    else:
        plt.show()
    plt.close()


def _pos_neg_hist(y_true, y_pred_percent, val, title, save=False, model_name='', appendix='', ylim=0.0):
    plt.figure()
    plt.hist(y_pred_percent[np.where(y_true == val)[0]], bins=100)
    plt.title(title + " " + appendix)
    plt.xlim([0.0, 1.0])
    if ylim > 1:
        plt.ylim([0.0, ylim])
    if save:
        plt.savefig(os.path.join(Constants.MODELS_PATH, model_name, appendix + ' ' + title + '.png'))
    else:
        plt.show()
    plt.close()


def plot_positive_hist(y_true, y_pred_percent, save=False, model_name='', appendix=''):
    _pos_neg_hist(y_true, y_pred_percent, 1, 'Positive histogram', save, model_name, appendix, ylim=19000)


def plot_negative_hist(y_true, y_pred_percent, save=False, model_name='', appendix=''):
    _pos_neg_hist(y_true, y_pred_percent, 0, 'Negative histogram', save, model_name, appendix)


def dataset_info(train, validation, test, do_print=True):
    y_train = _get_y_true(train)
    y_val = _get_y_true(validation)
    y_test = _get_y_true(test)

    train_count = torch.bincount(y_train)
    val_count = torch.bincount(y_val)
    test_count = torch.bincount(y_test)

    i_p = lambda c: 100 * float(c[1]) / float(c[0] + c[1])
    labels_p = [i_p(train_count), i_p(val_count), i_p(test_count)]

    all_atoms = len(y_train) + len(y_val) + len(y_test)
    all_pdbs = len(train) + len(validation) + len(test)
    d_p = lambda d: len(d) / all_atoms
    pdb_p = lambda p: len(p) / all_pdbs

    if do_print:
        print('\nDataset info:')
        print(f'Counts: {train_count}, {val_count}, {test_count}')
        print(f'Interaction percentage: {labels_p[0]:.2f}%, {labels_p[1]:.2f}%, {labels_p[2]:.2f}%')
        print(f'Atom distribution: {d_p(y_train):.2f}, {d_p(y_val):.2f}, {d_p(y_test):.2f}')
        print(f'Pdb distribution: {pdb_p(train):.2f}, {pdb_p(validation):.2f}, {pdb_p(test):.2f}')
        print()
        print()
    return labels_p


def feature_importance(net, graphs, model_name, n_graphs=50, n_steps=2, save=False):
    n_graphs = min(n_graphs, len(graphs))
    input_d = dgl.batch([graphs[i] for i in range(n_graphs)])

    # ig = IntegratedGradients(net)
    # ig_attr_test = ig.attribute(input_d.ndata[Constants.NODE_FEATURES_NAME],
    #                             additional_forward_args=(dgl.batch([input_d] * n_steps)),
    #                             target=1, n_steps=n_steps)
    #
    # ig_attr_test_sum = ig_attr_test.detach().numpy().sum(0)
    # ig_attr_test_norm_sum = ig_attr_test_sum / np.linalg.norm(ig_attr_test_sum, ord=1)

    dl = DeepLift(net)
    dl_attr_test = dl.attribute(input_d.ndata[Constants.NODE_FEATURES_NAME],
                                additional_forward_args=(dgl.batch([input_d] * 2)),
                                target=1)

    dl_attr_test_sum = dl_attr_test.detach().numpy().sum(0)
    dl_attr_test_norm_sum = dl_attr_test_sum / np.linalg.norm(dl_attr_test_sum, ord=1)

    index_array = np.argsort(-np.absolute(dl_attr_test_norm_sum))
    n = 25
    names = np.array(Constants.FEATURE_NAMES)[index_array]
    attrs = dl_attr_test_norm_sum[index_array]
    for i in range(int(np.ceil(len(Constants.FEATURE_NAMES) / n))):
        maxi = min((i + 1) * n, len(Constants.FEATURE_NAMES))

        plt.figure()
        plt.barh(names[maxi:i * n:-1], attrs[maxi:i * n:-1])
        plt.title('Feature importance')
        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(Constants.MODELS_PATH, model_name, f'feature_importance_DL_{i}.png'))
        else:
            plt.show()
