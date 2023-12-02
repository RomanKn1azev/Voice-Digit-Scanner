import torch
import torch.nn.functional as F


from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score


def create_accuracy_metric():
    def accuracy(y_true, y_pred):
        _, prediction = torch.max(y_pred, 1)
        correct_prediction += (prediction == y_true).sum().item()
        total_prediction += prediction.shape[0]
        
        return correct_prediction / total_prediction
    return accuracy


def create_f1_metric():
    def f1(y_true, y_pred):
        y_pred_classes = torch.argmax(y_pred, dim=1)
        y_pred_numpy = y_pred_classes.numpy()
        y_true_numpy = y_true.numpy()
        return f1_score(y_true_numpy, y_pred_numpy)
    return f1


def create_roc_auc_metric():
    def roc_auc(y_true, y_pred):
        ...
        return roc_auc_score()
    return roc_auc


def create_precision_metric():
    def precision(y_true, y_pred):
        ...
        return precision_score()
    return precision


def create_recall_metric():
    def recall(y_true, y_pred):
        ...
        return recall_score()
    return recall