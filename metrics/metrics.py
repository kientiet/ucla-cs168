from sklearn.metrics import *

def accuracy(predictions, labels):
    return accuracy_score(labels, predictions)

def f1_score(predictions, labels):
    return f1_score(labels, predictions)

def precision(predictions, labels):
    return precision_score(labels, predictions)

def recall(predictions, labels):
    return recall_score(labels, predictions)

def auc_roc(predictions, labels):
    return roc_auc_score(labels, predictions)