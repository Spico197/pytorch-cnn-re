import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score, classification_report

from data_helper import rel2id, id2rel


def evaluate(model, test_loader, prc=False, report=False):
    acc = precision = recall = f1_micro = f1_macro = 0.0

    y_pred = np.array([]).reshape((1, -1))
    y_true = np.array([]).reshape((1, -1))
    y_scores = np.array([]).reshape((1, -1))
    with torch.no_grad():
        for data in test_loader:
            tokens, pos1, pos2, labels = data
            outputs = model(tokens, pos1, pos2)
            scores, predicted = torch.max(outputs.data, 1)

            predicted = predicted.numpy().reshape((1, -1))
            labels = labels.numpy().reshape((1, -1))
            scores = scores.numpy().reshape((1, -1))

            y_pred = np.concatenate([y_pred, predicted], axis=-1)
            y_true = np.concatenate([y_true, labels], axis=-1)
            y_scores = np.concatenate([y_scores, scores], axis=-1)
        
        acc = accuracy_score(y_true[0], y_pred[0])
        precision = precision_score(y_true[0], y_pred[0], average='micro')
        recall = recall_score(y_true[0], y_pred[0], average='micro')
        f1_micro = f1_score(y_true[0], y_pred[0], average='micro')
        f1_macro = f1_score(y_true[0], y_pred[0], average='macro')

    if report:
        ids = sorted([int(key) for key in id2rel.keys()])
        names = [id2rel[key] for key in ids]
        print(classification_report(y_true[0], y_pred[0], digits=4, labels=ids, target_names=names))

    if prc:
        y_true_prc = (y_true[0] == y_pred[0])
        return acc, precision, recall, f1_micro, f1_macro, precision_recall_curve(y_true_prc, y_scores[0])
    else:
        return acc, precision, recall, f1_micro, f1_macro
