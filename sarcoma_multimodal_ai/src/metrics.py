import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

def compute_metrics(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    acc = accuracy_score(y_true, np.round(y_pred))
    return auc, acc
