from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def cal_accuracy(y_true, y_pred):
    """准确率"""
    return accuracy_score(y_true, y_pred)


def cal_precision(y_ture, y_pred):
    """精确率"""
    return precision_score(y_ture, y_pred)


def cal_recall(y_true, y_pred):
    """召回率"""
    return recall_score(y_true, y_pred)


def cal_f1(y_true, y_pred):
    """f1值"""
    return f1_score(y_true, y_pred)


def cal_common_metrics(y_true, y_pred):
    acc = cal_accuracy(y_true, y_pred)
    p = cal_precision(y_true, y_pred)
    r = cal_recall(y_true, y_pred)
    f = cal_f1(y_true, y_pred)
    print(f"Test Results:    Accuracy: {acc:.4f},  Precision: {p:.4f},  Recall: {r:.4f},  F1: {f:.4f}")
    return acc, p, r, f

