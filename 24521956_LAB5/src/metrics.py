import itertools
import numpy as np


def count_labels(labels, k):
    counts = []

    for i in range(k):
        counts.append(int(np.sum(labels == i)))

    return counts


def clustering_accuracy(y_true, y_pred, k):
    best_acc = 0
    best_map = None

    for p in itertools.permutations(range(k)):
        new_pred = np.zeros_like(y_pred)

        for old_label in range(k):
            new_pred[y_pred == old_label] = p[old_label]

        acc = np.mean(new_pred == y_true)

        if acc > best_acc:
            best_acc = acc
            best_map = p

    return float(best_acc), best_map
