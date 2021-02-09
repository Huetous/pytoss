import numpy as np
from collections import Counter

def euc_dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

        pass

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        dists = [euc_dist(x, x_train) for x_train in self.X_train]

        k_indices = np.argsort(dists)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]

        return Counter(k_labels).most_common(1)[0][0]
