from builtins import range
from builtins import object
import numpy as np


class KNearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y.astype(int)  

    def predict(self, X, k=1, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    # Assignment 1
    def compute_distances_two_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train), dtype=np.float64)

        for i in range(num_test):
            x_i = X[i]
            for j in range(num_train):
                diff = x_i - self.X_train[j]
                dists[i, j] = np.linalg.norm(diff, ord=2)

        return dists
    
    # Assignment 2
    def compute_distances_one_loop(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train), dtype=np.float64)

        for i in range(num_test):
            diff = self.X_train - X[i]               
            dists[i, :] = np.sqrt(np.sum(diff * diff, axis=1))  

        return dists

    # Assignment 3
    def compute_distances_no_loops(self, X):
        X_sq = np.sum(X * X, axis=1)
        T_sq = np.sum(self.X_train * self.X_train, axis=1)

        cross = X @ self.X_train.T  

        dist_sq = X_sq[:, None] + T_sq[None, :] - 2.0 * cross

        
        np.maximum(dist_sq, 0.0, out=dist_sq)

        
        dists = np.sqrt(dist_sq, dtype=np.float64)  

        return dists

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test, dtype=int)

        for i in range(num_test):
            nearest_idx = np.argsort(dists[i])[:k]
            nearest_labels = self.y_train[nearest_idx]  

            y_pred[i] = np.argmax(np.bincount(nearest_labels))

        return y_pred
