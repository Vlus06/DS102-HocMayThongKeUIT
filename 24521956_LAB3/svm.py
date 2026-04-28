import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


class SVM:
    def __init__(self, C=1.0, lr=1e-4, n_iterations=100):
        self.C = C
        self.lr = lr
        self.n_iterations = n_iterations

        self.W = None
        self.b = None
        self.losses = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, dim = X.shape
        self.W = np.zeros(dim)
        self.b = 0.0

        for epoch in tqdm(range(self.n_iterations)):
            indices = np.random.permutation(N)

            for ith in indices:
                x_i = X[ith]
                y_i = y[ith]

                y_pred = self.predict(x_i)

                if y_i * y_pred >= 1:
                    dW = self.W
                    db = 0.0
                else:
                    dW = self.W - self.C * y_i * x_i
                    db = -self.C * y_i

                self.W -= self.lr * dW
                self.b -= self.lr * db

            y_pred = self.predict(X)
            loss = self.loss_fn(y, y_pred)
            self.losses.append(loss)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W + self.b

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.predict(X) >= 0, 1, -1)

    def loss_fn(self, y :np.ndarray, y_hat: np.ndarray) -> float:
        hinge_sum = np.sum(np.maximum(0.0, 1.0 - y * y_hat))
        return 0.5 * np.dot(self.W, self.W) + self.C * hinge_sum

    def get_metrics(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred = self.predict_labels(X)

        return {
            "Accuracy":  np.mean(y_pred == y),
            "Precision": precision_score(y, y_pred, zero_division=0),
            "Recall":    recall_score(y, y_pred, zero_division=0),
            "F1":        f1_score(y, y_pred, zero_division=0),
        }
    def plot_loss(self):
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, self.n_iterations + 1), self.losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Objective")
        plt.grid(True)
        plt.tight_layout()
        plt.show()