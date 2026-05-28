import numpy as np
from src.kmeans import KMeans


class GMM:
    def __init__(self, k=3, max_iter=100, tol=0.0001, seed=0):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

        self.weights = None
        self.means = None
        self.covs = None
        self.responsibility = None
        self.labels = None
        self.log_likelihood = None
        self.n_iter = 0
        self.history = []

    def init_params(self, X):
        kmeans = KMeans(k=self.k, max_iter=50, seed=self.seed)
        kmeans.fit(X)

        labels = kmeans.labels
        centers = kmeans.centers

        n = X.shape[0]
        d = X.shape[1]

        self.weights = np.zeros(self.k)
        self.means = centers.copy()
        self.covs = np.zeros((self.k, d, d))

        for j in range(self.k):
            points = X[labels == j]

            if len(points) == 0:
                self.weights[j] = 1.0 / self.k
                self.covs[j] = np.eye(d)
            else:
                self.weights[j] = len(points) / n
                diff = points - self.means[j]
                self.covs[j] = np.dot(diff.T, diff) / len(points)
                self.covs[j] += 0.000001 * np.eye(d)

    def gaussian_pdf(self, X, mean, cov):
        d = X.shape[1]

        cov = cov + 0.000001 * np.eye(d)

        det = np.linalg.det(cov)
        if det <= 0:
            det = 0.000001

        inv = np.linalg.pinv(cov)

        diff = X - mean
        power = np.sum(np.dot(diff, inv) * diff, axis=1)

        a = 1.0 / np.sqrt(((2 * np.pi) ** d) * det)
        b = np.exp(-0.5 * power)

        return a * b

    def e_step(self, X):
        n = X.shape[0]
        prob = np.zeros((n, self.k))

        for j in range(self.k):
            prob[:, j] = self.weights[j] * self.gaussian_pdf(X, self.means[j], self.covs[j])

        total_prob = np.sum(prob, axis=1).reshape(-1, 1)
        total_prob[total_prob == 0] = 0.0000000001

        responsibility = prob / total_prob
        log_likelihood = np.sum(np.log(total_prob))

        return responsibility, float(log_likelihood)

    def m_step(self, X):
        n = X.shape[0]
        d = X.shape[1]

        Nk = np.sum(self.responsibility, axis=0)

        for j in range(self.k):
            if Nk[j] == 0:
                Nk[j] = 0.0000000001

            self.weights[j] = Nk[j] / n
            self.means[j] = np.sum(self.responsibility[:, j].reshape(-1, 1) * X, axis=0) / Nk[j]

            diff = X - self.means[j]
            self.covs[j] = np.dot((self.responsibility[:, j].reshape(-1, 1) * diff).T, diff) / Nk[j]
            self.covs[j] += 0.000001 * np.eye(d)

    def fit(self, X):
        X = np.array(X, dtype=float)

        self.init_params(X)
        self.history = []

        old_log = None

        for step in range(self.max_iter):
            self.responsibility, log_likelihood = self.e_step(X)
            self.m_step(X)

            self.history.append(log_likelihood)

            if old_log is not None:
                if abs(log_likelihood - old_log) < self.tol:
                    break

            old_log = log_likelihood

        self.responsibility, self.log_likelihood = self.e_step(X)
        self.labels = np.argmax(self.responsibility, axis=1)
        self.n_iter = step + 1

        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        responsibility, value = self.e_step(X)
        return np.argmax(responsibility, axis=1)

    def get_result(self):
        return {
            "weights": self.weights,
            "means": self.means,
            "covariances": self.covs,
            "labels": self.labels,
            "log_likelihood": self.log_likelihood,
            "n_iter": self.n_iter,
            "history": self.history,
            "seed": self.seed
        }
