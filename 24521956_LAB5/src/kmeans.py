import numpy as np


class KMeans:
    def __init__(self, k=3, max_iter=100, tol=0.0001, seed=0):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

        self.centers = None
        self.labels = None
        self.inertia = None
        self.n_iter = 0
        self.history = []

    def init_centers(self, X):
        np.random.seed(self.seed)
        n = X.shape[0]
        index = np.random.choice(n, self.k, replace=False)
        return X[index].copy()

    def calc_distance(self, X, centers):
        distances = np.zeros((X.shape[0], self.k))

        for j in range(self.k):
            diff = X - centers[j]
            distances[:, j] = np.sum(diff * diff, axis=1)

        return distances

    def assign_cluster(self, X, centers):
        distances = self.calc_distance(X, centers)
        labels = np.argmin(distances, axis=1)
        return labels

    def update_centers(self, X, labels, old_centers):
        new_centers = old_centers.copy()

        for j in range(self.k):
            points = X[labels == j]

            if len(points) > 0:
                new_centers[j] = np.mean(points, axis=0)
            else:
                random_id = np.random.randint(0, X.shape[0])
                new_centers[j] = X[random_id]

        return new_centers

    def calc_inertia(self, X, labels, centers):
        total = 0

        for j in range(self.k):
            points = X[labels == j]

            if len(points) > 0:
                diff = points - centers[j]
                total += np.sum(diff * diff)

        return float(total)

    def fit(self, X):
        X = np.array(X, dtype=float)

        centers = self.init_centers(X)
        self.history = []

        for step in range(self.max_iter):
            labels = self.assign_cluster(X, centers)
            new_centers = self.update_centers(X, labels, centers)

            new_labels = self.assign_cluster(X, new_centers)
            inertia = self.calc_inertia(X, new_labels, new_centers)
            self.history.append(inertia)

            move = np.linalg.norm(new_centers - centers)

            centers = new_centers

            if move < self.tol:
                break

        self.centers = centers
        self.labels = self.assign_cluster(X, centers)
        self.inertia = self.calc_inertia(X, self.labels, centers)
        self.n_iter = step + 1

        return self

    def predict(self, X):
        if self.centers is None:
            raise Exception("Model chưa được train.")

        X = np.array(X, dtype=float)
        return self.assign_cluster(X, self.centers)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels

    def get_result(self):
        return {
            "centers": self.centers,
            "labels": self.labels,
            "inertia": self.inertia,
            "n_iter": self.n_iter,
            "history": self.history,
            "seed": self.seed
        }
