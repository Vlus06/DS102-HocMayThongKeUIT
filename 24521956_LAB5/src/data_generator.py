import numpy as np


class DataGenerator:
    def __init__(self, seed=42):
        self.seed = seed

    def make_data(self, means, covariances, sizes, shuffle=True):
        np.random.seed(self.seed)

        X_list = []
        y_list = []

        for i in range(len(means)):
            points = np.random.multivariate_normal(
                mean=means[i],
                cov=covariances[i],
                size=sizes[i]
            )
            labels = np.full(sizes[i], i)

            X_list.append(points)
            y_list.append(labels)

        X = np.vstack(X_list)
        y = np.concatenate(y_list)

        if shuffle:
            index = np.random.permutation(len(X))
            X = X[index]
            y = y[index]

        return X, y


def get_assignment1_data():
    means = [
        [2, 2],
        [8, 3],
        [3, 6]
    ]

    sigma = [
        [1, 0],
        [0, 1]
    ]

    covariances = [sigma, sigma, sigma]
    sizes = [200, 200, 200]

    generator = DataGenerator(seed=42)
    return generator.make_data(means, covariances, sizes)


def get_assignment2_data():
    means = [
        [2, 2],
        [8, 3],
        [3, 6]
    ]

    sigma = [
        [1, 0],
        [0, 1]
    ]

    covariances = [sigma, sigma, sigma]

    sizes = [1200, 200, 1000]

    generator = DataGenerator(seed=42)
    return generator.make_data(means, covariances, sizes)


def get_assignment3_data():
    means = [
        [2, 2],
        [8, 3],
        [3, 6]
    ]

    sigma1 = [
        [1, 0],
        [0, 1]
    ]

    sigma2 = [
        [10, 0],
        [0, 1]
    ]

    covariances = [sigma1, sigma1, sigma2]
    sizes = [200, 200, 200]

    generator = DataGenerator(seed=42)
    return generator.make_data(means, covariances, sizes)
