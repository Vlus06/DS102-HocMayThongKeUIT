import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.data_generator import get_assignment3_data
from src.kmeans import KMeans
from src.gmm import GMM
from src.metrics import clustering_accuracy, count_labels
from src.io_utils import make_folder, save_json
from src.visualize import plot_points, plot_line


def run():
    X, y_true = get_assignment3_data()

    out_dir = os.path.join(ROOT, "outputs", "assignment_4")
    make_folder(out_dir)

    kmeans = KMeans(k=3, max_iter=100, seed=7)
    kmeans.fit(X)

    gmm = GMM(k=3, max_iter=100, seed=7)
    gmm.fit(X)

    acc_kmeans, map_kmeans = clustering_accuracy(y_true, kmeans.labels, 3)
    acc_gmm, map_gmm = clustering_accuracy(y_true, gmm.labels, 3)

    plot_points(X, y_true, "Bài 4 - Dữ liệu gốc", os.path.join(out_dir, "data.png"))
    plot_points(X, kmeans.labels, "Bài 4 - KMeans", os.path.join(out_dir, "kmeans.png"), kmeans.centers)
    plot_points(X, gmm.labels, "Bài 4 - GMM", os.path.join(out_dir, "gmm.png"), gmm.means)
    plot_line(gmm.history, "Bài 4 - Log-likelihood của GMM", os.path.join(out_dir, "log_likelihood.png"), "Log-likelihood")

    summary = {
        "assignment": 4,
        "algorithm": "GMM",
        "title": "Bài 4 - Gaussian Mixture Model",
        "comment": "GMM mô hình hóa cả mean và covariance nên phù hợp hơn KMeans khi cụm có dạng elip.",
        "kmeans_accuracy": acc_kmeans,
        "gmm_accuracy": acc_gmm,
        "kmeans_inertia": kmeans.inertia,
        "gmm_log_likelihood": gmm.log_likelihood,
        "gmm_n_iter": gmm.n_iter,
        "kmeans_counts": count_labels(kmeans.labels, 3),
        "gmm_counts": count_labels(gmm.labels, 3),
        "gmm_weights": gmm.weights,
        "gmm_means": gmm.means,
        "gmm_covariances": gmm.covs,
        "figures": {
            "data": os.path.join(out_dir, "data.png"),
            "kmeans": os.path.join(out_dir, "kmeans.png"),
            "gmm": os.path.join(out_dir, "gmm.png"),
            "log_likelihood": os.path.join(out_dir, "log_likelihood.png")
        }
    }

    save_json(summary, os.path.join(out_dir, "summary.json"))
    print(summary)
    return summary


if __name__ == "__main__":
    run()
