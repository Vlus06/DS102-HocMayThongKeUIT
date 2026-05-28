import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.data_generator import get_assignment2_data
from src.kmeans import KMeans
from src.metrics import clustering_accuracy, count_labels
from src.io_utils import make_folder, save_json
from src.visualize import plot_points, plot_hist


def run():
    X, y_true = get_assignment2_data()

    out_dir = os.path.join(ROOT, "outputs", "assignment_2")
    make_folder(out_dir)

    all_results = []

    for seed in range(10):
        model = KMeans(k=3, max_iter=100, seed=seed)
        model.fit(X)
        all_results.append(model.get_result())

    best = min(all_results, key=lambda r: r["inertia"])
    acc, label_map = clustering_accuracy(y_true, best["labels"], 3)

    plot_points(X, y_true, "Bài 2 - Dữ liệu gốc", os.path.join(out_dir, "data.png"))
    plot_points(X, best["labels"], "Bài 2 - KMeans", os.path.join(out_dir, "kmeans.png"), best["centers"])
    plot_hist([r["inertia"] for r in all_results], "Bài 2 - Inertia qua nhiều seed", os.path.join(out_dir, "inertia.png"))

    summary = {
        "assignment": 2,
        "algorithm": "KMeans",
        "title": "Bài 2 - Kích thước cụm không cân bằng 1200-200-1000",
        "comment": "Dữ liệu bài 2 có kích thước 1200, 200 và 1000 nên các cụm không cân bằng. KMeans vẫn có thể phân cụm tốt khi các cụm tách biệt, nhưng cụm nhỏ 200 điểm dễ bị ảnh hưởng bởi cách khởi tạo centroid hơn.",
        "accuracy": acc,
        "label_map": label_map,
        "best_seed": best["seed"],
        "best_inertia": best["inertia"],
        "n_iter": best["n_iter"],
        "cluster_counts": count_labels(best["labels"], 3),
        "centers": best["centers"],
        "figures": {
            "data": os.path.join(out_dir, "data.png"),
            "kmeans": os.path.join(out_dir, "kmeans.png"),
            "inertia": os.path.join(out_dir, "inertia.png")
        }
    }

    save_json(summary, os.path.join(out_dir, "summary.json"))
    print(summary)
    return summary


if __name__ == "__main__":
    run()
