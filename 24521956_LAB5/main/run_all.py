import os
import sys
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.io_utils import make_folder, save_json
from main_assignment1 import run as run1
from main_assignment2 import run as run2
from main_assignment3 import run as run3
from main_assignment4 import run as run4
from main_assignment5 import run as run5


def main():
    out_dir = os.path.join(ROOT, "outputs")
    make_folder(out_dir)

    summaries = []

    summaries.append(run1())
    summaries.append(run2())
    summaries.append(run3())
    summaries.append(run4())
    summaries.append(run5())

    save_json(summaries, os.path.join(out_dir, "all_summaries.json"))

    csv_path = os.path.join(out_dir, "summary.csv")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["assignment", "algorithm", "title", "main_score", "comment"])

        for s in summaries:
            if "accuracy" in s:
                score = s["accuracy"]
            elif "gmm_accuracy" in s:
                score = s["gmm_accuracy"]
            else:
                score = s["log_likelihood"]

            writer.writerow([
                s["assignment"],
                s["algorithm"],
                s["title"],
                score,
                s["comment"]
            ])

    print("Đã chạy xong 5 bài.")
    print("Kết quả nằm trong thư mục outputs.")


if __name__ == "__main__":
    main()
