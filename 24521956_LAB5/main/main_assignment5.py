import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.image_filter import ImageFilterGMM
from src.io_utils import make_folder, save_json
from src.visualize import save_image, plot_line


def run():
    image_path = os.path.join(ROOT, "data", "cow.jpg")

    out_dir = os.path.join(ROOT, "outputs", "assignment_5")
    make_folder(out_dir)

    model = ImageFilterGMM(k=3, seed=42)
    result = model.filter(image_path)

    save_image(result["original"], os.path.join(out_dir, "original.png"))
    save_image(result["mask"], os.path.join(out_dir, "mask.png"))
    save_image(result["result"], os.path.join(out_dir, "foreground.png"))
    plot_line(result["history"], "Bài 5 - Log-likelihood GMM trên ảnh", os.path.join(out_dir, "log_likelihood.png"), "Log-likelihood")

    summary = {
        "assignment": 5,
        "algorithm": "GMM image filtering",
        "title": "Bài 5 - Lọc nền ảnh bằng GMM",
        "comment": "Mỗi pixel được xem là một điểm dữ liệu RGB. GMM gom các pixel thành nhiều nhóm màu, sau đó chọn nhóm nền dựa trên số lượng pixel và độ xanh.",
        "background_id": result["background_id"],
        "component_counts": result["component_counts"],
        "component_means": result["component_means"],
        "log_likelihood": result["log_likelihood"],
        "n_iter": result["n_iter"],
        "figures": {
            "original": os.path.join(out_dir, "original.png"),
            "mask": os.path.join(out_dir, "mask.png"),
            "foreground": os.path.join(out_dir, "foreground.png"),
            "log_likelihood": os.path.join(out_dir, "log_likelihood.png")
        }
    }

    save_json(summary, os.path.join(out_dir, "summary.json"))
    print(summary)
    return summary


if __name__ == "__main__":
    run()
