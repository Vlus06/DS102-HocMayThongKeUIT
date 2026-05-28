import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_points(X, labels, title, save_path, centers=None):
    plt.figure(figsize=(7, 5))

    labels_unique = sorted(list(set(labels)))

    for label in labels_unique:
        points = X[labels == label]
        plt.scatter(points[:, 0], points[:, 1], s=18, alpha=0.7, label="Cụm " + str(label))

    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], s=170, marker="x", label="Tâm cụm")

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_line(values, title, save_path, ylabel):
    plt.figure(figsize=(7, 5))
    plt.plot(values)
    plt.title(title)
    plt.xlabel("Vòng lặp")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_hist(values, title, save_path):
    plt.figure(figsize=(7, 5))
    plt.hist(values)
    plt.title(title)
    plt.xlabel("Inertia")
    plt.ylabel("Số lần chạy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_image(array, save_path):
    array = np.array(array)

    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    img = Image.fromarray(array)
    img.save(save_path)
