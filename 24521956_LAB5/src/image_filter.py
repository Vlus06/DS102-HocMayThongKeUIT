import numpy as np
from PIL import Image

from src.gmm import GMM


class ImageFilterGMM:
    def __init__(self, k=3, seed=42):
        self.k = k
        self.seed = seed
        self.model = None

    def read_image(self, image_path):
        img = Image.open(image_path).convert("RGB")
        return np.array(img)

    def image_to_data(self, image):
        h, w, c = image.shape
        X = image.reshape(h * w, 3)
        X = X / 255.0
        return X

    def choose_background(self, labels, means):
        counts = []

        for i in range(self.k):
            counts.append(np.sum(labels == i))

        counts = np.array(counts, dtype=float)
        counts = counts / np.max(counts)

        green_score = means[:, 1] - 0.5 * (means[:, 0] + means[:, 2])
        green_score = green_score - np.min(green_score)

        if np.max(green_score) > 0:
            green_score = green_score / np.max(green_score)

        score = 0.7 * counts + 0.3 * green_score
        return int(np.argmax(score))

    def filter(self, image_path):
        image = self.read_image(image_path)
        h, w, c = image.shape

        X = self.image_to_data(image)

        self.model = GMM(k=self.k, max_iter=40, tol=0.0001, seed=self.seed)
        self.model.fit(X)

        labels = self.model.labels
        label_image = labels.reshape(h, w)

        bg_id = self.choose_background(labels, self.model.means)

        mask = label_image != bg_id

        result = image.copy()
        result[mask == False] = [255, 255, 255]

        mask_img = mask.astype(np.uint8) * 255

        return {
            "original": image,
            "labels": label_image,
            "mask": mask_img,
            "result": result,
            "background_id": bg_id,
            "component_counts": [int(np.sum(labels == i)) for i in range(self.k)],
            "component_means": self.model.means * 255,
            "log_likelihood": self.model.log_likelihood,
            "n_iter": self.model.n_iter,
            "history": self.model.history
        }
