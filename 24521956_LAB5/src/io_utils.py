import os
import json
import numpy as np


def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def to_normal_object(x):
    if isinstance(x, np.ndarray):
        return x.tolist()

    if isinstance(x, np.integer):
        return int(x)

    if isinstance(x, np.floating):
        return float(x)

    if isinstance(x, dict):
        new_dict = {}
        for key in x:
            new_dict[key] = to_normal_object(x[key])
        return new_dict

    if isinstance(x, list):
        return [to_normal_object(item) for item in x]

    if isinstance(x, tuple):
        return list(x)

    return x


def save_json(data, path):
    data = to_normal_object(data)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
