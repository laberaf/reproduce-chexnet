import os
import json
import shutil
from PIL import Image
from cxr_dataset import CXRDataset

def create_label_folders(labels, base_path):
    for label in labels:
        label_path = os.path.join(base_path, label)
        if not os.path.exists(label_path):
            os.makedirs(label_path)

def move_images_to_label_folders(dataset, base_path):
    for _, labels, img_name in dataset:
        for label_idx, label_value in enumerate(labels):
            if label_value:
                label_name = dataset.PRED_LABEL[label_idx]
                src_path = os.path.join(dataset.path_to_images, img_name)
                dst_path = os.path.join(base_path, label_name, img_name)
                shutil.copy(src_path, dst_path)
                create_json_file(dst_path, label_name, img_name)

def create_json_file(image_path, label, img_name):
    json_data = {
        "id": label,
        "title": label,
        "task": "Please intepret the CXR",
        "exemplar": "true",
        "category": "Chest X-rays",
        "finding": label,
        "description": "Some description of " + label.lower(),
        "image": img_name,
        "sections": [
            {
                "title": label,
                "tab_label": "A",
                "image": img_name,
                "descirption": label + " can indicate .... etc"
            }
        ]
    }

    json_path = os.path.join(os.path.dirname(image_path), img_name.split('.')[0] + ".json")
    with open(json_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)

if __name__ == "__main__":
    path_to_images = "starter_images/"
    sorted_images_path = "sorted_images/"
    fold = "train"

    dataset = CXRDataset(path_to_images, fold)

    create_label_folders(dataset.PRED_LABEL, sorted_images_path)
    move_images_to_label_folders(dataset, sorted_images_path)
