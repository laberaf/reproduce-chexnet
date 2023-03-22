import os
import json
import shutil
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from cxr_dataset import CXRDataset
from torchvision import transforms
from matplotlib.pyplot import imsave
import matplotlib.pyplot as plt

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
                "image": label + "_Overlay_" + img_name,
                "descirption": label + " can indicate .... etc"
            }
        ]
    }

    json_path = os.path.join(os.path.dirname(image_path), img_name.split('.')[0] + ".json")
    with open(json_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)

def generate_overlay_image(dataset, model, label):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1)
    
    show_next(iter(dataloader), model, label)

def show_next(dataloader, model, LABEL):
    with torch.no_grad():
        for data, target, filename in dataloader:
            data = data.cpu()
            output = model(data)
            idx = model.PRED_LABEL.index(LABEL)
            output_np = output.cpu().numpy()[0][idx]
            img = Image.open(os.path.join(dataloader.dataset.path_to_images, filename[0]))

            hmap_np = np.maximum(output_np, 0)
            hmap_np = hmap_np / np.max(hmap_np)
            hmap = sns.heatmap(hmap_np, cmap='jet', cbar=False, alpha=0.5, xticklabels=False, yticklabels=False)
            plt.axis('off')

            plt.imshow(img, cmap='gray', aspect='auto')
            plt.title(LABEL + " Overlayed Heatmap")
            plt.savefig(os.path.join(dataloader.dataset.path_to_images, LABEL + "_Overlay_" + filename[0]), bbox_inches='tight', pad_inches=0)
            imsave(os.path.join(dataloader.dataset.path_to_images, LABEL + "_Overlay_" + filename[0]), hmap.get_array())
            plt.close()

if __name__ == "__main__":
    path_to_images = "starter_images/"
    sorted_images_path = "sorted_images/"
    fold = "train"

    dataset = CXRDataset(path_to_images, fold)

    create_label_folders(dataset.PRED_LABEL, sorted_images_path)

    path_to_model = "path/to/your/model.pth"  # Set the path to your model
    model_checkpoint = torch.load(path_to_model, map_location=lambda storage, loc: storage)
    model = model_checkpoint['model']
    model.cpu()

    for label in dataset.PRED_LABEL:
        generate_overlay_image(dataset, model, label)

    move_images_to_label_folders(dataset, sorted_images_path)
