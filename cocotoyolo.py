import os
import json
from pathlib import Path

# D:\GitHub\animals-ij5d2\data\train\_annotations.coco.json - train 
# D:\GitHub\animals-ij5d2\data\valid\_annotations.coco.json - validation
# D:\GitHub\animals-ij5d2\data\test\_annotations.coco.json - test

json_path = r"D:\GitHub\animals-ij5d2\data\test\_annotations.coco.json" 
images_dir = r"D:\GitHub\animals-ij5d2\data\test"
output_dir = r"D:\GitHub\datasets\animals"


os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
os.makedirs(f"{output_dir}/labels/valid", exist_ok=True)
os.makedirs(f"{output_dir}/labels/test", exist_ok=True)

with open(json_path, "r") as f:
    coco_data = json.load(f)


image_sizes = {img["id"]: (img["width"], img["height"]) for img in coco_data["images"]}


for annotation in coco_data["annotations"]:
    image_id = annotation["image_id"]
    category_id = annotation["category_id"]
    bbox = annotation["bbox"]
    width, height = image_sizes[image_id]

    x_center = (bbox[0] + bbox[2] / 2) / width
    y_center = (bbox[1] + bbox[3] / 2) / height
    norm_width = bbox[2] / width
    norm_height = bbox[3] / height

    yolo_label = f"{category_id - 1} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n"
    label_path = f"{output_dir}/labels/test/{image_id}.txt" # train or valid or test
    with open(label_path, "a") as label_file:
        label_file.write(yolo_label)
