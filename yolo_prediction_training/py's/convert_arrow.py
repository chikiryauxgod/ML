import os
from PIL import Image
from datasets import Dataset

full_path = "C:\\Users\\lewac\\.cache\\huggingface\\datasets\\rokmr___pets\\default\\0.0.0\\04eef6a4ebc9d32572aac5b0889b645e0a4ea086\\"
modes_paths = ["pets-train.arrow", "pets-test.arrow"] 

def save_format(dataset, output_dir):
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for i, example in enumerate(dataset):
        try:
            image = example["image"]
            label = example["label"]

            image_path = os.path.join(images_dir, f"{i}.jpg")
            image.save(image_path)

            label_path = os.path.join(labels_dir, f"{i}.txt")
            with open(label_path, "w") as f:
                f.write(f"{label}\n")

            print(f"Saved example {i} successfully.")

        except Exception as e:
            print(f"Error processing example {i}: {e}")

def load_and_process_data():
    for mode in modes_paths:
        
        arrow_file_path = os.path.join(full_path, mode)
        print(f"Loading dataset from {arrow_file_path}")

        dataset = Dataset.from_file(arrow_file_path)

        output_dir = os.path.join("output_dataset", mode.split('.')[0])  
        os.makedirs(output_dir, exist_ok=True)

        save_format(dataset, output_dir)

load_and_process_data()
