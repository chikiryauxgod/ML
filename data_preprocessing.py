import os
import shutil

# Пути к изображениям и меткам
images_dir = r"D:\GitHub\animals-ij5d2\data\test"
output_images_dir = r"D:\GitHub\datasets\animals\images"
labels_dir = r"D:\GitHub\datasets\animals\labels"

# Создание директорий
os.makedirs(f"{output_images_dir}/train", exist_ok=True)
os.makedirs(f"{output_images_dir}/val", exist_ok=True)

# Перенос изображений
for subset in ["train", "val"]:
    label_files = os.listdir(f"{labels_dir}/{subset}")
    for label_file in label_files:
        # Имя изображения совпадает с именем файла аннотации (без расширения)
        image_id = os.path.splitext(label_file)[0]
        for ext in [".jpg", ".png"]:  # Попробуйте несколько форматов
            image_path = os.path.join(images_dir, f"{image_id}{ext}")
            if os.path.exists(image_path):
                shutil.copy(image_path, f"{output_images_dir}/{subset}")
                break
