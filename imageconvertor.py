import os
import shutil


source_dir = r"D:\GitHub\animals-ij5d2\data\train"
target_dir = r"D:\GitHub\datasets\animals\images\train"


os.makedirs(target_dir, exist_ok=True)

# Получаем список всех файлов в исходной папке
files = [f for f in os.listdir(source_dir) if f.endswith(".jpg")]

# Перемещаем файлы с индексацией
for idx, file in enumerate(files):
    # Определяем новый путь и имя файла
    new_name = f"{idx}.jpg"
    source_path = os.path.join(source_dir, file)
    target_path = os.path.join(target_dir, new_name)
    
    # Перемещаем файл
    shutil.move(source_path, target_path)

print(f"Файлы успешно перемещены в {target_dir} с новой индексацией.")
