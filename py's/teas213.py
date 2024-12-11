from datasets import Dataset

def load_arrow_dataset(arrow_file_path):
    """Загружает .arrow файл в объект Dataset."""
    return Dataset.from_file(arrow_file_path)

arrow_file_path = "C:\\Users\\lewac\\.cache\\huggingface\\datasets\\rokmr___pets\\default\\0.0.0\\04eef6a4ebc9d32572aac5b0889b645e0a4ea086\\pets-test.arrow"
dataset = load_arrow_dataset(arrow_file_path)
print("Columns in dataset:", dataset.column_names)
for example in dataset:
    print("Example:", example)
    break
