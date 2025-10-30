import os
import pandas as pd

SEARCH_ROOT_DIR = "data"

def find_jpg_files():
    jpg_file_paths = []
    for current_dir, _subdirs, files in os.walk(SEARCH_ROOT_DIR):
        for file_name in files:
            if file_name.lower().endswith(".jpg"):
                jpg_file_paths.append(os.path.join(current_dir, file_name))
    return jpg_file_paths

def generate_dataframe(jpg_file_paths):
    data = []
    for image_path in jpg_file_paths:
        label = os.path.basename(os.path.dirname(image_path))
        data.append({"image_path": image_path, "label": label})
    return pd.DataFrame(data, columns=["image_path", "label"])

if __name__ == "__main__":
    jpg_file_paths = find_jpg_files()
    df = generate_dataframe(jpg_file_paths)
    print(df.head())
    df.to_csv("annotations.csv", index=False)
