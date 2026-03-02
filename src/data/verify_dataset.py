import os

# -----------------------------
# Configuration
# -----------------------------
dataset_root = "data/raw/KITTI/training"
image_folder = os.path.join(dataset_root, "image_2")
label_folder = os.path.join(dataset_root, "label_2")
image_extensions = (".png",)
label_extensions = (".txt",) 

# -----------------------------
# Check folder existence
# -----------------------------
for folder in [image_folder, label_folder]:
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

# -----------------------------
# Count images and labels
# -----------------------------
image_files = [f for f in os.listdir(image_folder) if f.endswith(image_extensions)]
label_files = [f for f in os.listdir(label_folder) if f.endswith(label_extensions)]

num_images = len(image_files)
num_labels = len(label_files)

print(f"Number of images: {num_images}")
print(f"Number of labels: {num_labels}")

if num_images != num_labels:
    print("Warning: Number of images and labels do not match!")
else:
    print("Image and label counts match.")

# -----------------------------
# Check file readability
# -----------------------------
corrupted_files = []

for folder_path, files in [(image_folder, image_files), (label_folder, label_files)]:
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, "rb") as f:
                f.read(1024)  # read first 1KB to test
        except Exception:
            corrupted_files.append(file_path)

if corrupted_files:
    print("Corrupted/unreadable files found:")
    for f in corrupted_files:
        print(f" - {f}")
else:
    print("All files are readable and not corrupted.")