import os
import cv2
import hashlib
from PIL import Image
from tqdm import tqdm


dataset_root = r"D:\DEPI\Final broject\data_quality\data\raw\KITTI\training" 

image_folder = os.path.join(dataset_root, "image_2")
label_folder = os.path.join(dataset_root, "label_2")
image_extensions = (".png",)

def validate_kitti_quality():
    corrupted_images = []
    missing_labels = []
    invalid_labels = []
    duplicates = []
    hashes = {}
    
    print("--- Starting KITTI Data Quality Audit ---")

    #  Step 1 - Validate Images and Labels
    if not os.path.exists(image_folder):
        print(f"Error: Image folder not found at {image_folder}")
        return

    # List all PNG images in the image folder
    images = [f for f in os.listdir(image_folder) if f.lower().endswith(image_extensions)]
    
    for img_name in tqdm(images):
        img_path = os.path.join(image_folder, img_name)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(label_folder, label_name)

        # Detect corrupted images by trying to open them with PIL and read with OpenCV
        try:
            with Image.open(img_path) as img:
                img.verify()
            cv_img = cv2.imread(img_path)
            if cv_img is None: raise Exception
            img_h, img_w = cv_img.shape[:2]
        except:
            corrupted_images.append(img_name)
            continue

        # Check for exact duplicates using MD5 hash
        with open(img_path, "rb") as f:
            img_hash = hashlib.md5(f.read()).hexdigest()
        if img_hash in hashes:
            duplicates.append((img_name, hashes[img_hash]))
            continue
        hashes[img_hash] = img_name

        # Check if corresponding label file exists
        if not os.path.exists(label_path):
            missing_labels.append(img_name)
            continue

        # step 2 - Validate Bounding Boxes in Label Files
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 15:
                        invalid_labels.append(f"{img_name} (Missing Columns)")
                        break
                    
                    left, top, right, bottom = map(float, parts[4:8])
                    
                    if left < 0 or top < 0 or right > img_w or bottom > img_h or left >= right or top >= bottom:
                        invalid_labels.append(img_name)
                        break
        except Exception:
            invalid_labels.append(img_name)
       
       # Step 3 - Save Clean Dataset Index      
    clean_samples = [
        img for img in images
        if img not in corrupted_images
         and img not in missing_labels
        and img not in invalid_labels
    ]
    with open("clean_index.txt", "w") as f:
        for img in clean_samples:
            f.write(os.path.splitext(img)[0] + "\n")

    print(f"Clean dataset index saved with {len(clean_samples)} samples.")

    # Step 4 - Generate Report
    print("\n" + "="*40)
    print("      KITTI QUALITY ASSESSMENT REPORT")
    print("="*40)
    print(f"Total PNG Images Found:     {len(images)}")
    print(f"Corrupted Images:           {len(corrupted_images)}")
    print(f"Missing Label Files:        {len(missing_labels)}")
    print(f"Invalid/Out-of-bounds Bboxes: {len(invalid_labels)}")
    print(f"Exact Duplicates Found:     {len(duplicates)}")
    print(f"Final Clean Images:         {len(clean_samples)}")
    print("="*40)
    
    if not any([corrupted_images, missing_labels, invalid_labels, duplicates]):
        print("Result: Dataset is CLEAN and ready for preprocessing!")
    else:
        print("Warning: Issues found. Please check the lists.")

    return {
        "corrupted": corrupted_images,
        "missing": missing_labels,
        "invalid": invalid_labels,
        "duplicates": duplicates
    }

if __name__ == "__main__":
    results = validate_kitti_quality()