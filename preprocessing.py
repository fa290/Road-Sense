import os
import cv2
import random
import shutil

#paths

image_path = "dataset/images"
label_path = "dataset/labels"

output_path = "processed_dataset"

# folders
train_img = os.path.join(output_path,"images/train")
val_img = os.path.join(output_path,"images/val")
test_img = os.path.join(output_path,"images/test")

train_lbl = os.path.join(output_path,"labels/train")
val_lbl = os.path.join(output_path,"labels/val")
test_lbl = os.path.join(output_path,"labels/test")

for p in [train_img,val_img,test_img,train_lbl,val_lbl,test_lbl]:
    os.makedirs(p,exist_ok=True)

# resize images

target_size = (640,640)

for img_name in os.listdir(image_path):

    img_file = os.path.join(image_path,img_name)

    image = cv2.imread(img_file)

    resized = cv2.resize(image,target_size)

    cv2.imwrite(img_file,resized)

print("Images resized")

# convert KITTI to YOLO

class_map = {
    "Car":0,
    "Pedestrian":1,
    "Cyclist":2
}

for label_file in os.listdir(label_path):

    file_path = os.path.join(label_path,label_file)

    new_lines = []

    with open(file_path,"r") as f:
        lines = f.readlines()

    for line in lines:

        parts = line.split()

        obj = parts[0]

        if obj not in class_map:
            continue

        xmin = float(parts[4])
        ymin = float(parts[5])
        xmax = float(parts[6])
        ymax = float(parts[7])

        width = xmax - xmin
        height = ymax - ymin

        x_center = xmin + width/2
        y_center = ymin + height/2

        img_w = 640
        img_h = 640

        x_center /= img_w
        y_center /= img_h
        width /= img_w
        height /= img_h

        cls = class_map[obj]

        new_line = f"{cls} {x_center} {y_center} {width} {height}\n"

        new_lines.append(new_line)

    with open(file_path,"w") as f:
        f.writelines(new_lines)

print("Annotations converted to YOLO")

# split dataset

images = os.listdir(image_path)

random.shuffle(images)

train_split = int(0.7*len(images))
val_split = int(0.2*len(images))

train = images[:train_split]
val = images[train_split:train_split+val_split]
test = images[train_split+val_split:]

# move files

def move_files(files,img_dest,lbl_dest):

    for img in files:

        img_src = os.path.join(image_path,img)
        lbl_src = os.path.join(label_path,img.replace(".png",".txt"))

        shutil.copy(img_src,os.path.join(img_dest,img))

        if os.path.exists(lbl_src):
            shutil.copy(lbl_src,os.path.join(lbl_dest,img.replace(".png",".txt")))

move_files(train,train_img,train_lbl)
move_files(val,val_img,val_lbl)
move_files(test,test_img,test_lbl)

print("Dataset split done")

print("Dataset ready for training")
