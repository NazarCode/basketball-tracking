import os
import shutil

# Paths
dataset_dir = "dataset_ball1"  # root of your dataset
output_dir = "ball_only_dataset"
ball_class_id = 0  # basketball

splits = ["train", "valid", "test"]  

for split in splits:
    os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

    images_dir = os.path.join(dataset_dir, split, "images")
    labels_dir = os.path.join(dataset_dir, split, "labels")

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue

        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, "r") as f:
            lines = f.readlines()

        # Keep only basketball (class 0)
        ball_lines = [line for line in lines if line.strip().split()[0] == str(ball_class_id)]
        if not ball_lines:
            continue  # skip images with no basketball

        # Copy image
        img_file = label_file.replace(".txt", ".jpg")  # adjust if images are .png
        src_img = os.path.join(images_dir, img_file)
        dst_img = os.path.join(output_dir, split, "images", img_file)
        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)

        # Write filtered labels
        dst_label = os.path.join(output_dir, split, "labels", label_file)
        with open(dst_label, "w") as f:
            f.writelines(ball_lines)

print("Ball-only dataset with train/val/test created at:", output_dir)
