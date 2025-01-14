"""
Using torchvision, download celebA dataset into a local folder
Then arrange the folder by splits, into a pseudo class
such that it's compatible with ImageFolder()
"""

import os
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
import blobfile as bf
from PIL import Image
from tqdm import trange

N_IMGS = 11788
BIRDS_DIR_ORIGINAL = "datasets/birds_original"
BIRDS_DIR_NEW = "datasets/birds"


if __name__ == "__main__":
    old_img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), BIRDS_DIR_ORIGINAL)
    new_img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), BIRDS_DIR_NEW)
    os.makedirs(new_img_dir, exist_ok=True)

    names_lines = open(os.path.join(BIRDS_DIR_ORIGINAL, "images.txt")).readlines()
    bboxes_lines = open(os.path.join(BIRDS_DIR_ORIGINAL, "bounding_boxes.txt")).readlines()

    grayscale_imgs_count = 0
    print("Preprocess the images and then save into the new folder...")

    for i in trange(N_IMGS):
        tokens = names_lines[i].split()
        img_idx, img_filename = int(tokens[0]), tokens[1]
        assert img_filename.endswith('.jpg')
        img_raw = np.array(Image.open(os.path.join(BIRDS_DIR_ORIGINAL, "images", img_filename)))
        tokens = bboxes_lines[i].split()
        assert img_idx == int(tokens[0])
        x_bbox, y_bbox, w_bbox, h_bbox = (
            int(float(tokens[1])), 
            int(float(tokens[2])), 
            int(float(tokens[3])), 
            int(float(tokens[4])))
        if img_raw.ndim == 2:
            # strangely, for grayscale image, while loading with transforms.ToTensor() gives 
            # three channels with 1 being the first dimension
            # loading with np.array() leads to only two channels
            img_raw = np.tile(np.expand_dims(img_raw, -1), (1,1,3))
            grayscale_imgs_count +=1 
        img_cropped = img_raw[y_bbox:y_bbox+h_bbox, x_bbox:x_bbox+w_bbox, :]
        img_cropped = Image.fromarray(img_cropped)
        # ensure uniform size here
        img_cropped = img_cropped.resize((224, 224))
        img_filename_new = f'{i+1:06d}.jpg'
        img_path = os.path.join(new_img_dir, img_filename_new)
        assert not os.path.isfile(img_path)
        img_cropped.save(img_path)

    print(f"A total of {grayscale_imgs_count} grayscale bird images")
    print("Script finished!")
