"""
Using torchvision, download celebA dataset into a local folder
Then arrange the folder by splits, into a pseudo class
such that it's compatible with ImageFolder()
"""

import os
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
import tensorflow_datasets as tfds
from PIL import Image

NORB_DIR = "datasets/norb"


if __name__ == "__main__":
    new_img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), NORB_DIR)
    os.makedirs(new_img_dir, exist_ok=True)
    # load the NORB using tensorflow dataset
    # training set
    ds_tf_train = tfds.as_numpy(
                    tfds.load('smallnorb', 
                              data_dir=NORB_DIR,
                              split='train', 
                              shuffle_files=False))
    
    n_train_imgs = 24_300
    # copy the images into the dataset folder
    print("Copying over training NORB images...")
    for i, sample in enumerate(tqdm(ds_tf_train)):
        img_raw = sample['image']
        # tile one channel to three channels here
        # since otherwise the Image.fromarray() reports error, understandably
        img_raw = np.tile(img_raw, (1,1,3))
        img = Image.fromarray(img_raw)
        img_filename = f'{i:06d}.jpg'
        img_path = os.path.join(new_img_dir, img_filename)
        assert not os.path.isfile(img_path)
        img.save(img_path)

    print("Script finished!")
