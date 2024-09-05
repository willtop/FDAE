"""
Using torchvision, download celebA dataset into a local folder
Then arrange the folder by splits, into a pseudo class
such that it's compatible with ImageFolder()
"""

import os
import numpy as np
import datetime
from tqdm import trange
from PIL import Image
import h5py


SHAPES3D_SOURCE_DIR = "data/"
SHAPES3D_TARGET_DIR = "datasets/shapes3d/"


if __name__ == "__main__":
    # load the shapes3D dataset from the downloaded h5 file
    datafile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                SHAPES3D_SOURCE_DIR, 
                                "shapes3d.h5")

    print(f"Loading shapes3d data from {datafile_path}...")
    shapes3d_data = h5py.File(datafile_path, 'r')
    shapes3d_imgs = shapes3d_data['images'][()]
    n_imgs = shapes3d_imgs.shape[0]
    assert np.shape(shapes3d_imgs) == (n_imgs, 64, 64, 3)
    assert n_imgs == 480_000
    n_train_imgs = 400_000
    
    
    # arrange the images by split
    img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           SHAPES3D_TARGET_DIR)
    # since this is to be used for unsupervised deepcluster training
    # just have a pseudo class "cls1" created
    os.makedirs(img_dir, exist_ok=True)

    # save the images into corresponding split folder
    print(f"Saving training Shapes3D images...")
    for i in trange(n_train_imgs):
        img = Image.fromarray(shapes3d_imgs[i])
        img_filename = f'{i+1:06d}.jpg'
        img_path = os.path.join(img_dir, img_filename)
        assert not os.path.isfile(img_path)
        img.save(img_path)

    print("Script finished for Shapes3D!")
