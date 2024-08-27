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

DSTYPE = "Toy"
# DSTYPE = "Complex"

MPI3D_SOURCE_DIR = "data/"
if DSTYPE == "Toy":
    MPI3D_TARGET_DIR = "datasets/mpi3d_toy/"
else:
    MPI3D_TARGET_DIR = "datasets/mpi3d_real_complex/"

if __name__ == "__main__":
    # load the MPI3D dataset from the downloaded npz file
    if DSTYPE == "Toy":
        datafile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 MPI3D_SOURCE_DIR, 
                                 "mpi3d_toy.npz")
    else:
        datafile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 MPI3D_SOURCE_DIR, 
                                 "real3d_complicated_shapes_ordered.npz")
    print(f"Loading mpi3d data from {datafile_path}...")
    start_time = datetime.datetime.now().replace(microsecond=0)
    mpi3d_data = np.load(datafile_path)['images']
    n_imgs = mpi3d_data.shape[0]
    if DSTYPE == "Toy":
        assert n_imgs == 1_036_800
        n_train_imgs = 1_000_000
    else: 
        assert n_imgs == 460_800
        n_train_imgs = 400_000
    end_time = datetime.datetime.now().replace(microsecond=0)
    print(f"[MPI3D {DSTYPE}] data loaded, took time: {end_time-start_time}.")
    
    
    # arrange the images by split
    img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           MPI3D_TARGET_DIR)
    # since this is to be used for unsupervised deepcluster training
    # just have a pseudo class "cls1" created
    os.makedirs(img_dir)

    # save the images into corresponding split folder
    print(f"Saving training MPI3D {DSTYPE} images...")
    for i in trange(n_train_imgs):
        img = Image.fromarray(mpi3d_data[i])
        img_filename = f'{i+1:06d}.jpg'
        img_path = os.path.join(img_dir, img_filename)
        assert not os.path.isfile(img_path)
        img.save(img_path)

    print(f"Script finished for MPI3D {DSTYPE}!")