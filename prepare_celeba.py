"""
Using torchvision, download celebA dataset into a local folder
Then arrange the folder by splits, into a pseudo class
such that it's compatible with ImageFolder()
"""

import os
from tqdm import trange
import torchvision.transforms as transforms
from torchvision.datasets.celeba import CelebA
from PIL import Image

CELEBA_DIR = "datasets/celeba"

celeba_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToPILImage()
            ])

if __name__ == "__main__":
    new_img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), CELEBA_DIR)
    os.makedirs(new_img_dir, exist_ok=True)
    # load the CelebA using torchvision into the folder "celeba_dataset"
    # no need to specify the label type or transformation here
    _ = CelebA(CELEBA_DIR, 
                split="all",
                download=True)
    
    # arrange the images by split
    orig_img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           CELEBA_DIR,
                           "celeba",
                           "img_align_celeba")

    n_imgs = 202_599
    n_train_imgs = 162_770
    # move the images into the dataset folder
    print("Moving training CelebA images...")
    # the downloaded celeba images are named with idexing from 1
    for i in trange(1, n_train_imgs+1):
        img_filename = f'{i:06d}.jpg'
        old_img_loc = os.path.join(orig_img_dir, img_filename)
        new_img_loc = os.path.join(new_img_dir, img_filename)
        orig_img = Image.open(old_img_loc)
        new_img = celeba_transforms(orig_img)
        assert os.path.isfile(old_img_loc)
        assert not os.path.isfile(new_img_loc)
        new_img.save(new_img_loc)
    
    print("Moving testing CelebA images...")
    for i in trange(n_train_imgs+1, n_imgs+1):
        img_filename = f'{i:06d}.jpg'
        old_img_loc = os.path.join(orig_img_dir, img_filename)
        new_img_loc = os.path.join(new_img_dir, img_filename)
        orig_img = Image.open(old_img_loc)
        new_img = celeba_transforms(orig_img)
        assert os.path.isfile(old_img_loc)
        assert not os.path.isfile(new_img_loc)
        new_img.save(new_img_loc)


    print("Script finished!")
