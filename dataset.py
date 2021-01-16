# %%
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from byol_transforms import DataTransform, denormalize


class BYOLDataset(Dataset):
    def __init__(self, images):
        self.images = images
        self.transform = DataTransform()

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        return self.transform(img)

    def __len__(self):
        return len(self.images)


# ===================================================================================
import albumentations as A
T_CLASSIFIER = A.Compose(
    [
        A.Resize(150, 150),
        A.HorizontalFlip(p=0.5),
        A.Normalize(),
    ]
)


class ClassifierDataset(Dataset):
    def __init__(self, folder, transforms=T_CLASSIFIER, training=True):
        self.folder = folder
        self.transforms = transforms
        self.training = training

    def __getitem__(self, index):
        img_tuple = self.folder.imgs[index]

        img = np.array(Image.open(img_tuple[0]))
        label = img_tuple[1]

        if self.training:
            img = self.transforms(image=img)['image']
        else:
            img = self.transforms[0](image=img)['image']
            img = self.transforms[2](image=img)['image']

        return torch.as_tensor(img, dtype=torch.float).permute(2, 0, 1), torch.as_tensor(label)

    def __len__(self):
        return len(self.folder)


# # %%
# import glob
# from PIL import Image
# import matplotlib.pyplot as plt
# data = glob.glob('data/no_label/*.jpg')
# dset = BYOLDataset(data)


# for i in range(10):
#     img_1, img_2 = dset[i]
#     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
#     axes[0].imshow(img_1.permute(1,2,0))
#     axes[1].imshow(img_2.permute(1,2,0))
