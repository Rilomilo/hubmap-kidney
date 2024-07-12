import pandas as pd
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rasterio
import albumentations as A
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, Subset
from rasterio.windows import Window
from torchvision.transforms import Compose, ToTensor, Normalize
Image.MAX_IMAGE_PIXELS = None

from utils import rle_decode, make_grid, rle_numba_encode

identity = rasterio.Affine(1, 0, 0, 0, 1, 0)


class HubDataset(Dataset):
    def __init__(self, root_dir, transform, window_size=256, overlap=32, threshold=100):
        """
            Parameters:
            - root_dir (str): The root directory of the dataset.
            - transform (callable): A function/transform to be applied on the data.
            - window (int): The size of the sliding window for cropping slices. Default is 256.
            - overlap (int): The amount of overlap between adjacent slices. Default is 32.
            - threshold (int): The threshold for considering a slice as valid. Default is 100.
        """
        
        path = Path(root_dir)
        annotations = pd.read_csv((path / 'train.csv').as_posix(), index_col=[0])

        # load data and crop to slices
        masks = []
        images = []
        for i, filename in enumerate(annotations.index.values):
            filepath = (path / 'train'/(filename+'.tiff')).as_posix()
            print('Transform', filename)
            
            # process a whole slide image
            with rasterio.open(filepath, transform=identity) as wsi:
                mask = rle_decode(annotations.loc[filename, 'encoding'], wsi.shape)
                slices = make_grid(wsi.shape, window=window_size, min_overlap=overlap)

                for slice in tqdm(slices):
                    x1, x2, y1, y2 = slice
                    if mask[x1:x2, y1:y2].sum() > threshold:
                        image = wsi.read([1, 2, 3], window=Window.from_slices((x1, x2), (y1, y2)))

                        image = np.moveaxis(image, 0, -1)
                        images.append(image)
                        masks.append(mask[x1:x2, y1:y2])
            # break

        self.images = images
        self.masks = masks
        self.transform = transform
        self.to_tensor = Compose([
            ToTensor(),
            Normalize([0.625, 0.448, 0.688],
                        [0.131, 0.177, 0.101]),
        ])

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return len(self.images)
    
    def __getitem__(self, index):
        image, mask = self.images[index], self.masks[index]
        if self.transform:
            augments = self.transform(image=image, mask=mask)
            image, mask = augments['image'], augments['mask']
        image = self.to_tensor(image)
        mask = mask[None]

        return image, mask


def get_dataset(data_path, window, min_overlap, new_size):
    trfm = A.Compose([
        A.Resize(new_size,new_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        
        A.OneOf([
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            A.ColorJitter(brightness=0.07, contrast=0.07,
                    saturation=0.1, hue=0.1, always_apply=False, p=0.3),
            ], p=0.3),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(),
            A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=0.0),
        A.ShiftScaleRotate(),
    ])

    dataset = HubDataset(data_path, window_size=window, overlap=min_overlap, transform=trfm)
    

    valid_idx = [i for i in range(len(dataset)) if i%7==0] # 14% of the data
    train_idx = [i for i in range(len(dataset)) if i%7!=0]

    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, valid_idx)


    return train_dataset, valid_dataset

if __name__=="__main__":
    train_dataset, valid_dataset = get_dataset("data", 1024, 32, 256)
    dataset=HubDataset("data", None, window_size=1024, overlap=32)
    image, mask=dataset[2]

    plt.figure(figsize=(16,8))
    plt.subplot(121)
    plt.imshow(mask[0])
    plt.subplot(122)
    plt.imshow(image[0])
    plt.savefig("datasample.png")

    _ = rle_numba_encode(mask[0]) # compile function with numba
    pass