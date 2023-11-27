import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import normalize_input, compute_data_mean

class AnimeDataSet(Dataset):
    def __init__(self, args, transform=None):
        data_directory = args.data_dir
        dataset = args.dataset

        anime_files_dir = os.path.join(data_directory, dataset)

        if not os.path.exists(anime_files_dir):
            raise FileNotFoundError(f'{anime_files_dir} directory does not exist')

        self.mean = compute_data_mean(os.path.join(anime_files_dir, 'style'))
        print(f'Mean(B, G, R) of {dataset} are {self.mean}')

        self.debug_samples = args.debug_samples or 0
        self.data_dir = data_directory
        self.image_files =  {}
        self.photo = f'{data_directory}/train_photo'
        self.style = f'{anime_files_dir}/style'
        self.smooth =  f'{anime_files_dir}/smooth'
        self.dummy = torch.zeros(3, 256, 256)
        for opt in [self.photo, self.style, self.smooth]:
            files = os.listdir(opt)
            self.image_files[opt] = [os.path.join(opt, fi) for fi in files]
        self.transform = transform
        print(f'Dataset: real {len(self.image_files[self.photo])} style {self.len_anime}, smooth {self.len_smooth}')

    def __len__(self):
        return self.debug_samples or len(self.image_files[self.photo])

    @property
    def len_anime(self):
        return len(self.image_files[self.style])

    @property
    def len_smooth(self):
        return len(self.image_files[self.smooth])

    def __getitem__(self, index):
        # Incase enough anime data is not present when compared to train data,
        # we cycle through the existing data

        # get real image
        real_image_path = self.image_files[self.photo][index]
        image = cv2.imread(real_image_path)[:,:,::-1]
        image = self._transform(image, addmean=False)
        image = image.transpose(2, 0, 1)
        real_image = torch.tensor(image)

        # get anime image
        anime_index = index
        if anime_index > self.len_anime - 1:
            anime_index -= self.len_anime * (index // self.len_anime)
        anime_path = self.image_files[self.style][index]
        image = cv2.imread(anime_path)[:,:,::-1]
        image_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        image_gray = np.stack([image_gray, image_gray, image_gray], axis=-1)
        image_gray = self._transform(image_gray, addmean=False)
        image_gray = image_gray.transpose(2, 0, 1)
        image = self._transform(image, addmean=True)
        image = image.transpose(2, 0, 1)
        anime, anime_gray = torch.tensor(image), torch.tensor(image_gray)

        # get smooth gray anime image
        smooth_anime_image_path = self.image_files[self.smooth][index]
        image = cv2.imread(smooth_anime_image_path, cv2.IMREAD_GRAYSCALE)
        image = np.stack([image, image, image], axis=-1)
        image = self._transform(image, addmean=False)
        image = image.transpose(2, 0, 1)
        smooth_gray = torch.tensor(image)
    
        # return get_item call
        return real_image, anime, anime_gray, smooth_gray

    def _transform(self, img, addmean=True):
        if self.transform is not None:
            img =  self.transform(image=img)['image']
        img = img.astype(np.float32)
        if addmean:
            img += self.mean
        return normalize_input(img)
