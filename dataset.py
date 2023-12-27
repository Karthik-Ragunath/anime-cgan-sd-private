import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

def compute_data_mean(directory_path):
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f'{directory_path} directory not found')
    image_files = os.listdir(directory_path)
    total = np.zeros(3)
    for img_file in tqdm(image_files):
        path = os.path.join(directory_path, img_file)
        image = cv2.imread(path)
        total += image.mean(axis=(0, 1))
    channel_mean = total / len(image_files)
    mean = np.mean(channel_mean)
    return mean - channel_mean[...,::-1]  # Converting to BGR

class AnimeDataSet(Dataset):
    def __init__(self, args):
        data_directory = args.data_directory
        dataset = args.dataset
        anime_files_dir = os.path.join(data_directory, dataset)
        if not os.path.exists(anime_files_dir):
            raise FileNotFoundError(f'{anime_files_dir} directory does not exist')
        self.mean = compute_data_mean(os.path.join(anime_files_dir, 'style'))
        self.data_dir = data_directory
        self.image_files_dict =  dict()
        self.photo_directory = f'{data_directory}/train_photo'
        self.style_directory = f'{anime_files_dir}/style'
        self.smooth_directory =  f'{anime_files_dir}/smooth'
        for opt in [self.photo_directory, self.style_directory, self.smooth_directory]:
            files = os.listdir(opt)
            self.image_files_dict[opt] = [os.path.join(opt, fi) for fi in files]
        print(f'Dataset: real {len(self.image_files_dict[self.photo_directory])} style {self.len_anime}, smooth {self.len_smooth}')

    def __len__(self):
        return len(self.image_files_dict[self.photo_directory])

    @property
    def len_anime(self):
        return len(self.image_files_dict[self.style_directory])

    @property
    def len_smooth(self):
        return len(self.image_files_dict[self.smooth_directory])
    
    def normalize(self, img, addmean=True):
        img = img.astype(np.float32)
        if addmean:
            img += self.mean
        return img / 127.5 - 1.0

    def __getitem__(self, index):
        # Incase enough anime data is not present when compared to train data,
        # we cycle through the existing data

        # get real image
        real_image_path = self.image_files_dict[self.photo_directory][index]
        image = cv2.imread(real_image_path)[:,:,::-1]
        image = self.normalize(image, addmean=False)
        image = image.transpose(2, 0, 1)
        real_image = torch.tensor(image)

        # get anime image
        anime_index = index
        if anime_index > self.len_anime - 1:
            anime_index -= self.len_anime * (index // self.len_anime)
        anime_path = self.image_files_dict[self.style_directory][anime_index]
        image = cv2.imread(anime_path)[:,:,::-1]
        image_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        image_gray = np.stack([image_gray, image_gray, image_gray], axis=-1)
        image_gray = self.normalize(image_gray, addmean=False)
        image_gray = image_gray.transpose(2, 0, 1)
        image = self.normalize(image, addmean=True)
        image = image.transpose(2, 0, 1)
        anime, anime_gray = torch.tensor(image), torch.tensor(image_gray)

        # get smooth gray anime image
        smooth_anime_image_path = self.image_files_dict[self.smooth_directory][anime_index]
        image = cv2.imread(smooth_anime_image_path, cv2.IMREAD_GRAYSCALE)
        image = np.stack([image, image, image], axis=-1)
        image = self.normalize(image, addmean=False)
        image = image.transpose(2, 0, 1)
        smooth_gray = torch.tensor(image)
    
        # return get_item call
        return real_image, anime, anime_gray, smooth_gray
