import os
import argparse
import torch
from modeling.anime_gan import Generator
import gc
import cv2
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_file_path', type=str, help='source file path', required=True)
    parser.add_argument('--destination_file_path', type=str, help='destination file path', required=True)
    parser.add_argument('--checkpoint_path', type=str, help="checkpoint_file_path", required=True)
    args = parser.parse_args()
    return args

def preprocess_images(images):
    images = images.astype(np.float32)
    images = images / 127.5 - 1.0 # Normalize the image pixel values to between [-1, 1]
    images = torch.from_numpy(images)
    if torch.cuda.is_available():
        images = images.cuda()
    if len(images.shape) == 3:
        images = images.unsqueeze(0) # Adding batch dimension if not present already
    images = images.permute(0, 3, 1, 2) # change to channel first for torch convention
    return images

def resize_image(image, inter=cv2.INTER_AREA):
    h, w = image.shape[:2]
    resized_width = w - (w % 32)
    resized_height = h - (h % 32)
    cv2.resize(image, (resized_width, resized_height),  interpolation=inter)

def generate_anime_images(checkpoint_path, source_file_path, dest_file_path):
    G = Generator()
    if torch.cuda.is_available():
        G = G.cuda()
    checkpoint = torch.load(checkpoint_path,  map_location='cuda:0') if torch.cuda.is_available() else \
        torch.load(checkpoint_path,  map_location='cpu')
    G.load_state_dict(checkpoint['model_state_dict'], strict=True)
    del checkpoint
    torch.cuda.empty_cache()
    gc.collect()
    G.eval()
    image = cv2.imread(source_file_path)[: ,: ,::-1]
    h, w = image.shape[:2]
    resized_width = w - (w % 32)
    resized_height = h - (h % 32)
    cv2.resize(image, (resized_width, resized_height),  interpolation=cv2.INTER_AREA)
    with torch.no_grad():
        anime_img = G(preprocess_images(image))
        anime_img = anime_img.detach().cpu().numpy()
        anime_img = anime_img.transpose(0, 2, 3, 1)[0]
    # denormalizing image
    anime_img = anime_img * 127.5 + 127.5
    anime_img = anime_img.astype(np.int16)
    cv2.imwrite(dest_file_path, anime_img[..., ::-1])

def validate_file_paths(checkpoint_path, source_file_path, dest_file_path):
    if not os.path.exists(checkpoint_path):
        return False
    if not os.path.isfile(checkpoint_path):
        return False
    if not os.path.exists(source_file_path):
        return False
    if not os.path.isfile(source_file_path):
        return False
    return True


def main():
    args = parse_args()
    validated = validate_file_paths(args.checkpoint_path, args.source_file_path, args.destination_file_path)
    if validated:
        generate_anime_images(checkpoint_path=args.checkpoint_path, source_file_path=args.source_file_path, dest_file_path=args.destination_file_path)
    else:
        print("Not a valid file path")

if __name__ == '__main__':
    main()
