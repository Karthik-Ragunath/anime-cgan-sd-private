import torch
import argparse
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file_path", required=True, help="provide file path where source image is present")
    parser.add_argument("--destination_file_path", required=True, help="provide file path where destination image is present")
    parser.add_argument("--edit_condition", required=True, help="edit condition to applied to the source image")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    id = "timbrooks/instruct-pix2pix"
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(id, torch_dtype=torch.float16, safety_checker=None)
    pipeline.to("cuda")
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    args = parse_arguments()
    image = Image.open(args.source_file_path)
    images = pipeline(args.edit_condition, image=image).images
    images[0].save(args.destination_file_path)