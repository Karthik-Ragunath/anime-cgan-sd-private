#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to fine-tune Stable Diffusion for InstructPix2Pix."""

import argparse
import logging
import math
import os
import shutil
from pathlib import Path

import accelerate
import datasets
import numpy as np
import PIL
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.25.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "fusing/instructpix2pix-1000-samples": ("input_image", "edit_prompt", "edited_image"),
}
WANDB_TABLE_COL_NAMES = ["original_image", "edited_image", "edit_prompt"]


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for InstructPix2Pix.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--original_image_column",
        type=str,
        default="input_image",
        help="The column of the dataset containing the original image on which edits where made.",
    )
    parser.add_argument(
        "--edited_image_column",
        type=str,
        default="edited_image",
        help="The column of the dataset containing the edited image.",
    )
    parser.add_argument(
        "--edit_prompt_column",
        type=str,
        default="edit_prompt",
        help="The column of the dataset containing the edit instruction.",
    )
    parser.add_argument(
        "--val_image_url",
        type=str,
        default=None,
        help="URL to the original image that you would like to edit (used during inference for debugging purposes).",
    )
    parser.add_argument(
        "--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="instruct-pix2pix-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=None,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir) # accelerator_project_config = ProjectConfiguration(project_dir='instruct-pix2pix-model', logging_dir='instruct-pix2pix-model/logs', automatic_checkpoint_naming=False, total_limit=None, iteration=0, save_on_each_node=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, # 4
        mixed_precision=args.mixed_precision, # 'fp16'
        log_with=args.report_to, # 'tensorboard'
        project_config=accelerator_project_config, # ProjectConfiguration(project_dir='instruct-pix2pix-model', logging_dir='instruct-pix2pix-model/logs', automatic_checkpoint_naming=False, total_limit=None, iteration=0, save_on_each_node=False)
    )

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None: # 'instruct-pix2pix-model'
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler") # <DDPMScheduler, len() = 1000>
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision # args.revision = None
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant # args.variant = None
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    ) # <AutoencoderKL>
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision # args.non_ema_revision = None
    ) # <UNet2DConditionModel>

    # InstructPix2Pix uses an additional image for conditioning. To accommodate that,
    # it uses 8 channels (instead of 4) in the first (conv) layer of the UNet. This UNet is
    # then fine-tuned on the custom InstructPix2Pix dataset. This modified UNet is initialized
    # from the pre-trained checkpoints. For the extra channels added to the first layer, they are
    # initialized to zero.
    logger.info("Initializing the InstructPix2Pix UNet from the pretrained UNet.")
    in_channels = 8
    out_channels = unet.conv_in.out_channels # 320
    unet.register_to_config(in_channels=in_channels)

    with torch.no_grad():
        new_conv_in = nn.Conv2d(
            in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
        ) # in_channels = 8, out_channels = 320, unet.conv_in.kernel_size = (3, 3), unet.conv_in.stride = (1, 1), unet.conv_in.padding = (1, 1)
        new_conv_in.weight.zero_() # new_conv_in.weight.shape = torch.Size([320, 8, 3, 3])
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight) # new_conv_in.weight[:, :4, :, :].shape = torch.Size([320, 4, 3, 3])
        unet.conv_in = new_conv_in # new_conv_in = Conv2d(8, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Create EMA for the unet.
    if args.use_ema: # False
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention: # True
        if is_xformers_available(): # True
            import xformers

            xformers_version = version.parse(xformers.__version__) # <Version('0.0.23.post1')>
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"): # <Version('0.25.0')>
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing: # True
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam: # False
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate, # 5e-05
        betas=(args.adam_beta1, args.adam_beta2), # (0.9, 0.999)
        weight_decay=args.adam_weight_decay, # 0.01
        eps=args.adam_epsilon, # 1e-08
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name, # 'fusing/instructpix2pix-1000-samples'
            args.dataset_config_name, # None
            cache_dir=args.cache_dir, # None
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/main/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names # ['input_image', 'edit_prompt', 'edited_image']

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None) # ('input_image', 'edit_prompt', 'edited_image')
    if args.original_image_column is None:
        original_image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        original_image_column = args.original_image_column # 'input_image'
        if original_image_column not in column_names:
            raise ValueError(
                f"--original_image_column' value '{args.original_image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.edit_prompt_column is None:
        edit_prompt_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        edit_prompt_column = args.edit_prompt_column # 'edit_prompt'
        if edit_prompt_column not in column_names: # column_names = ['input_image', 'edit_prompt', 'edited_image']
            raise ValueError(
                f"--edit_prompt_column' value '{args.edit_prompt_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.edited_image_column is None: # 'edited_image'
        edited_image_column = dataset_columns[2] if dataset_columns is not None else column_names[2]
    else:
        edited_image_column = args.edited_image_column # 'edited_image'
        if edited_image_column not in column_names:
            raise ValueError(
                f"--edited_image_column' value '{args.edited_image_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(captions):
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt" # tokenizer.model_max_length = 77 # captions = ['']
        ) # tokenizer.model_max_length = 77
        return inputs.input_ids # inputs.keys() = dict_keys(['input_ids', 'attention_mask']) # inputs.input_ids.shape = torch.Size([4, 77])

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution), # args.center_crop = False # args.resolution = 256
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x), # args.random_flip = True
        ]
    )

    def preprocess_images(examples):
        original_images = np.concatenate(
            [convert_to_np(image, args.resolution) for image in examples[original_image_column]] # original_image_column = 'input_image' # examples.keys() = dict_keys(['input_image', 'edit_prompt', 'edited_image']) # args.resolution = 256 # examples['input_image'][0].size = (512, 512)
        ) # original_images.shape = (12, 256, 256)
        edited_images = np.concatenate(
            [convert_to_np(image, args.resolution) for image in examples[edited_image_column]] # edited_image_column = 'edited_image' # examples['edited_image'][0].size = (512, 512)
        ) # edited_images.shape = (12, 256, 256)
        # We need to ensure that the original and the edited images undergo the same
        # augmentation transforms.
        images = np.concatenate([original_images, edited_images]) # (24, 256, 256)
        images = torch.tensor(images) # torch.Size([24, 256, 256])
        images = 2 * (images / 255) - 1 # torch.Size([24, 256, 256])
        return train_transforms(images)

    def preprocess_train(examples):
        # Preprocess images.
        preprocessed_images = preprocess_images(examples) # preprocessed_images.shape = torch.Size([24, 256, 256]) # dict_keys(['input_image', 'edit_prompt', 'edited_image'])
        # Since the original and edited images were concatenated before
        # applying the transformations, we need to separate them and reshape
        # them accordingly.
        original_images, edited_images = preprocessed_images.chunk(2) # torch.Size([12, 256, 256]) # torch.Size([12, 256, 256])
        original_images = original_images.reshape(-1, 3, args.resolution, args.resolution) # torch.Size([4, 3, 256, 256])
        edited_images = edited_images.reshape(-1, 3, args.resolution, args.resolution) # torch.Size([4, 3, 256, 256])

        # Collate the preprocessed images into the `examples`.
        examples["original_pixel_values"] = original_images # examples.keys() = dict_keys(['input_image', 'edit_prompt', 'edited_image'])
        examples["edited_pixel_values"] = edited_images

        # Preprocess the captions.
        captions = list(examples[edit_prompt_column]) # len(captions) = 4 # ['Make the ghost a can...corn ghost', 'have them be on a beach', 'Make it a watercolor', 'the puzzle is made out of wood']
        examples["input_ids"] = tokenize_captions(captions) # torch.Size([4, 77])
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None: # None
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train) # len(dataset["train"]) = 1000 # dataset["train"] # Dataset({features: ['input_image', 'edit_prompt', 'edited_image'], num_rows: 1000}) # see down

    def collate_fn(examples):
        original_pixel_values = torch.stack([example["original_pixel_values"] for example in examples]) # torch.Size([4, 3, 256, 256]) # len(examples) = 4 # examples[0]["original_pixel_values"].shape = torch.Size([3, 256, 256])
        original_pixel_values = original_pixel_values.to(memory_format=torch.contiguous_format).float() # torch.Size([4, 3, 256, 256])
        edited_pixel_values = torch.stack([example["edited_pixel_values"] for example in examples]) # torch.Size([4, 3, 256, 256])
        edited_pixel_values = edited_pixel_values.to(memory_format=torch.contiguous_format).float() # torch.Size([4, 3, 256, 256])
        input_ids = torch.stack([example["input_ids"] for example in examples]) # torch.Size([4, 77])
        return {
            "original_pixel_values": original_pixel_values,
            "edited_pixel_values": edited_pixel_values,
            "input_ids": input_ids,
        }

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, # len(train_dataset) = 1000
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size, # 4
        num_workers=args.dataloader_num_workers, # 0
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) # 63 # len(train_dataloader) = 250 # args.gradient_accumulation_steps = 4
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler, # 'constant'
        optimizer=optimizer, # see at bottom
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes, # args.lr_warmup_steps = 0, accelerator.num_processes = 1
        num_training_steps=args.max_train_steps * accelerator.num_processes, # args.max_train_steps = 15000 # accelerator.num_processes = 1
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16": # 'fp16'
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("instruct-pix2pix", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs): # first_epoch = 0 # args.num_train_epochs = 239
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader): # len(train_dataloader) = 250
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step: # args.resume_from_checkpoint = None
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # We want to learn the denoising process w.r.t the edited images which
                # are conditioned on the original image (which was edited) and the edit instruction.
                # So, first, convert images to latent space.
                latents = vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist.sample() # vae = <AutoencoderKL> # batch["edited_pixel_values"].shape = torch.Size([4, 3, 256, 256]) # weight_dtype = torch.float16 # vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist = <diffusers.models.autoencoders.vae.DiagonalGaussianDistribution object at 0x7f404241c8e0>
                latents = latents * vae.config.scaling_factor # latents.shape = torch.Size([4, 4, 32, 32]) # vae.config.scaling_factor = 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents) # torch.Size([4, 4, 32, 32])
                bsz = latents.shape[0] # 4
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device) # timesteps.shape = torch.Size([4]) # noise_scheduler.config.num_train_timesteps = 1000 # bsz = 4
                timesteps = timesteps.long() # timesteps.shape = torch.Size([4])
                # timesteps = tensor([561, 969,  58,  71], device='cuda:0')
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps) # noisy_latents.shape = torch.Size([4, 4, 32, 32]) # noise_scheduler = <DDPMScheduler, len() = 1000>

                # Get the text embedding for conditioning.
                encoder_hidden_states = text_encoder(batch["input_ids"])[0] # torch.Size([4, 77, 768]) # batch["input_ids"].shape = torch.Size([4, 77]) # len(text_encoder(batch["input_ids"])) = 2 # text_encoder(batch["input_ids"])[0].shape = torch.Size([4, 77, 768]) # text_encoder(batch["input_ids"])[1].shape = torch.Size([4, 768])

                # Get the additional image embedding for conditioning.
                # Instead of getting a diagonal Gaussian here, we simply take the mode.
                original_image_embeds = vae.encode(batch["original_pixel_values"].to(weight_dtype)).latent_dist.mode() # original_image_embeds.shape = torch.Size([4, 4, 32, 32]) # batch["original_pixel_values"].shape = torch.Size([4, 3, 256, 256])

                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                if args.conditioning_dropout_prob is not None: # 0.05
                    random_p = torch.rand(bsz, device=latents.device, generator=generator) # torch.Size([4])
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob # torch.Size([4]) # bsz = 4
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1) # torch.Size([4, 1, 1])
                    # Final text conditioning.
                    null_conditioning = text_encoder(tokenize_captions([""]).to(accelerator.device))[0] # torch.Size([1, 77, 768]) # tokenize_captions([""]).shape = torch.Size([1, 77]) # len(text_encoder(tokenize_captions([""])) = 2 # text_encoder(tokenize_captions([""]).to(accelerator.device))[0].shape = torch.Size([1, 77, 768]) # text_encoder(tokenize_captions([""]).to(accelerator.device))[1].shape = torch.Size([1, 768])
                    encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states) # torch.Size([4, 77, 768]) # prompt_mask.shape = torch.Size([4, 1, 1]) # null_conditioning.shape = torch.Size([1, 77, 768]) # encoder_hidden_states.shape = torch.Size([4, 77, 768])

                    # Sample masks for the original images.
                    image_mask_dtype = original_image_embeds.dtype # torch.float16
                    image_mask = 1 - (
                        (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype) # args.conditioning_dropout_prob = 0.05 # random_p.shape = torch.Size([4]) # image_mask_dtype =torch.float16
                        * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1) # torch.Size([4])
                    # Final image conditioning.
                    original_image_embeds = image_mask * original_image_embeds # torch.Size([4, 4, 32, 32]) # original_image_embeds.shape = torch.Size([4, 4, 32, 32])

                # Concatenate the `original_image_embeds` with the `noisy_latents`.
                concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1) # torch.Size([4, 8, 32, 32]) # noisy_latents.shape = torch.Size([4, 4, 32, 32]) # original_image_embeds.shape = torch.Size([4, 4, 32, 32])

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise # torch.Size([4, 4, 32, 32])
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(concatenated_noisy_latents, timesteps, encoder_hidden_states).sample # torch.Size([4, 4, 32, 32]) # vars(unet(concatenated_noisy_latents, timesteps, encoder_hidden_states)) = {'sample': tensor([[[[-0.9248, ...ackward0>)}
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean") # tensor(0.1638, device='cuda:0', grad_fn=<MseLossBackward0>)

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean() # tensor(0.1638, device='cuda:0', grad_fn=<MeanBackward0>)
                train_loss += avg_loss.item() / args.gradient_accumulation_steps # 0.040958549827337265

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients: # False
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm) # args.max_grad_norm = 1.0
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if (
                (args.val_image_url is not None)
                and (args.validation_prompt is not None)
                and (epoch % args.validation_epochs == 0)
            ):
                logger.info(
                    f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                    f" {args.validation_prompt}."
                )
                # create pipeline
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())
                # The models need unwrapping because for compatibility in distributed training mode.
                pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=accelerator.unwrap_model(unet),
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    vae=accelerator.unwrap_model(vae),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)

                # run inference
                original_image = download_image(args.val_image_url)
                edited_images = []
                with torch.autocast(
                    str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"
                ):
                    for _ in range(args.num_validation_images):
                        edited_images.append(
                            pipeline(
                                args.validation_prompt,
                                image=original_image,
                                num_inference_steps=20,
                                image_guidance_scale=1.5,
                                guidance_scale=7,
                                generator=generator,
                            ).images[0]
                        )

                for tracker in accelerator.trackers:
                    if tracker.name == "wandb":
                        wandb_table = wandb.Table(columns=WANDB_TABLE_COL_NAMES)
                        for edited_image in edited_images:
                            wandb_table.add_data(
                                wandb.Image(original_image), wandb.Image(edited_image), args.validation_prompt
                            )
                        tracker.log({"validation": wandb_table})
                if args.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(unet.parameters())

                del pipeline
                torch.cuda.empty_cache()

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=accelerator.unwrap_model(text_encoder),
            vae=accelerator.unwrap_model(vae),
            unet=unet,
            revision=args.revision,
            variant=args.variant,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

        if args.validation_prompt is not None:
            edited_images = []
            pipeline = pipeline.to(accelerator.device)
            with torch.autocast(str(accelerator.device).replace(":0", "")):
                for _ in range(args.num_validation_images):
                    edited_images.append(
                        pipeline(
                            args.validation_prompt,
                            image=original_image,
                            num_inference_steps=20,
                            image_guidance_scale=1.5,
                            guidance_scale=7,
                            generator=generator,
                        ).images[0]
                    )

            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    wandb_table = wandb.Table(columns=WANDB_TABLE_COL_NAMES)
                    for edited_image in edited_images:
                        wandb_table.add_data(
                            wandb.Image(original_image), wandb.Image(edited_image), args.validation_prompt
                        )
                    tracker.log({"test": wandb_table})

    accelerator.end_training()


if __name__ == "__main__":
    main()

# deterministic: False
# vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist.logvar.shape
# torch.Size([4, 4, 32, 32])
# vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist.mean.shape
# torch.Size([4, 4, 32, 32])
# vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist.parameters.shape
# torch.Size([4, 8, 32, 32])
# vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist.std.shape
# torch.Size([4, 4, 32, 32])
# vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist.var.shape
# torch.Size([4, 4, 32, 32])
# torch.min(latents)
# tensor(-5.5273, device='cuda:0', dtype=torch.float16)
# torch.max(latents)
# tensor(5.3867, device='cuda:0', dtype=torch.float16)
# torch.max(noise)
# tensor(4.0312, device='cuda:0', dtype=torch.float16)
# torch.min(noise)
# tensor(-4.0586, device='cuda:0', dtype=torch.float16)
# inputs - [""]
# {'input_ids': tensor([[49406, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
#          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
#          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
#          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
#          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
#          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
#          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
#          49407, 49407, 49407, 49407, 49407, 49407, 49407]]), 'attention_mask': tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0]])}

# optimizer
# AcceleratedOptimizer (
# Parameter Group 0
#     amsgrad: False
#     betas: (0.9, 0.999)
#     capturable: False
#     differentiable: False
#     eps: 1e-08
#     foreach: None
#     fused: None
#     initial_lr: 5e-05
#     lr: 5e-05
#     maximize: False
#     weight_decay: 0.01
# )

# args
# Namespace(
# pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5', 
# revision=None, variant=None, 
# dataset_name='fusing/instructpix2pix-1000-samples', 
# dataset_config_name=None, 
# train_data_dir=None, 
# original_image_column='input_image', 
# edited_image_column='edited_image', 
# edit_prompt_column='edit_prompt', 
# val_image_url=None, 
# validation_prompt=None, 
# num_validation_images=4, 
# validation_epochs=1, 
# max_train_samples=None, 
# output_dir='instruct-pix2pix-model', 
# cache_dir=None, 
# seed=42, 
# resolution=256, 
# center_crop=False, 
# random_flip=True, 
# train_batch_size=4, 
# num_train_epochs=100, 
# max_train_steps=15000, 
# gradient_accumulation_steps=4, 
# gradient_checkpointing=True, 
# learning_rate=5e-05, 
# scale_lr=False, 
# lr_scheduler='constant', 
# lr_warmup_steps=0, 
# conditioning_dropout_prob=0.05, 
# use_8bit_adam=False, 
# allow_tf32=False, 
# use_ema=False, 
# non_ema_revision=None, 
# dataloader_num_workers=0, 
# adam_beta1=0.9, 
# adam_beta2=0.999, 
# adam_weight_decay=0.01, 
# adam_epsilon=1e-08,
#  max_grad_norm=1.0, 
# push_to_hub=False, 
# hub_token=None, 
# hub_model_id=None, 
# logging_dir='logs', 
# mixed_precision='fp16', 
# report_to='tensorboard', 
# local_rank=-1, 
# checkpointing_steps=5000, 
# checkpoints_total_limit=1, 
# resume_from_checkpoint=None, 
# enable_xformers_memory_efficient_attention=True
# )

# tokenizer
# CLIPTokenizer(name_or_path='runwayml/stable-diffusion-v1-5', vocab_size=49408, model_max_length=77, is_fast=False, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<|startoftext|>', 'eos_token': '<|endoftext|>', 'unk_token': '<|endoftext|>', 'pad_token': '<|endoftext|>'}, clean_up_tokenization_spaces=True),  added_tokens_decoder={
# 	49406: AddedToken("<|startoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
# 	49407: AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
# }

# text_encoder
# CLIPTextModel(
#   (text_model): CLIPTextTransformer(
#     (embeddings): CLIPTextEmbeddings(
#       (token_embedding): Embedding(49408, 768)
#       (position_embedding): Embedding(77, 768)
#     )
#     (encoder): CLIPEncoder(
#       (layers): ModuleList(
#         (0-11): 12 x CLIPEncoderLayer(
#           (self_attn): CLIPAttention(
#             (k_proj): Linear(in_features=768, out_features=768, bias=True)
#             (v_proj): Linear(in_features=768, out_features=768, bias=True)
#             (q_proj): Linear(in_features=768, out_features=768, bias=True)
#             (out_proj): Linear(in_features=768, out_features=768, bias=True)
#           )
#           (layer_norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#           (mlp): CLIPMLP(
#             (activation_fn): QuickGELUActivation()
#             (fc1): Linear(in_features=768, out_features=3072, bias=True)
#             (fc2): Linear(in_features=3072, out_features=768, bias=True)
#           )
#           (layer_norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#         )
#       )
#     )
#     (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#   )
# )

# dataset["train"]['input_image'][0].size
# (512, 512)
# dataset["train"]['edited_image'][0].size
# (512, 512)
# dataset["train"]['edit_prompt'][0]
# 'Turn it into a photo'

# inputs
# {'input_ids': tensor([[49406,  1078,   518,  7108,   320,  6417,  5894,  7108, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407],
        # [49406,   720,  1180,   655,   525,   320,  2117, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407],
        # [49406,  1078,   585,   320, 14211, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407],
        # [49406,   518, 12875,   533,  1105,   620,   539,  1704, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #  49407, 49407, 49407, 49407, 49407, 49407, 49407]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #  0, 0, 0, 0, 0],
        # [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #  0, 0, 0, 0, 0],
        # [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #  0, 0, 0, 0, 0],
        # [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #  0, 0, 0, 0, 0]])}