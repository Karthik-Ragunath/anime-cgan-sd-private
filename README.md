# Conditional GAN + Stable-Diffusion To Create Anime-Styled Images And Make Custom Animation Edits

## WORK INSPIRED FROM FOLLOWING PAPERS

```
1. AnimeGAN: A Novel Lightweight GAN for Photo Animation
2. InstructPix2Pix: Learning to Follow Image Editing Instructions
```

----------------------
## 1.SUMMARY

Training and Inference code for a model to convert input images into anime_style
Inference code to use Stable-Diffusion model to edit the anime_style image generated

-----------------------
## 2. DATA AND CHECKPOINTS

### 2.1 DOWNLOAD DATASET FOR TRAINING CGAN

```bash
wget -O anime-gan.zip https://github.com/ptran1203/pytorch-animeGAN/releases/download/v1.0/dataset_v1.zip
unzip anime-gan.zip -d /content
```

-----------------------
## 3. VSCODE DEBUGGER CONFIGS TO RUN TRAIN AND INFERENCE SCRIPTS

### 3.1 Train CGAN for converting images into anime style

```
{
    "name": "train",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/train.py",
    "console": "integratedTerminal",
    "justMyCode": true,
    "args": [
        "--resume_cond", "gen_dis",
        "--dataset", "Hayao",
        "--use_spectral_norm",
        "--lr-discriminator", "0.00004",
        "--batch-size", "6",
        "--initial-epochs", "1",
        "--initial-lr", "0.0001",
        "--save-interval", "1",
        "--lr-generator", "0.00002",
        "--checkpoint-dir", "checkpoints",
        "--adversarial_loss_disc_weight", "10.0",
        "--save-image-dir", "save_imgs",          
        "--adversarial_loss_gen_weight", "10.0",              
        "--content_loss_weight", "1.5",              
        "--gram_loss_weight", "3.0",                
        "--chromatic_loss_weight", "30.0",                
    ]
}
```

### 2.2 Inference With CGAN
```
{
    "name": "inference",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/inference.py",
    "console": "integratedTerminal",
    "justMyCode": true,
    "args": [
        "--checkpoint_path", "checkpoints/generator_Hayao.pth",
        "--source_file_path", "example/result/140.jpeg",
        "--destination_file_path", "save_imgs/inference_images/140_anime.jpg",
    ]
}
```

### 2.3 Edit Anime Styled Image With Stable-Diffusion
```
{
    "name": "stable_diffusion_edits",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/stable_diffusion_inference.py",
    "console": "integratedTerminal",
    "justMyCode": true,
    "args": [
        "--source_file_path", "save_imgs/inference_images/140_anime.jpg",
        "--destination_file_path", "save_imgs/inference_images/140_anime_stable_diffused.jpg",
        "--edit_condition", "change the color of the bus to black"
    ]
}
```

-----------------------
## 3. RESULTS

### 3.1 INPUT IMAGE
![Input Image](example/10.jpg)

### 3.2 ANIME STYLED IMAGE GENERATED
![Anime Styled Image](save_imgs/inference_images/10_anime.jpg)

### 3.3 STABLE DIFFUSION EDITED IMAGE
```
CONDITIONAL QUERY - "turn green chairs into blue"
```
![Stable Diffusion Edited Image](save_imgs/inference_images/10_anime_stable_diffused.jpg)

-----------------------
