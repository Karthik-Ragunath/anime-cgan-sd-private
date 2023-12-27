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
unzip anime-gan.zip
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

### 2.4 TRAIN STABLE DIFFUSION
```
{
    "name": "train_sd",
    "type": "python",
    "request": "launch",
    "module": "accelerate.commands.launch",
    "console": "integratedTerminal",
    "justMyCode": true,
    "args": [
        "train_instruct_pix2pix.py",
        "--pretrained_model_name_or_path", "runwayml/stable-diffusion-v1-5",
        "--dataset_name", "fusing/instructpix2pix-1000-samples",
        "--enable_xformers_memory_efficient_attention",
        "--resolution", "256",
        "--random_flip",
        "--train_batch_size", "4", 
        "--gradient_accumulation_steps", "4",
        "--gradient_checkpointing",
        "--max_train_steps", "4708",
        "--checkpointing_steps", "1000",
        "--checkpoints_total_limit", "1",
        "--learning_rate", "5e-05", 
        "--max_grad_norm", "1", 
        "--lr_warmup_steps", "0",
        "--conditioning_dropout_prob", "0.05",
        "--mixed_precision", "fp16",
        "--num_train_epochs", "100",
        "--seed", "42",
        // "--push_to_hub"
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
## 4. STABLE DIFFUSION MODEL TRAINING PROCESS

### 4.1 CONCEPTS INVOLVED FROM CODE IMPLEMENTATION POV

The stable diffusion model involves 4 individual models which works as part of a diffusion pipeline:

__1.__ `Noise Scheduler:`

We have used `DDPMScheduler` as the scheduler for our diffusion process. This scheduler controls how much noise is added at each timestep controlled via:
`add_noise` function.

```
Example:
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler") 
# args.pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"

noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps) 
# noisy_latents.shape = torch.Size([4, 4, 32, 32]) 
# noise_scheduler = <DDPMScheduler, len() = 1000>
```

Work by Sohl-Dickstein et al., has shown that we can sample xt at any arbitrary timestep (noise level) conditioned based on x at 0th timestep (x0). 
Refer - `https://huggingface.co/blog/annotated-diffusion`

By taking advantage of this gaussian property, we can sample from a distribution which has noise levels at arbitrary timesteps, example, lets say that if total number of timesteps in the diffusion process is 1000, we can sample from distribution which mimicks the noise level at 0th, 29th, 39th, etc., or any arbitrary timestep for that matter and use it during training process.

For details about the pretrained model used, please refer:
`https://huggingface.co/runwayml/stable-diffusion-v1-5`

The diffusers subfolder as mentioned in 
```
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
```
is found in:
`cd ~/anaconda3/envs/sd/lib/python3.10/site-packages/diffusers/schedulers/`

__2.__ `CLIPTokenizer`

As used in CLIP Paper, we also use this tokenizer with the following config:
```
max_length=77
padding="max_length"
truncation=True
return_tensors="pt"
```

By this config, we pad the output to `max_lenth` of 77 tokens and incase number of input tokens is greater than 77, we `truncate` the sentence to first 77 tokens.
The output values will indicate the token_id (integers) which denote the id of that particular token in CLIPTokenizer's vocabulary corpus.
`return_tensors="pt"` config indicates that output is of data type - torch.Tensor

```
# Preprocessing the datasets.
# We need to tokenize input captions and transform the images.
def tokenize_captions(captions):
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt" 
        # tokenizer.model_max_length = 77 
        # captions = ['']
    ) 
    # tokenizer.model_max_length = 77
    
    return inputs.input_ids 
    # inputs.keys() = dict_keys(['input_ids', 'attention_mask']) 
    # inputs.input_ids.shape = torch.Size([4, 77])
```

__3.__ `CLIPTextModel`

```
text_encoder = CLIPTextModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="text_encoder"
)
encoder_hidden_states = text_encoder(batch["input_ids"])[0] # torch.Size([4, 77, 768]) 
# batch["input_ids"].shape = torch.Size([4, 77]) 
# len(text_encoder(batch["input_ids"])) = 2 
# text_encoder(batch["input_ids"])[0].shape = torch.Size([4, 77, 768]) 
# text_encoder(batch["input_ids"])[1].shape = torch.Size([4, 768])
```

The input to this `CLIPTextModel` instance: `text_encoder` is the padded token ids from `CLIPTokenizer` which we saw previously

The `text_encoder's` output is a tuple:

__(i)__ last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model.

__(ii)__ pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.

__4.__ AutoencoderKL

Pre-trained Autoencoder model trained with KL divergence loss function is used for converting images into latent encodings.

```
vae = AutoencoderKL.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="vae"
) # <AutoencoderKL>
```

We use auto-encoder model variant from `diffusers` PyPi package.

Now, lets see in-depth of concepts involved with `AutoencoderKL`.

```
latents = vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist.sample() # vae = <AutoencoderKL> 
# batch["edited_pixel_values"].shape = torch.Size([4, 3, 256, 256]) 
# weight_dtype = torch.float16 
# vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist = <diffusers.models.autoencoders.vae.DiagonalGaussianDistribution object at 0x7f404241c8e0>

latents = latents * vae.config.scaling_factor 
# latents.shape = torch.Size([4, 4, 32, 32]) 
# vae.config.scaling_factor = 0.18215
```

As seen above, the input to the `vae` model are rgb images of shape 256 * 256 (with 3 channels).

The outputs are encoding latents of shape - (batch_size, 4, 32, 32)

The output is of type `DiagonalGaussianDistribution` which means that both `mean` and `covariance` outputs/parameters of gaussian model are of same dimension. 
This basically means that this model assumes that there is zero co-variance between different dimensions of the gaussian model and hence the covariance matrix is basically a diagonal matrix. Hence, instead of requiring N^2 dimensional data to represent co-variance output, co-variance can be represented with only N dimensional data (same as that of mean of gaussian model).

__5.__ UNet2DConditionModel


UNet2DConditionModel is the most important component of the diffusion pipeline.

```
unet = UNet2DConditionModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision # args.non_ema_revision = None
) # <UNet2DConditionModel>
```

UNet2DConditionModel is a variant of the unet model which takes text encoding (text queries) as conditional input offered by diffusers PyPi package. We use the pre-trained weights from HuggingFace platform.

Deeper look at UNet2DConditionalModel:

```
# Predict the noise residual and compute loss

model_pred = unet(concatenated_noisy_latents, timesteps, encoder_hidden_states).sample 
# torch.Size([4, 4, 32, 32]) 
# vars(unet(concatenated_noisy_latents, timesteps, encoder_hidden_states)) = {'sample': tensor([[[[-0.9248, ...ackward0>)}

loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean") 
# tensor(0.1638, device='cuda:0', grad_fn=<MseLossBackward0>)
```

This component produces the output whose shape is same as that of the input latent encoding. (batch_size, 4, 32, 32)
This output basically represents the noise level added at the specific timestep.

Then the predicted noise is compared with ground truth noise which was added at that particular timestep. We use L2 loss to compute the loss between predicted and ground truth noise.

This loss guides the stable-diffusion pipelines backpropagation updates to fine-tune the models chained in this stable-diffusion pipeline. 
In our experiment, we are only fine-tuning the UNet2DConditionalModel present in the pipeline.

## 4.2 `Input and Output:`

The inputs for our training process are:
__(i)__ `input_image` - Image of shape 512 * 512
__(ii)__ `edit_prompt` - Edit instruction in the form of text
__(iii)__ `edited_image` - Edited image of shape 512 * 512

The output generated by stable-diffusion model pipeline is of same dimension as input `image embedding` generated (or can also be considered to have same dimension as `random_noise embedding` generated).

## 4.3 STEPS INVOLVED IN TRAINING INSTRUCT PIX2PIX

__(i)__ image latents are computed with the vae (AutoencoderKL) model for the edited image.
```
latents = vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist.sample() 
# vae = <AutoencoderKL> 
# batch["edited_pixel_values"].shape = torch.Size([4, 3, 256, 256]) 
# weight_dtype = torch.float16 
# vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist = <diffusers.models.autoencoders.vae.DiagonalGaussianDistribution object at 0x7f404241c8e0>

latents = latents * vae.config.scaling_factor 
# latents.shape = torch.Size([4, 4, 32, 32]) 
# vae.config.scaling_factor = 0.18215
```

__(ii)__ random noise is generated with the same dimension as edited image embeddings
```
noise = torch.randn_like(latents) # torch.Size([4, 4, 32, 32])
```

__(iii)__ batch_size number of arbitrary time-step indices are sampled from total number of diffusion time-steps
```
bsz = latents.shape[0] # 4
# Sample a random timestep for each image

timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device) # timesteps.shape = torch.Size([4]) 
# noise_scheduler.config.num_train_timesteps = 1000 
# bsz = 4

timesteps = timesteps.long() 
# timesteps.shape = torch.Size([4])
```

__(iv)__ `DDPMScheduler` is used to compute noise values corresponding to arbitrary-timesteps which is to be added to the latents.
```
noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps) 
# noisy_latents.shape = torch.Size([4, 4, 32, 32]) # noise_scheduler = <DDPMScheduler, len() = 1000>
```

__(v)__ `CLIPTextModel` is used to compute text embeddings for the input text query. `AutoencoderKL` model is used to compute image embeddings for the original (unedited) image
```
# Get the text embedding for conditioning.
encoder_hidden_states = text_encoder(batch["input_ids"])[0] 
# torch.Size([4, 77, 768]) 
# batch["input_ids"].shape = torch.Size([4, 77]) 
# len(text_encoder(batch["input_ids"])) = 2 
# text_encoder(batch["input_ids"])[0].shape = torch.Size([4, 77, 768]) 
# text_encoder(batch["input_ids"])[1].shape = torch.Size([4, 768])

# Get the additional image embedding for conditioning.
# Instead of getting a diagonal Gaussian here, we simply take the mode.
original_image_embeds = vae.encode(batch["original_pixel_values"].to(weight_dtype)).latent_dist.mode() 
# original_image_embeds.shape = torch.Size([4, 4, 32, 32]) 
# batch["original_pixel_values"].shape = torch.Size([4, 3, 256, 256])
```

__(vi)__ As per the technique mentioned in classfier-free guidance paper, text embeddings are also generated for `no-text-query` input condition.
Based on random generated values, either `text-query` input embedding or `no-text-query` input embedding are considered.
Similarly, based on random generated values, `original_image_embeds` are also masked.

Random values are generated such that, 
`no-text-query` input condition exists for 5% of total input during training
`original_image_embeds` is masked for 5% of total input during training
both `no-text-query` and `original_image_embeds` condition exists for 5% of total input during training

This helps to have the capability to handle conditional or unconditional denoising with respect to both or either conditional inputs.

```
if args.conditioning_dropout_prob is not None: # 0.05
    random_p = torch.rand(bsz, device=latents.device, generator=generator) 
    # torch.Size([4])
    # Sample masks for the edit prompts.
    
    prompt_mask = random_p < 2 * args.conditioning_dropout_prob 
    # torch.Size([4]) # bsz = 4
    
    prompt_mask = prompt_mask.reshape(bsz, 1, 1) 
    # torch.Size([4, 1, 1])
    
    # Final text conditioning.
    null_conditioning = text_encoder(tokenize_captions([""]).to(accelerator.device))[0] 
    # torch.Size([1, 77, 768]) 
    # tokenize_captions([""]).shape = torch.Size([1, 77]) 
    # len(text_encoder(tokenize_captions([""])) = 2 
    # text_encoder(tokenize_captions([""]).to(accelerator.device))[0].shape = torch.Size([1, 77, 768]) 
    # text_encoder(tokenize_captions([""]).to(accelerator.device))[1].shape = torch.Size([1, 768])
    
    encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states) 
    # torch.Size([4, 77, 768]) 
    # prompt_mask.shape = torch.Size([4, 1, 1]) 
    # null_conditioning.shape = torch.Size([1, 77, 768]) 
    # encoder_hidden_states.shape = torch.Size([4, 77, 768])

    # Sample masks for the original images.
    image_mask_dtype = original_image_embeds.dtype 
    # torch.float16
    
    image_mask = 1 - (
        (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype) 
        # args.conditioning_dropout_prob = 0.05 
        # random_p.shape = torch.Size([4]) 
        # image_mask_dtype = torch.float16

        * 
        
        (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
    )
    image_mask = image_mask.reshape(bsz, 1, 1, 1) 
    # torch.Size([4])
    
    # Final image conditioning.

    original_image_embeds = image_mask * original_image_embeds 
    # torch.Size([4, 4, 32, 32]) 
    # original_image_embeds.shape = torch.Size([4, 4, 32, 32])
```

__(vii)__ Noisy edited image embeddings are then concatenaed with randomly masked original_image_embeds.
```
# Concatenate the `original_image_embeds` with the `noisy_latents`.
concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1) 
# torch.Size([4, 8, 32, 32]) 
# noisy_latents.shape = torch.Size([4, 4, 32, 32]) 
# original_image_embeds.shape = torch.Size([4, 4, 32, 32])
```

__(viii)__ Target is considered as the noise generated for arbitrary time-step which was generated by DDPMScheduler.
```
target = noise # torch.Size([4, 4, 32, 32])
```

__(ix)__ Noise prediction by the unet model and loss computation by comparing predicted noise with gt-noise added with L2 metrics.
```
# Predict the noise residual and compute loss

model_pred = unet(concatenated_noisy_latents, timesteps, encoder_hidden_states).sample 
# torch.Size([4, 4, 32, 32]) 
# vars(unet(concatenated_noisy_latents, timesteps, encoder_hidden_states)) = {'sample': tensor([[[[-0.9248, ...ackward0>)}

loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean") 
# tensor(0.1638, device='cuda:0', grad_fn=<MseLossBackward0>)
```

__(x)__ UNet2DConditionModel model parameters are finetuned with backpropagation.
```
# Backpropagate
accelerator.backward(loss)

if accelerator.sync_gradients: # False
    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm) 
    # args.max_grad_norm = 1.0

optimizer.step()
lr_scheduler.step()
optimizer.zero_grad()
```

## 4.4 HOW THE PREDICTED OUTPUT IS USED

This output generated by the model during x_t th time-step is used as input by the Diffusion-Scheduler instance (in our case, DDPMScheduler) to remove the residual noise predicted from the image to generate image distribution which existed during x_t-1 th time-step of the forward diffusion process.
This is done during inference process (recursively) to get sample from x_0 th time-step distribution of the input.

-----------------------
