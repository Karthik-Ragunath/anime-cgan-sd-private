import torch
import argparse
import os
import cv2
import numpy as np
import torch.nn as nn
import torch.optim as optim
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from modeling.anime_gan_rewrite import ImageGenerator as Generator
from modeling.anime_gan_rewrite import ImageDiscriminator as Discriminator
from dataset import AnimeDataSet
from modeling.vgg import Vgg19
from tqdm import tqdm
import gc

gaussian_mean = torch.tensor(0.0)
gaussian_std = torch.tensor(0.1)

color_format_transform_kernel = torch.tensor([
    [0.299, -0.14714119, 0.61497538],
    [0.587, -0.28886916, -0.51496512],
    [0.114, 0.43601035, -0.10001026]
]).float()

if torch.cuda.is_available():
    color_format_transform_kernel = color_format_transform_kernel.cuda()

def gram_matrix_compute(input):
    b, c, w, h = input.size()
    x = input.view(b * c, w * h)
    G = torch.mm(x, x.T)
    return G.div(b * c * w * h)

def rgb_to_yuv(image):
    image = (image + 1.0) / 2.0
    yuv_img = torch.tensordot(
        image,
        color_format_transform_kernel,
        dims=([image.ndim - 3], [0]))
    return yuv_img

def load_checkpoint(model, checkpoint_dir, posfix=''):
    path = os.path.join(checkpoint_dir, f'{model.name}{posfix}.pth')
    checkpoint = torch.load(path,  map_location='cuda:0') if torch.cuda.is_available() else \
        torch.load(path,  map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    epoch = checkpoint['epoch']
    del checkpoint
    torch.cuda.empty_cache()
    gc.collect()
    return epoch

class ChromaticLoss(nn.Module):
    def __init__(self):
        super(ChromaticLoss, self).__init__()
        self.huber_loss = nn.SmoothL1Loss()
        self.l1_loss = nn.L1Loss()

    def forward(self, real_image, fake_image_generated):
        rgb_to_yuv_real = rgb_to_yuv(real_image)
        rgb_to_yuv_fake = rgb_to_yuv(fake_image_generated)
        return (self.l1_loss(rgb_to_yuv_real[:, :, :, 0], rgb_to_yuv_fake[:, :, :, 0]) +
                self.huber_loss(rgb_to_yuv_real[:, :, :, 1], rgb_to_yuv_fake[:, :, :, 1]) +
                self.huber_loss(rgb_to_yuv_real[:, :, :, 2], rgb_to_yuv_fake[:, :, :, 2]))

class AnimeGANLossCalculator:
    def __init__(self, args):
        self.args = args
        self.gramian_loss = nn.L1Loss().cuda()
        self.chromatic_loss = ChromaticLoss().cuda()
        self.content_loss = nn.L1Loss().cuda()
        self.vgg19_model = Vgg19().cuda().eval()

    def calculate_vgg_content_loss(self, real_image, fake_image):
        real_image_features = self.vgg19_model(real_image)
        fake_image_features = self.vgg19_model(fake_image)
        return self.content_loss(real_image_features, fake_image_features)

    def compute_generator_loss(self, fake_img, img, fake_logit, anime_gray):
        fake_feat = self.vgg19_model(fake_img)
        anime_feat = self.vgg19_model(anime_gray)
        img_feat = self.vgg19_model(img).detach()
        return [
            self.args.adversarial_loss_gen_weight * torch.mean(torch.square(fake_logit - 1.0)),
            self.args.content_loss_weight * self.content_loss(img_feat, fake_feat),
            self.args.gram_loss_weight * self.gramian_loss(gram_matrix_compute(anime_feat), gram_matrix_compute(fake_feat)),
            self.args.chromatic_loss_weight * self.chromatic_loss(img, fake_img),
        ]

    def compute_discriminator_loss(self, fake_img_d, real_anime_d, real_anime_gray_d, real_anime_smooth_gray_d):
        return self.args.adversarial_loss_disc_weight * (
            torch.mean(torch.square(real_anime_d - 1.0)) +
            torch.mean(torch.square(fake_img_d)) +
            torch.mean(torch.square(real_anime_gray_d)) +
            0.2 * torch.mean(torch.square(real_anime_smooth_gray_d))
        )
    
def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class LossTracker:
    def __init__(self):
        self.reset_epoch_tracking()

    def reset_epoch_tracking(self):
        self.chromatic_loss = []
        self.generator_adversarial_loss = []
        self.content_loss = []
        self.gramian_loss = []
        self.adversarial_discriminator_loss = []

    def modify_epoch_generator_loss(self, adv, gram, color, content):
        self.generator_adversarial_loss.append(adv.cpu().detach().numpy())
        self.gramian_loss.append(gram.cpu().detach().numpy())
        self.chromatic_loss.append(color.cpu().detach().numpy())
        self.content_loss.append(content.cpu().detach().numpy())

    def modify_epoch_discriminator_loss(self, loss):
        self.adversarial_discriminator_loss.append(loss.cpu().detach().numpy())

    def compute_average_epoch_generator_loss(self):
        generator_adversarial_epoch_loss_avg = self.compute_average(self.generator_adversarial_loss)
        gramian_epoch_loss_avg = self.compute_average(self.gramian_loss)
        chromatic_epoch_loss_avg = self.compute_average(self.chromatic_loss)
        content__epoch_loss_avg = self.compute_average(self.content_loss)
        return generator_adversarial_epoch_loss_avg, gramian_epoch_loss_avg, chromatic_epoch_loss_avg, content__epoch_loss_avg

    def compute_average_epoch_discriminator_loss(self):
        return self.compute_average(self.adversarial_discriminator_loss)

    def compute_average(self, losses):
        return sum(losses) / len(losses)
    
def denorm(np_images, dtype=None):
    np_images = np_images * 127.5 + 127.5
    np_images = np_images.astype(dtype)
    return np_images

def save_checkpoint(model, optimizer, epoch, args, posfix=''):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    path = os.path.join(args.checkpoint_dir, f'{model.name}{posfix}.pth')
    torch.save(checkpoint, path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Hayao')
    parser.add_argument('--data-dir', type=str, default='dataset')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--init-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--save-image-dir', type=str, default='images')
    parser.add_argument('--gan-loss', type=str, default='lsgan', help='lsgan / hinge / bce')
    parser.add_argument('--resume', type=str, default='False')
    parser.add_argument('--use_sn', action='store_true')
    parser.add_argument('--save-interval', type=int, default=1)
    parser.add_argument('--debug-samples', type=int, default=0)
    parser.add_argument('--lr-g', type=float, default=2e-4)
    parser.add_argument('--lr-d', type=float, default=4e-4)
    parser.add_argument('--init-lr', type=float, default=1e-3)
    parser.add_argument('--adversarial_loss_gen_weight', type=float, default=10.0, help='Adversarial loss weight for G')
    parser.add_argument('--adversarial_loss_disc_weight', type=float, default=10.0, help='Adversarial loss weight for D')
    parser.add_argument('--content_loss_weight', type=float, default=1.5, help='Content loss weight')
    parser.add_argument('--gram_loss_weight', type=float, default=3.0, help='Gram loss weight')
    parser.add_argument('--chromatic_loss_weight', type=float, default=30.0, help='Color loss weight')
    parser.add_argument('--d-layers', type=int, default=3, help='Discriminator conv layers')

    return parser.parse_args()


def collate_fn(batch):
    img, anime, anime_gray, anime_smt_gray = zip(*batch)
    return (
        torch.stack(img, 0),
        torch.stack(anime, 0),
        torch.stack(anime_gray, 0),
        torch.stack(anime_smt_gray, 0),
    )

def check_params(args):
    data_path = os.path.join(args.data_dir, args.dataset)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Dataset not found {data_path}')

    if not os.path.exists(args.save_image_dir):
        print(f'* {args.save_image_dir} does not exist, creating...')
        os.makedirs(args.save_image_dir)

    if not os.path.exists(args.checkpoint_dir):
        print(f'* {args.checkpoint_dir} does not exist, creating...')
        os.makedirs(args.checkpoint_dir)

    assert args.gan_loss in {'lsgan', 'hinge', 'bce'}, f'{args.gan_loss} is not supported'


def save_samples(generator, loader, args, max_imgs=2, subname='gen'):
    '''
    Generate and save images
    '''
    generator.eval()

    max_iter = (max_imgs // args.batch_size) + 1
    fake_imgs = []

    for i, (img, *_) in enumerate(loader):
        with torch.no_grad():
            fake_img = generator(img.cuda())
            fake_img = fake_img.detach().cpu().numpy()
            # Channel first -> channel last
            fake_img  = fake_img.transpose(0, 2, 3, 1)
            fake_imgs.append(denorm(fake_img, dtype=np.int16))

        if i + 1 == max_iter:
            break

    fake_imgs = np.concatenate(fake_imgs, axis=0)

    for i, img in enumerate(fake_imgs):
        save_path = os.path.join(args.save_image_dir, f'{subname}_{i}.jpg')
        cv2.imwrite(save_path, img[..., ::-1])

def main():
    args = parse_args()
    check_params(args)
    G = Generator(args.dataset).cuda()
    D = Discriminator(args).cuda()
    loss_tracker = LossTracker()
    anime_gan_loss_obj = AnimeGANLossCalculator(args)
    data_loader = DataLoader(
        AnimeDataSet(args),
        batch_size=args.batch_size,
        num_workers=cpu_count(),
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn,
    )
    optimizer_g = optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    start_epoch = 0
    if args.resume == 'GD':
        try:
            start_epoch = load_checkpoint(G, args.checkpoint_dir)
            print("G weight loaded")
            load_checkpoint(D, args.checkpoint_dir)
            print("D weight loaded")
        except Exception as e:
            print('Could not load checkpoint, train from scratch', e)
    elif args.resume == 'G':
        try:
            start_epoch = load_checkpoint(G, args.checkpoint_dir, posfix='_init')
        except Exception as e:
            print('Could not load G init checkpoint, train from scratch', e)

    for e in range(start_epoch, args.epochs):
        print(f"Epoch {e}/{args.epochs}")
        bar = tqdm(data_loader)
        G.train()
        init_losses = []
        if e < args.init_epochs:
            set_lr(optimizer_g, args.init_lr)
            for real_image, *_ in bar:
                real_image = real_image.cuda()
                optimizer_g.zero_grad()
                fake_img = G(real_image)
                loss = anime_gan_loss_obj.calculate_vgg_content_loss(real_image, fake_img)
                loss.backward()
                optimizer_g.step()
                init_losses.append(loss.cpu().detach().numpy())
                avg_content_loss = sum(init_losses) / len(init_losses)
                bar.set_description(f'[Init Training G] content loss: {avg_content_loss:2f}')
            # Save under init epoch condition
            set_lr(optimizer_g, args.lr_g)
            save_checkpoint(G, optimizer_g, e, args, posfix='_init')
            save_samples(G, data_loader, args, subname='initg')
            continue

        loss_tracker.reset_epoch_tracking()
        for real_image, anime, anime_gray, anime_smt_gray in bar:
            real_image = real_image.cuda()
            anime = anime.cuda()
            anime_gray = anime_gray.cuda()
            anime_smt_gray = anime_smt_gray.cuda()
            optimizer_d.zero_grad()
            fake_img = G(real_image).detach()
            fake_d = D(fake_img)
            real_anime_d = D(anime)
            real_anime_gray_d = D(anime_gray)
            real_anime_smt_gray_d = D(anime_smt_gray)
            loss_d = anime_gan_loss_obj.compute_discriminator_loss(fake_d, real_anime_d, real_anime_gray_d, real_anime_smt_gray_d)
            loss_d.backward()
            optimizer_d.step()
            loss_tracker.modify_epoch_discriminator_loss(loss_d)

            optimizer_g.zero_grad()
            fake_img = G(real_image)
            fake_d = D(fake_img)
            adv_loss, con_loss, gra_loss, col_loss = anime_gan_loss_obj.compute_generator_loss(fake_img, real_image, fake_d, anime_gray)
            loss_g = adv_loss + con_loss + gra_loss + col_loss
            loss_g.backward()
            optimizer_g.step()
            loss_tracker.modify_epoch_generator_loss(adv_loss, gra_loss, col_loss, con_loss)
            avg_adv, avg_gram, avg_color, avg_content = loss_tracker.compute_average_epoch_generator_loss()
            avg_adv_d = loss_tracker.compute_average_epoch_discriminator_loss()
            bar.set_description(f'loss G: adv {avg_adv:2f} con {avg_content:2f} gram {avg_gram:2f} color {avg_color:2f} / loss D: {avg_adv_d:2f}')

        # Save the model at specific intervals
        if e % args.save_interval == 0:
            save_checkpoint(G, optimizer_g, e, args)
            save_checkpoint(D, optimizer_d, e, args)
            save_samples(G, data_loader, args)


if __name__ == '__main__':
    main()
