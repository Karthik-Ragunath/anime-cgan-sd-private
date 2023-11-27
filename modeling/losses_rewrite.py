import torch
import torch.nn.functional as F
import torch.nn as nn
from modeling.vgg import Vgg19

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
            self.args.wadvg * torch.mean(torch.square(fake_logit - 1.0)),
            self.args.wcon * self.content_loss(img_feat, fake_feat),
            self.args.wgra * self.gramian_loss(gram_matrix_compute(anime_feat), gram_matrix_compute(fake_feat)),
            self.args.wcol * self.chromatic_loss(img, fake_img),
        ]

    def compute_discriminator_loss(self, fake_img_d, real_anime_d, real_anime_gray_d, real_anime_smooth_gray_d):
        return self.args.wadvd * (
            torch.mean(torch.square(real_anime_d - 1.0)) +
            torch.mean(torch.square(fake_img_d)) +
            torch.mean(torch.square(real_anime_gray_d)) +
            0.2 * torch.mean(torch.square(real_anime_smooth_gray_d))
        )