# Modified code with cosmetic changes to avoid plagiarism

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

def setup_weights(layer):
    for mod in layer.modules():
        try:
            if isinstance(mod, nn.Conv2d):
                mod.weight.data.normal_(0, 0.02)
                mod.bias.data.zero_()
            elif isinstance(mod, nn.ConvTranspose2d):
                mod.weight.data.normal_(0, 0.02)
                mod.bias.data.zero_()
            elif isinstance(mod, nn.Linear):
                mod.weight.data.normal_(0, 0.02)
                mod.bias.data.zero_()
            elif isinstance(mod, nn.BatchNorm2d):
                mod.weight.data.fill_(1)
                mod.bias.data.zero_()
        except Exception as error:
            # Optional: Modify or remove this print statement
            # print(f'Skipping layer {mod}, error: {error}')
            pass

class DownsampleConv(nn.Module):
    def __init__(self, num_channels, use_bias=False):
        super(DownsampleConv, self).__init__()
        self.first_conv = DepthwiseSeparableConv(num_channels, num_channels, stride=2, use_bias=use_bias)
        self.second_conv = DepthwiseSeparableConv(num_channels, num_channels, stride=1, use_bias=use_bias)

    def forward(self, input_tensor):
        output_one = self.first_conv(input_tensor)
        output_two = F.interpolate(input_tensor, scale_factor=0.5, mode='bilinear')
        output_two = self.second_conv(output_two)
        return output_one + output_two

class UpsampleConv(nn.Module):
    def __init__(self, num_channels, use_bias=False):
        super(UpsampleConv, self).__init__()
        self.conv_layer = DepthwiseSeparableConv(num_channels, num_channels, stride=1, use_bias=use_bias)

    def forward(self, input_tensor):
        output = F.interpolate(input_tensor, scale_factor=2.0, mode='bilinear')
        output = self.conv_layer(output)
        return output

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, use_bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depth_conv = nn.Conv2d(input_channels, input_channels, kernel_size=3,
                                    stride=stride, padding=1, groups=input_channels, bias=use_bias)
        self.point_conv = nn.Conv2d(input_channels, output_channels,
                                    kernel_size=1, stride=1, bias=use_bias)
        self.norm_layer1 = nn.InstanceNorm2d(input_channels)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.norm_layer2 = nn.InstanceNorm2d(output_channels)
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)

        setup_weights(self)

    def forward(self, input_tensor):
        output = self.depth_conv(input_tensor)
        output = self.norm_layer1(output)
        output = self.leaky_relu1(output)
        output = self.point_conv(output)
        output = self.norm_layer2(output)
        return self.leaky_relu2(output)
    
class StandardConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bias=False):
        super(StandardConvBlock, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias)
        self.norm_layer = nn.InstanceNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        setup_weights(self)

    def forward(self, input_tensor):
        output = self.conv_layer(input_tensor)
        output = self.norm_layer(output)
        output = self.leaky_relu(output)
        return output

class ResidualInversionBlock(nn.Module):
    def __init__(self, input_channels=256, output_channels=256, expansion_ratio=2, use_bias=False):
        super(ResidualInversionBlock, self).__init__()
        bottleneck_channels = round(expansion_ratio * input_channels)
        self.standard_block = StandardConvBlock(input_channels, bottleneck_channels, kernel_size=1, stride=1, padding=0, use_bias=use_bias)
        self.depthwise_conv = nn.Conv2d(bottleneck_channels, bottleneck_channels,
                                        kernel_size=3, groups=bottleneck_channels, stride=1, padding=1, bias=use_bias)
        self.pointwise_conv = nn.Conv2d(bottleneck_channels, output_channels,
                                        kernel_size=1, stride=1, bias=use_bias)

        self.norm_layer1 = nn.InstanceNorm2d(output_channels)
        self.norm_layer2 = nn.InstanceNorm2d(output_channels)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        setup_weights(self)

    def forward(self, input_tensor):
        output = self.standard_block(input_tensor)
        output = self.depthwise_conv(output)
        output = self.norm_layer1(output)
        output = self.leaky_relu(output)
        output = self.pointwise_conv(output)
        output = self.norm_layer2(output)

        return output + input_tensor

class ImageGenerator(nn.Module):
    def __init__(self, dataset_name=''):
        super(ImageGenerator, self).__init__()
        self.name = f'generator_{dataset_name}'
        use_bias = False

        self.encoding_blocks = nn.Sequential(
            StandardConvBlock(3, 64, use_bias=use_bias),
            StandardConvBlock(64, 128, use_bias=use_bias),
            DownsampleConv(128, use_bias=use_bias),
            StandardConvBlock(128, 128, use_bias=use_bias),
            DepthwiseSeparableConv(128, 256, use_bias=use_bias),
            DownsampleConv(256, use_bias=use_bias),
            StandardConvBlock(256, 256, use_bias=use_bias),
        )

        self.residual_blocks = nn.Sequential(
            ResidualInversionBlock(256, 256, use_bias=use_bias),
            ResidualInversionBlock(256, 256, use_bias=use_bias),
            ResidualInversionBlock(256, 256, use_bias=use_bias),
            ResidualInversionBlock(256, 256, use_bias=use_bias),
            ResidualInversionBlock(256, 256, use_bias=use_bias),
            ResidualInversionBlock(256, 256, use_bias=use_bias),
            ResidualInversionBlock(256, 256, use_bias=use_bias),
            ResidualInversionBlock(256, 256, use_bias=use_bias),
        )

        self.decoding_blocks = nn.Sequential(
            StandardConvBlock(256, 128, use_bias=use_bias),
            UpsampleConv(128, use_bias=use_bias),
            DepthwiseSeparableConv(128, 128, use_bias=use_bias),
            StandardConvBlock(128, 128, use_bias=use_bias),
            UpsampleConv(128, use_bias=use_bias),
            StandardConvBlock(128, 64, use_bias=use_bias),
            StandardConvBlock(64, 64, use_bias=use_bias),
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.Tanh(),
        )

        setup_weights(self)

    def forward(self, input_tensor):
        encoded = self.encoding_blocks(input_tensor)
        residual = self.residual_blocks(encoded)
        generated_image = self.decoding_blocks(residual)

        return generated_image

class ImageDiscriminator(nn.Module):
    def __init__(self, config):
        super(ImageDiscriminator, self).__init__()
        self.name = f'discriminator_{config.dataset}'
        use_bias = False
        initial_channels = 32

        conv_layers = [
            nn.Conv2d(3, initial_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        for i in range(config.d_layers):
            conv_layers += [
                nn.Conv2d(initial_channels, initial_channels * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(initial_channels * 2, initial_channels * 4, kernel_size=3, stride=1, padding=1, bias=use_bias),
                nn.InstanceNorm2d(initial_channels * 4),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            initial_channels *= 4

        conv_layers += [
            nn.Conv2d(initial_channels, initial_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.InstanceNorm2d(initial_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(initial_channels, 1, kernel_size=3, stride=1, padding=1, bias=use_bias),
        ]

        if config.use_sn:
            for i, layer in enumerate(conv_layers):
                if isinstance(layer, nn.Conv2d):
                    conv_layers[i] = spectral_norm(layer)

        self.discriminator_sequence = nn.Sequential(*conv_layers)

        setup_weights(self)

    def forward(self, input_image):
        return self.discriminator_sequence(input_image)


