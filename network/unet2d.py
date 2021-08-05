import torch
import torch.nn as nn
import torch.nn.functional as F

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module,
                                                                                        nn.ConvTranspose2d) or isinstance(
                module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

class encoder(nn.Module):
    def __init__(self, in_channels, initial_filter_size, kernel_size, do_instancenorm, dropout):
        super().__init__()
        self.contr_1_1 = self.contract(in_channels, initial_filter_size, kernel_size, instancenorm=do_instancenorm)
        self.contr_1_2 = self.contract(initial_filter_size, initial_filter_size, kernel_size,
                                       instancenorm=do_instancenorm)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.contr_2_1 = self.contract(initial_filter_size, initial_filter_size * 2, kernel_size,
                                       instancenorm=do_instancenorm)
        self.contr_2_2 = self.contract(initial_filter_size * 2, initial_filter_size * 2, kernel_size,
                                       instancenorm=do_instancenorm)

        self.contr_3_1 = self.contract(initial_filter_size * 2, initial_filter_size * 2 ** 2, kernel_size,
                                       instancenorm=do_instancenorm)
        self.contr_3_2 = self.contract(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 2, kernel_size,
                                       instancenorm=do_instancenorm)

        self.contr_4_1 = self.contract(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 3, kernel_size,
                                       instancenorm=do_instancenorm, dropout=dropout)
        self.contr_4_2 = self.contract(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 3, kernel_size,
                                       instancenorm=do_instancenorm, dropout=dropout)
        self.center = nn.Sequential(
            nn.Conv2d(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 4, 3, padding=1),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 4, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        contr_1 = self.contr_1_2(self.contr_1_1(x))
        pool = self.pool(contr_1)

        contr_2 = self.contr_2_2(self.contr_2_1(pool))
        pool = self.pool(contr_2)

        contr_3 = self.contr_3_2(self.contr_3_1(pool))
        pool = self.pool(contr_3)

        contr_4 = self.contr_4_2(self.contr_4_1(pool))
        pool = self.pool(contr_4)

        out = self.center(pool)
        return out, contr_4, contr_3, contr_2, contr_1
        
    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, instancenorm=True, dropout=0):
        if instancenorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.Dropout2d(dropout),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.Dropout2d(dropout),
                nn.LeakyReLU(inplace=True))
        return layer

class decoder(nn.Module):
    def __init__(self, initial_filter_size, classes, dropout):
        super().__init__()
        # self.concat_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.upscale5 = nn.ConvTranspose2d(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 3, kernel_size=2,
                                           stride=2)
        self.expand_4_1 = self.expand(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 3, 3, dropout)
        self.expand_4_2 = self.expand(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 3, 3, dropout)
        self.upscale4 = nn.ConvTranspose2d(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 2, kernel_size=2,
                                           stride=2)

        self.expand_3_1 = self.expand(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 2)
        self.expand_3_2 = self.expand(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 2)
        self.upscale3 = nn.ConvTranspose2d(initial_filter_size * 2 ** 2, initial_filter_size * 2, 2, stride=2)

        self.expand_2_1 = self.expand(initial_filter_size * 2 ** 2, initial_filter_size * 2)
        self.expand_2_2 = self.expand(initial_filter_size * 2, initial_filter_size * 2)
        self.upscale2 = nn.ConvTranspose2d(initial_filter_size * 2, initial_filter_size, 2, stride=2)

        self.expand_1_1 = self.expand(initial_filter_size * 2, initial_filter_size)
        self.expand_1_2 = self.expand(initial_filter_size, initial_filter_size)
        
        self.head = nn.Sequential(
                nn.Conv2d(initial_filter_size, classes, kernel_size=1,
                          stride=1, bias=False))

    def forward(self, x, contr_4, contr_3, contr_2, contr_1):

        concat_weight = 1
        upscale = self.upscale5(x)
        crop = self.center_crop(contr_4, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)

        expand = self.expand_4_2(self.expand_4_1(concat))
        out2 = expand
        upscale = self.upscale4(expand)

        crop = self.center_crop(contr_3, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)
 
        expand = self.expand_3_2(self.expand_3_1(concat))
        out3 = expand
        upscale = self.upscale3(expand)

        crop = self.center_crop(contr_2, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)

        expand = self.expand_2_2(self.expand_2_1(concat))
        out4 = expand
        upscale = self.upscale2(expand)

        crop = self.center_crop(contr_1, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)

        expand = self.expand_1_2(self.expand_1_1(concat))

        out = self.head(expand)
        return out

    @staticmethod
    def center_crop(layer, target_width, target_height):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy1:(xy1 + target_width), xy2:(xy2 + target_height)]

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3, dropout=0):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.Dropout2d(dropout),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
        return layer

class decoder_ds(nn.Module):
    def __init__(self, initial_filter_size, classes):
        super().__init__()
        # self.concat_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.upscale5 = nn.ConvTranspose2d(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 3, kernel_size=2,
                                           stride=2)
        self.expand_4_1 = self.expand(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 3)
        self.expand_4_2 = self.expand(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 3)
        self.upscale4 = nn.ConvTranspose2d(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 2, kernel_size=2,
                                           stride=2)

        self.expand_3_1 = self.expand(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 2)
        self.expand_3_2 = self.expand(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 2)
        self.upscale3 = nn.ConvTranspose2d(initial_filter_size * 2 ** 2, initial_filter_size * 2, 2, stride=2)

        self.expand_2_1 = self.expand(initial_filter_size * 2 ** 2, initial_filter_size * 2)
        self.expand_2_2 = self.expand(initial_filter_size * 2, initial_filter_size * 2)
        self.upscale2 = nn.ConvTranspose2d(initial_filter_size * 2, initial_filter_size, 2, stride=2)

        self.expand_1_1 = self.expand(initial_filter_size * 2, initial_filter_size)
        self.expand_1_2 = self.expand(initial_filter_size, initial_filter_size)
        
        # use deep supervision
        self.ds22_1x1_conv = nn.Sequential(
            nn.Conv2d(initial_filter_size * 2, classes,  kernel_size=1, stride=1, bias=False),
            nn.LeakyReLU(inplace=True)
        )
        self.ds32_1x1_conv = nn.Sequential(
            nn.Conv2d(initial_filter_size * 2 * 2, classes,  kernel_size=1, stride=1, bias=False),
            nn.LeakyReLU(inplace=True)
        )
        self.head = nn.Sequential(
                nn.Conv2d(initial_filter_size, classes, kernel_size=1,
                          stride=1, bias=False))

    def forward(self, x, contr_4, contr_3, contr_2, contr_1):

        concat_weight = 1
        upscale = self.upscale5(x)
        crop = self.center_crop(contr_4, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)

        expand = self.expand_4_2(self.expand_4_1(concat))
        out2 = expand
        upscale = self.upscale4(expand)

        crop = self.center_crop(contr_3, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)
 
        expand = self.expand_3_2(self.expand_3_1(concat))
        out3 = expand
        expand_ds32 = self.ds32_1x1_conv(expand)
        upscale = self.upscale3(expand)

        crop = self.center_crop(contr_2, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)

        expand = self.expand_2_2(self.expand_2_1(concat))
        out4 = expand
        expand_ds22 = self.ds22_1x1_conv(expand)
        upscale = self.upscale2(expand)

        crop = self.center_crop(contr_1, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)

        expand = self.expand_1_2(self.expand_1_1(concat))

        out = self.head(expand)
        # use deep supervision
        expand_ds32_up = nn.functional.interpolate(expand_ds32, scale_factor=2, mode='bilinear', align_corners=False)
        expand_ds22 = expand_ds22 + expand_ds32_up
        expand_ds22_up = nn.functional.interpolate(expand_ds22, scale_factor=2, mode='bilinear', align_corners=False)
        out = out + expand_ds22_up
        return out, out2, out3, out4

    @staticmethod
    def center_crop(layer, target_width, target_height):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy1:(xy1 + target_width), xy2:(xy2 + target_height)]

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
        return layer

class UNet2D(nn.Module):
    def __init__(self, in_channels=1, initial_filter_size=32, kernel_size=3, classes=4, do_instancenorm=True, dropout=0):
        super().__init__()
        
        self.encoder = encoder(in_channels, initial_filter_size, kernel_size, do_instancenorm, dropout)
        self.decoder = decoder(initial_filter_size, classes, dropout)

        self.apply(InitWeights_He(1e-2))

    def forward(self, x):

        x_1, contr_4, contr_3, contr_2, contr_1 = self.encoder(x)
        out = self.decoder(x_1, contr_4, contr_3, contr_2, contr_1)
        return out

class UNet2D_ds(nn.Module):
    def __init__(self, in_channels=1, initial_filter_size=32, kernel_size=3, classes=4, do_instancenorm=True):
        super().__init__()
        
        self.encoder = encoder(in_channels, initial_filter_size, kernel_size, do_instancenorm)
        self.decoder = decoder_ds(initial_filter_size, classes)

        self.apply(InitWeights_He(1e-2))

    def forward(self, x):

        x_1, contr_4, contr_3, contr_2, contr_1 = self.encoder(x)
        out, _, _, _ = self.decoder(x_1, contr_4, contr_3, contr_2, contr_1)
        return out

if __name__ == '__main__':
    input = torch.randn(1,1,256,256)
    model = UNet2D(in_channels=1, initial_filter_size=32, kernel_size=3, classes=2, dropout=1)
    out = model(input)
    print(f'out:{out.shape}')
