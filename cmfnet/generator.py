'''
follow implementation in https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py
'''
import torch.nn as nn
import torch



class SeperateMaskGenerator(nn.Module):
    def __init__(self, latent_dim, num_masks, img_size=64, use_fp16=False, channel_list=[384, 256, 128, 64],
                 num_groups=32):
        super(SeperateMaskGenerator, self).__init__()
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.init_size = img_size // 2**4 # 4 times of x2 upsample
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 384 * self.init_size ** 2))
        self.num_masks = num_masks
        self.latent_dim = latent_dim
        self.conv_blocks = nn.ModuleList()
        in_dim = 384
        for out_dim in channel_list:
            conv_block = nn.Sequential(
                nn.GroupNorm(num_groups, in_dim),  # 4
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_dim, out_dim, 3, stride=1, padding=1, bias=False),
                nn.GroupNorm(num_groups, out_dim),
                nn.SiLU(),  # 8
            )
            in_dim = out_dim
            self.conv_blocks.append(conv_block)
        self.conv_blocks_img = nn.Conv2d(in_dim, 1, 3, stride=1, padding=1)
        self.mask_normalize_block = nn.Softmax(dim=1)  # mask

    def forward(self, z):
        # size of input z: N x num_masks x mask_code
        # convert z from N x num_masks x mask_code to (N x num_masks) x mask_code
        N, num_masks, mask_code_dim = z.size()
        assert self.num_masks == num_masks
        assert self.latent_dim == mask_code_dim
        z = z.view(N * num_masks, mask_code_dim)
        out = self.l1(z)
        out = out.view(out.shape[0], 384, self.init_size, self.init_size)
        for block in self.conv_blocks:
            out = block(out)
        out = self.conv_blocks_img(out)
        _, _, H, W = out.size()
        out = out.view(N, num_masks, H, W)
        out = self.mask_normalize_block(out)
        return out
