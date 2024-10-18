import torch
from torch.nn import parallel
import torch.nn as nn
import torch.nn.functional as F

"""
Code for transformer architecture
"""

class PatchEmbed(nn.Module): #Note, from Meta DINO ViT code ()
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=64, patch_size=16, in_chans=3, embed_dim=512):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        y = self.proj(x)
        #print('conv output before flattening and transpose', y.shape)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x