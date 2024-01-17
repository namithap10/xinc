import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers


def pad_input(img,block_size):
    """
        pad input image tensor so that it is divisible by block size
        Image - NCHW.
        Will return ** original image ** if it is already divisible by block size.
    """
    w_pad_size = h_pad_size = 0
    w_padding = h_padding = (0,0)

    if img.shape[3] % block_size != 0:
        w_pad_size = (block_size - img.shape[3] % block_size) % block_size
    
    if img.shape[2] % block_size != 0:
        h_pad_size = (block_size - img.shape[2] % block_size) % block_size

    if w_pad_size == 0 and h_pad_size == 0:
        return img

    if w_pad_size!=0:
        if w_pad_size%2==0:
            w_padding = (w_pad_size//2,w_pad_size//2)
        else:
            w_padding = (w_pad_size//2,w_pad_size//2+1)


    if h_pad_size!=0:
        if h_pad_size%2==0:
            h_padding = (h_pad_size//2,h_pad_size//2)
        else:
            h_padding = (h_pad_size//2,h_pad_size//2+1)

    #pad = nn.ConstantPad2d( (w_padding[0],w_padding[1],h_padding[0],h_padding[1]), value=0)
    #use zero padding
    pad = nn.ZeroPad2d( (w_padding[0],w_padding[1],h_padding[0],h_padding[1]))

    img = pad(img)
    
    return img

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels = 3, \
                 dim_head = 64,num_classes=None,vae_mode=False):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        self.patch_size = patch_size

        self.num_classes = num_classes
        self.vae_mode = vae_mode
        self.latent_dim = dim

        if self.vae_mode:
            dim = dim * 2 
            print('VAE mode is on, dim is doubled')
        

        # pad function takes care. 
        #assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)


        self.to_latent = nn.Identity()
        if num_classes is not None:
            self.linear_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_classes)
            )

    def forward(self, img):
        img = pad_input(img,self.patch_size)

        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)

        if self.vae_mode:
            mu,logvar = torch.split(x,self.latent_dim,dim=-1)
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            z = mu + eps*std # reparameterization trick

            return z,mu,logvar

        if self.num_classes is not None:
            return self.linear_head(x)  
        
        return x

if __name__ == '__main__':
    model = SimpleViT(image_size=178,patch_size=8,dim=512,depth=2,mlp_dim=1024,heads=1,vae_mode=False).cuda()
    img = torch.randn(1, 3, 178, 178).cuda()
    preds = model(img)
    print(preds.shape)
