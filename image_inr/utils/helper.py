from builtins import breakpoint
import imp
import numpy as np
from utils.state_tools import StateDictOperator
from zmq import device
import torch
import torch.nn as nn
import torch.distributed as dist
from torch._C import dtype
from typing import Dict

from . import dct,patching

import torchvision.transforms as transforms

import json,os,pickle
import cv2
from PIL import Image
import glob 
import av
import av.datasets
import math
import matplotlib.pyplot as plt
import random,sys
import importlib
from omegaconf import DictConfig,OmegaConf
from dahuffman import HuffmanCodec
from pytorch_msssim import ssim
import compress_pickle
import clip

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


DTYPE_BIT_SIZE: Dict[dtype, int] = {
    torch.float32: 32,
    torch.float: 32,
    torch.float64: 64,
    torch.double: 64,
    torch.float16: 16,
    torch.half: 16,
    torch.bfloat16: 16,
    torch.complex32: 32,
    torch.complex64: 64,
    torch.complex128: 128,
    torch.cdouble: 128,
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.short: 16,
    torch.int32: 32,
    torch.int: 32,
    torch.int64: 64,
    torch.long: 64,
    torch.bool: 1
}

def quantize_per_tensor(t, bit=8, axis=-1):
    if axis == -1:
        t_valid = t!=0
        if t_valid.sum()==0:
            scale = torch.tensor(0).to(t.device)
            t_min = torch.tensor(0).to(t.device)
        else:
            t_min, t_max =  t[t_valid].min(), t[t_valid].max()
            scale = (t_max - t_min) / 2**bit
    elif axis == 0:
        min_max_list = []
        for i in range(t.size(0)):
            t_valid = t[i]!=0
            if t_valid.sum():
                min_max_list.append([t[i][t_valid].min(), t[i][t_valid].max()])
            else:
                min_max_list.append([0, 0])
        min_max_tf = torch.tensor(min_max_list).to(t.device)        
        scale = (min_max_tf[:,1] - min_max_tf[:,0]) / 2**bit
        if t.dim() == 4:
            scale = scale[:,None,None,None]
            t_min = min_max_tf[:,0,None,None,None]
        elif t.dim() == 2:
            scale = scale[:,None]
            t_min = min_max_tf[:,0,None]
    elif axis == 1:
        min_max_list = []
        for i in range(t.size(1)):
            t_valid = t[:,i]!=0
            if t_valid.sum():
                min_max_list.append([t[:,i][t_valid].min(), t[:,i][t_valid].max()])
            else:
                min_max_list.append([0, 0])
        min_max_tf = torch.tensor(min_max_list).to(t.device)             
        scale = (min_max_tf[:,1] - min_max_tf[:,0]) / 2**bit
        if t.dim() == 4:
            scale = scale[None,:,None,None]
            t_min = min_max_tf[None,:,0,None,None]
        elif t.dim() == 2:
            scale = scale[None,:]
            t_min = min_max_tf[None,:,0]            
     
    quant_t = ((t - t_min) / (scale + 1e-19)).round()
    #new_t = t_min + scale * quant_t #reconstruction.
    #return quant_t, new_t,scale,t_min
    return quant_t, scale,t_min


class CustomParallelWrapper(nn.Module):
    def __init__(self, module, device_ids):
        super().__init__()
        self.module = module
        self.device_ids = device_ids

    def forward(self, input1, input2,z=None):
        replicas = nn.parallel.replicate(self.module, self.device_ids)
        inputs2 = nn.parallel.scatter(input2, self.device_ids)
        # manually broadcast input1 to all devices
        inputs1 = [input1.to(device) for device in self.device_ids]
        inputs = list(zip(inputs1, inputs2))
        outputs = nn.parallel.parallel_apply(replicas, inputs)

        return nn.parallel.gather(outputs, self.device_ids[0])



def all_gather(tensors):
    """
    All gathers the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all gather across all processes in
        all machines.
    """

    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [
            torch.ones_like(tensor) for _ in range(world_size)
        ]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    
    print(gather_list)

    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
        #output_tensor.append(torch.stack(gathered_tensor, dim=0))

    return output_tensor


def all_reduce(tensors, average=True):
    """
    All reduce the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all reduce across all processes in
        all machines.
        average (bool): scales the reduced tensor by the number of overall
        processes across all machines.
    """

    for tensor in tensors:
        dist.all_reduce(tensor, async_op=False)
    if average:
        world_size = dist.get_world_size()
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors

def get_features_coord(img,args):
    
    """
        Patch shape and features shape will be the same for all images in a video. 
        Also either dct_patches or rgb_patches should be true.
    """
    #patch size is same as block size
    patch_shape = (args.block_size,args.block_size)

    #these features are nchw hw =  patch size. 

    if args.dct_patches:
        coordinates,features,features_shape = to_coordinates_and_features_patches_DCT(img,patch_shape=patch_shape,block_size=args.block_size,pad=True)
    elif args.rgb_patches: 
        coordinates,features,features_shape = to_coordinates_and_features_patches(img,patch_shape=patch_shape) 
    elif args.dct:
        coordinates,features,features_shape = to_coordinates_and_features_DCT(img,patch_shape=patch_shape,block_size=args.block_size,pad=True)
    else:
        coordinates,features,features_shape = to_coordinates_and_features(img,patch_shape=patch_shape)

    coordinates, features = coordinates.to(img.device, img.dtype), features.to(img.device, img.dtype)

    return coordinates,features,features_shape

def get_ssim(preds,frame):
    """
        Computes SSIM between preds and frame. 
    """
    
    return ssim(preds,frame,data_range=1.0)


def get_patch_wise_psnr(predicted_img,gt_img,args):

    _,gt_img_patches,_ = get_features_coord(gt_img,args)
    _,predicted_img_patches,_ = get_features_coord(predicted_img,args)

    psnr_list = []
    for i in range(gt_img_patches.shape[0]):
        psnr_list.append(psnr(gt_img_patches[i],predicted_img_patches[i]))

    full_psnr = get_clamped_psnr(predicted_img,gt_img)

    return full_psnr,psnr_list


def filter_state(state,pattern):
    """
        replace pattern with empty string in all keys of state dict.
    """
    new_state = {}
    for k,v in state.items():
        new_state[k.replace(pattern,'')] = v
    return new_state


def anneal_beta(epoch, max_epochs, start_beta, final_beta,warmup):
    """
    Anneals the beta parameter from start_beta to final_beta over the course of training.

    Parameters:
        epoch (int): The current epoch.
        max_epochs (int): The total number of epochs for training.
        start_beta (float): The initial value of beta at epoch 0.
        final_beta (float): The target value of beta at the final epoch.

    Returns:
        float: The annealed beta value for the current epoch.
    """

    if epoch < warmup:
        return start_beta

    annealed_beta = start_beta + (final_beta - start_beta) * epoch / max_epochs
    return annealed_beta




def patchwise_greater_than(psnr_1,psnr_2):
    psnr_1 = np.array(psnr_1)
    psnr_2 = np.array(psnr_2)

    #compare each patch psnr and select the list with highest wins. 
    #return the list with highest wins.
    temp = psnr_1 >= psnr_2

    if temp.sum() > len(temp)//2:
        return True
    return False	


def flatten_dictconfig(config, parent_key='', sep='_'):
    """
    This function flattens an OmegaConf DictConfig object.
    """
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, DictConfig):
            items.extend(flatten_dictconfig(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)




def add_noise(frame,noise_type,gauss_std=0.1,noise_amp=1,noise_amount=int(1e4)):

    """
        Called by loader. CHW.  
    """

    N,C,H,W = frame.size()
    frame = frame.clone()

    for i in range(N):

        #generate indices to place noise
        random_noise_ind = np.random.choice(W*H, size=noise_amount, replace=False)
        random_noise_x = random_noise_ind % W
        random_noise_y = random_noise_ind // W

        if noise_type in ['all_white','all_black','salt_pepper']:
            
            if noise_type == 'salt_pepper':
                noise_val = np.random.choice(2, size=noise_amount) 
                noise_val = np.stack((noise_val, noise_val, noise_val), axis=1)
            
                frame[i,:,random_noise_y, random_noise_x] = torch.from_numpy(noise_val).float().T.to(frame.device)

            elif noise_type == 'all_white':
                noise_val = 1
                frame[i,:,random_noise_y, random_noise_x] = noise_val
            elif noise_type == 'all_black':
                noise_val = 0
                frame[i,:,random_noise_y, random_noise_x] = noise_val

        elif noise_type == 'gaussian':
            noise_val = np.random.normal(0, gauss_std, (noise_amount, 3)) * noise_amp
            #frame[:,random_noise_y, random_noise_x] += torch.from_numpy(noise_val).float().T
            frame[i,:,random_noise_y, random_noise_x] += torch.from_numpy(noise_val).float().T.to(frame.device)

        elif noise_type =='random':
            noise_val = (np.random.rand(noise_amount, 3) * 2 - 1) * noise_amp
            #frame[:,random_noise_y, random_noise_x] += torch.from_numpy(noise_val).float().T
            frame[i,:,random_noise_y, random_noise_x] += torch.from_numpy(noise_val).float().T.to(frame.device)

        frame[i].clamp_(0,1)
    
    return frame

#dct_patches=False,rgb_patches=False,rgb_patch_volume=False,dct=False):
def process_outputs(out,features_shape,input_img_shape,patch_shape=None,**kwargs): 
  
        
    """
        For all DCT transorms, unpatch -> inv_dct -> unpad -> crop
        For all RGB transforms, unpatch -> unpad -> crop

        Input_img_shape is NCHW. 
        
    """

    N,C,H,W = input_img_shape

    
    if kwargs['type'] == 'dct_patches' or kwargs['type'] == 'rgb_patches':
        patcher = patching.Patcher(patch_shape)
        out_reshape = patcher.unpatch(out,features_shape)
    
    
    elif kwargs['type'] == 'rgb_patch_volume':
        vol_patch_shape = (3,patch_shape[0],patch_shape[1])
        patcher = patching.Patcher(vol_patch_shape)
        out_reshape = patcher.unpatch(out,features_shape)

    else:
        N,C,H,W = features_shape #N=1
        out_reshape = out.reshape(H,W,C).permute(2,0,1)

    #if dct or dct_patches:											
    if 'dct' in kwargs or 'dct_patches' in kwargs:
        out_reshape = dct.batch_idct(out_reshape.unsqueeze(0),device=out_reshape.device,block_size=patch_shape[0])

    #unpad wont hurt even if there is no padding
    out_reshape = unpad(out_reshape,(H,W))

    #out is between 0 and 1. Else causes artifacts.
    out_reshape.clamp_(0,1)
    

    if out_reshape.dim() == 3:
        out_reshape = out_reshape.unsqueeze(0)

    return out_reshape


def unpad(img,out_shape):
    """
        Undo padding. Since we do constant padding, we can just do a center crop.
    """
    unpadded_tensor = transforms.CenterCrop(out_shape)(img)
    return unpadded_tensor

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path,exist_ok=True)

def dump_to_json(data,filename):
    """
        Dumps data to a json file.
    """
    with open(filename,'w') as f:
        json.dump(data,f)

def load_pickle(filename,compressed=False):
    with open(filename, 'rb') as f:
        if compressed:
            return compress_pickle.load(f,compression='lzma')
        return pickle.load(f)

def load_json(filename):
    """
        Loads data from a json file.
    """
    with open(filename,'r') as f:
        return json.load(f)

def save_to_file(data,filename):
    """
        Saves data to a pickle file.
    """
    with open(filename,'w') as f:
        f.write(str(data))

def save_pickle(data,filename,compressed=False):
    with open(filename, 'wb') as f:
        if compressed:
            compress_pickle.dump(data, f,compression='lzma',set_default_extension=False,pickler_method='optimized_pickle')
        else:
            pickle.dump(data, f,protocol=pickle.HIGHEST_PROTOCOL)

def save_numpy_img(img,filename):
    #convert BGR to RGB
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imwrite(filename,img)

def save_videos(filename='video.avi',image_list = None,img_folder=None,**kwargs):
    """
        Saves a list of images to a video file.
    """

    assert img_folder is not None or image_list is not None, "Either image_list or img_folder must be provided."

    if image_list is None:
        image_list = glob.glob(img_folder+'/*/*.png')
        image_list = sorted(image_list,key=lambda x: int(x.split('_')[-1].split('.')[0]))

    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    fps = kwargs.get('fps',5)

    im = cv2.imread(image_list[0])
    height, width, layers = im.shape
    
    out = cv2.VideoWriter(filename,fourcc,fps,(width,height))

    for img_file in image_list:
        img = cv2.imread(img_file)
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        out.write(img)

    out.release()

def dict_to_str_vals(d):
    """
        Converts all keys,values of a dictionary to strings.
    """
    d_copy = {}
    for k,v in d.items():
        d_copy[str(k)] = str(v)
    return d_copy


def get_lr_scheduler(lr_schedule,optimizer):
    if 'step' in lr_schedule:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler  = None
    return scheduler



def get_coordinates(img_shape,patches=False,patch_shape=None,coord_range=[-1,1],devices=1):
    """
        Function to return normalized coordinates for any given size. 
        Used only for inference.

        Args:
            img_shape: (h,w) or (h,w,c) tuple

    """
    if len(img_shape) == 3:
        h,w,c = img_shape
    elif len(img_shape) == 4:
        n,c,h,w = img_shape
    else:
        h,w = img_shape


    coordinates = torch.ones((h,w)).nonzero(as_tuple=False).float()
    
    coordinates = coordinates / (max(h,w) - 1) 

    if coord_range == [-1,1]:
        coordinates -= 0.5
        coordinates *= 2

    if devices > 1:
        coordinates = coordinates.repeat(devices,1,1)

    return coordinates


def load_tensor(img_path):
    """
        load image as a tensor. 
        NCHW. Use PIL to load image.
    """
    img = Image.open(img_path).convert('RGB')
    img = transforms.ToTensor()(img).unsqueeze(0).float()
    return img




def to_coordinates_and_features(img,ch_norm=False):
    """Converts an image to a set of coordinates and features.

    Args:
        img (torch.Tensor): Shape (channels, height, width).
    """
    # Coordinates are indices of all non zero locations of a tensor of ones of
    # same shape as spatial dimensions of image
    coordinates = torch.ones(img.shape[1:]).nonzero(as_tuple=False).float() #CHW
    # Normalize coordinates to lie in [-.5, .5]
    coordinates = coordinates / (img.shape[1] - 1) - 0.5
    # Convert to range [-1, 1]
    coordinates *= 2
    # Convert image to a tensor of features of shape (num_points, channels)


    features = img.reshape(img.shape[0], -1).T
    return coordinates, features


def to_coord_patch_grid(img,patch_shape):
    """
        Converts an image to a set of patches (their centroids) and corersponding GRID features.
        We return features with shape : HxW

        Here we process the coordinates as meshgrid. So it has 2 channels. 

    """

    #coords = np.linspace(0, 1, img.shape[2], endpoint=False)
    x_coords = np.linspace(0, 1, img.shape[1], endpoint=False)
    y_coords = np.linspace(0, 1, img.shape[2], endpoint=False)

    x_grid,y_grid = np.meshgrid(x_coords,y_coords)
    xy_grid = np.stack([x_grid,y_grid], -1)
    xy_grid = torch.tensor(xy_grid).unsqueeze(0).permute(0, 3, 1, 2).float().contiguous()#@.to(device)

    patcher = patching.Patcher(patch_shape)
    temp = xy_grid.clone() * 0
    patched_grid,shape = patcher.patch(temp.squeeze(0))

    for k in range(len(patched_grid)):
        patched_grid[k][0][patch_shape[0]//2,patch_shape[1]//2] = 1
        patched_grid[k][1][patch_shape[0]//2,patch_shape[1]//2] = 1


    unpatched_grid =  patcher.unpatch(patched_grid,shape).squeeze()
    coordinates = unpatched_grid.nonzero(as_tuple=False).float()



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

    pad = nn.ConstantPad2d( (w_padding[0],w_padding[1],h_padding[0],h_padding[1]), value=0)

    img = pad(img)
    
    return img


def unpad(img,out_shape):
    """
        Undo padding. Since we do constant padding, we can just do a center crop.
    """
    unpadded_tensor = transforms.CenterCrop(out_shape)(img)
    return unpadded_tensor

def optimizer_to(optim, device):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)

        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def to_coordinates_and_features_patches(img,patch_shape,centroid=True):
    """
        Converts an RGB image to a set of patches (their centroids) and corresponding features.
    """
    #pad input image.
    img = pad_input(img.unsqueeze(0),block_size=patch_shape[0]).squeeze()


    patcher = patching.Patcher(patch_shape)
    patched, data_shape = patcher.patch(img) 


    if centroid:
        coordinates = torch.zeros(img.shape[1:])
        
        patch_coord,shape = patcher.patch(coordinates.unsqueeze(0))
        patch_coord = patch_coord.squeeze()
        
        for k in range(len(patch_coord)):
            patch_coord[k][patch_shape[0]//2,patch_shape[1]//2] = 1
        
        unpatched_coordinates =  patcher.unpatch(patch_coord.unsqueeze(1),shape).squeeze()
        coordinates = unpatched_coordinates.nonzero(as_tuple=False).float()
    
        #Standard transforms
        coordinates = coordinates / (img.shape[1] - 1) - 0.5
        coordinates *= 2

    # Convert image to a tensor of features of shape (num_points, channels)
    return coordinates, patched,data_shape


def to_coordinates_and_features_patches_DCT(img,patch_shape,block_size=None,pad=True,centroid=True):
    """
        Converts an image to a set of patches (their centroids) and corresponding features.

        Returns:
            coordinates (torch.Tensor): Shape (num_points, 2).
            features (torch.Tensor): Shape (num_points, channels).
            shape (tuple): Shape of image.
    """

    if block_size is None:
        raise ValueError("block_size must be specified")

    patcher = patching.Patcher(patch_shape)

    img = dct.batch_dct(img.unsqueeze(0),device=img.device,block_size=block_size,pad=pad).squeeze(0)
    #print('dct image shape: ',img.shape)
    

    patched, data_shape = patcher.patch(img) 


    """
        Get the coordinates for the entire image.
        #patch it 
        #fill the centroids of each patch with nonzero.
        unpatch again and get non zero indices -> they point to centroid.
    """
    
    if centroid:
        coordinates = torch.zeros(img.shape[1:])
        
        patch_coord,shape = patcher.patch(coordinates.unsqueeze(0))
        patch_coord = patch_coord.squeeze()
        
        for k in range(len(patch_coord)):
            patch_coord[k][patch_shape[0]//2,patch_shape[1]//2] = 1
        
        unpatched_coordinates =  patcher.unpatch(patch_coord.unsqueeze(1),shape).squeeze()
        coordinates = unpatched_coordinates.nonzero(as_tuple=False).float()
    
        #Standard transforms
        coordinates = coordinates / (img.shape[1] - 1) - 0.5
        coordinates *= 2

    else:
        #give unique ID to each patch.
        coordinates = torch.arange(patched.shape[0]).float()
        #positional encoding ?? NO.
        breakpoint()
        coordinates = coordinates / (img.shape[1] - 1) - 0.5
        # Convert to range [-1, 1]
        coordinates *= 2
        # Convert image to a tensor of features of shape (num_points, channels)

    return coordinates, patched,data_shape



def to_coordinates_and_features_DCT(img,ch_norm=False,block_size=8,pad=True):
    """Converts an image to a set of coordinates and features.

    Args:
        img (torch.Tensor): Shape (channels, height, width).
    """
    coordinates = torch.ones(img.shape[1:]).nonzero(as_tuple=False).float()
    coordinates = coordinates / (img.shape[1] - 1) - 0.5	# Convert to range [-1, 1]
    coordinates *= 2
    

    img = dct.batch_dct(img.unsqueeze(0),device=img.device,block_size=block_size,pad=pad).squeeze(0)


    features = img.reshape(img.shape[0], -1).T
    
    shape = img.shape

    return coordinates, features,shape

def get_model(cfg_network):

    model_name = cfg_network.model_name
    #import module from string
    module = importlib.import_module('models.'+model_name)
    model_class = getattr(module, model_name)
    
    print('network input: ',cfg_network.dim_in,' dim coordinates')
    model = model_class(cfg_network,dim_in=cfg_network.dim_in)	

    return model

def state_dict_size_in_bits(state_dict):
    """Calculate total number of bits to store `state_dict`."""
    return sum(sum(t.nelement() * DTYPE_BIT_SIZE[t.dtype] for t in tensors)
               for tensors in state_dict.values())

def model_size_in_bits(model):
    """Calculate total number of bits to store `model` parameters and buffers."""
    return sum(sum(t.nelement() * DTYPE_BIT_SIZE[t.dtype] for t in tensors)
               for tensors in (model.parameters(), model.buffers()))


def bpp(image, model):
    """Computes size in bits per pixel of model.

    Args:
        image (torch.Tensor): Image to be fitted by model.
        model (torch.nn.Module): Model used to fit image.
    """
    num_pixels = np.prod(image.shape) / 3  # Dividing by 3 because of RGB channels
    return model_size_in_bits(model=model) / num_pixels




def get_bpp(num_frames,og_size,img_size,sparsity=0,quant_level=32):
    """
        img_size: img_shape - (h,w)
    """

    og_bpp = (og_size *1e6 * 8) / (num_frames * img_size[0] * img_size[1])

    post_prune_bpp = (og_bpp * (1-sparsity)) + (og_bpp/32)

    quant_bpp = (og_bpp * (1-sparsity) *(quant_level/32) )   + (og_bpp/32)

    return og_bpp,post_prune_bpp,quant_bpp

def loss2psnr(loss):
    #return 10*torch.log10(4 /loss)
    return 20. * torch.log10(torch.tensor(1.0)) - 10. * torch.log10(loss)#.cpu()

def psnr(img1, img2):
    """Calculates PSNR between two images.

    Args:
        img1 (torch.Tensor):
        img2 (torch.Tensor):
    """
    return 20. * np.log10(1.) - 10. * (img1 - img2).detach().pow(2).mean().log10().to('cpu').item()


def clamp_image(img):
    """Clamp image values to like in [0, 1] and convert to unsigned int.

    Args:
        img (torch.Tensor):
    """
    # Values may lie outside [0, 1], so clamp input
    img_ = torch.clamp(img, 0., 1.)
    # Pixel values lie in {0, ..., 255}, so round float tensor
    return torch.round(img_ * 255) / 255.


def get_clamped_psnr(img, img_recon):
    """Get PSNR between true image and reconstructed image. As reconstructed
    image comes from output of neural net, ensure that values like in [0, 1] and
    are unsigned ints.

    Args:
        img (torch.Tensor): Ground truth image.
        img_recon (torch.Tensor): Image reconstructed by model.
    """
    return psnr(img, clamp_image(img_recon))



def get_key_frames(content="playground/og_bunny_nerv.avi"):

    container = av.open(content)

    stream = container.streams.video[0]
    stream.codec_context.skip_frame = "NONKEY"

    key_frames = []

    for idx,frame in enumerate(container.decode(stream)):

        print('key frame:',frame.pts)
        # We use `frame.pts` as `frame.index` won't make must sense with the `skip_frame`.
        key_frames.append(frame.pts)
    
    return key_frames


def get_cosine_lr(iteration, warmup_steps, max_lr,max_iters):
    if iteration < warmup_steps:
        # warmup phase
        return max_lr * iteration / warmup_steps
    else:
        # cosine decay phase
        progress = (iteration - warmup_steps) / (max_iters - warmup_steps)
        return max_lr * 0.5 * (1 + math.cos(math.pi * progress))

def adjust_lr(optimizer, current_iter,total_iters, cfg):
    
    if cfg.trainer.lr_schedule_type == 'cosine':
        lr = get_cosine_lr(current_iter,int(cfg.trainer.lr_warmup*total_iters),max_lr=cfg.trainer.lr,max_iters=total_iters)

    else:
        lr = cfg.trainer.lr

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr

    return optimizer,lr    

def find_ckpt(save_dir):
    """
        Function to recursively find ckpt files in the directroy 
        return the path of the latest one. 
    """
    ckpt_files = glob.glob(save_dir+'/**/*.ckpt',recursive=True)
    #ckpt_files = sorted(ckpt_files,key=lambda x: int(x.split('_')[-1].split('.')[0]))
    #sort accoriding to creation time. 
    ckpt_files = sorted(ckpt_files,key=os.path.getctime)
    if ckpt_files == []:
        return None
    return ckpt_files[-1]





def check_sparsity(state):
    n_zeros = 0
    n_params = 0
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            n_zeros += torch.sum(torch.abs(value)==0.0).item()
            n_params += value.numel()
    
    print('sparsity: ',n_zeros/n_params)

    return n_zeros/n_params


def save_tensor_img(tensor, filename='temp.png'):
    """
        conver to image and save.
    """	
    tensor = tensor_to_cv(tensor)
    tensor = cv2.cvtColor(tensor,cv2.COLOR_BGR2RGB)
    cv2.imwrite(filename, tensor)


def tensor_to_cv(tensor):
    """
    convert tensor to numpy.
    """
    if len(tensor.shape)==3:
        tensor = tensor.unsqueeze(0)

    #tensor = tensor.detach().cpu().permute(1, 2, 0).numpy() * 255
    tensor = tensor.detach().cpu().permute(0, 2, 3, 1).numpy() * 255
    tensor = tensor.astype(np.uint8)
    
    return tensor.squeeze()



def compress_weights(state,quant_bit,quant_axis,return_reconstructed=False,quant_filter=0,quant_diff=False,device=None):
    """
        Compress weights using entropy coding.
        return_recconstructed: return reconstructed weights, to save time. In some cases we only infer and dump.
        quant_diff: quantize the difference between the original and reconstructed weights.
        Quant_filter: make values below this threshold zero.
    """

    if device is None:
        device = torch.device("cuda:{}".format(0))

    if type(state) is not StateDictOperator:
        state = StateDictOperator(state)
    if quant_filter!=0 and quant_diff:
        state = state.filter(value=quant_filter).state_dict
    else:
        state = state.state_dict


    quant_weight_list = []
    scales = []
    t_min_vals = []
    shapes = []

    reconstructed_weights = {}

    for k,v in state.items():
        large_tf = (v.dim() in {2,4} and 'bias' not in k)
        quant_v,scale,t_min = quantize_per_tensor(v, quant_bit, quant_axis if large_tf else -1)

        valid_quant_v = quant_v[v!=0] # only include non-zero weights
        quant_weight_list.append(valid_quant_v.flatten())
        scales.append(scale.cpu().tolist())
        t_min_vals.append(t_min.cpu().tolist())
        shapes.append(tuple(v.shape))

        if return_reconstructed:
            t_min_vals = [torch.tensor(x).clone() for x in t_min_vals]
            scales = [torch.tensor(x).clone() for x in scales]

            reconstructed_tensor = t_min_vals[-1].to(device) + scales[-1].to(device) * quant_v
            reconstructed_weights[k] =  reconstructed_tensor

    cat_param = torch.cat(quant_weight_list)
    input_code_list = cat_param.tolist()
    input_code_list = [int(x) for x in input_code_list]

    codec = HuffmanCodec.from_data(input_code_list)
    encoded = codec.encode(input_code_list)

    info = {}
    info['encoding'] = encoded
    info['codec'] = codec
    info['scales'] = scales
    info['t_min_vals'] = t_min_vals
    info['quant_bit'] = quant_bit
    info['quant_axis'] = quant_axis
    info['shapes'] = shapes

    if return_reconstructed:
        return info,reconstructed_weights

    return info

def decompress_weights(compressed_dict,state_keys,device=None):
    """
        Pass the dict returned by compress_weights function.
        state_keys are the keys found in model state_dict.
    """
    
    quant_bit = compressed_dict['quant_bit']
    scales = compressed_dict['scales']
    t_min_vals = compressed_dict['t_min_vals']
    codec = compressed_dict['codec']
    encoding = compressed_dict['encoding']
    shapes = compressed_dict['shapes']
    decoded = codec.decode(encoding)

    t_min_vals = [torch.tensor(x) for x in t_min_vals]
    scales = [torch.tensor(x) for x in scales]


    reconstructed_state = {}
    start = 0

    for k,key in enumerate(state_keys):
        #temp = torch.tensor(decoded[start:np.prod(shapes[k])]).reshape(shapes[k])
        end = np.prod(shapes[k]) + start
        temp = torch.tensor(decoded[start:end]).reshape(shapes[k])
        
        temp = t_min_vals[k] + scales[k] * temp

        if device is not None:
            temp = temp.to(device)
        reconstructed_state[key] = temp
        start = end

    return reconstructed_state	

def save_get_size(obj, filename='temp.pt'):
    """Save object and return its size"""
    torch.save(obj, filename, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    return os.path.getsize(filename)

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def get_features(data,patch_shape,volume=False):

    """
        Extract patches from data.
        specify volume=True for 3D patches.
    """

    N,C,H,W = data.shape
    data = pad_input(data,patch_shape[0])
    
    if volume:
        vol_patch_shape = (3,patch_shape[0],patch_shape[1])
        patcher = patching.Patcher(vol_patch_shape)
    else:
        patcher = patching.Patcher(patch_shape)
        

    patched, data_shape = patcher.patch(data)
    return patched, data_shape


def slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        # L'Hopital's rule/LERP
        return (1.0-val) * low + val * high
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

# uniform interpolation between two points in latent space
def interpolate_points(p1, p2,num_interpolate_points=50):
    # interpolate ratios between the points
    ratios = np.linspace(0, 1, num=num_interpolate_points)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        v = slerp(ratio, p1, p2)
        vectors.append(v)
    return vectors


def prompt_to_template(prompt):
    """
        Convert prompt to template.
    """
    prompt_templates = ['a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.', 'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.', 'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.', 'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.', 'a painting of a {}.', 'a pixelated photo of the {}.', 'a sculpture of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.', 'a {} in a video game.', 'a photo of one {}.', 'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'the origami {}.', 'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.', 'a origami {}.', 'a low resolution photo of a {}.', 'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.', 'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.', 'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.', 'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a plushie {}.', 'a photo of the nice {}.', 'a photo of the small {}.', 'a photo of the weird {}.', 'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.', 'itap of a {}.', 'graffiti of the {}.', 'a toy {}.', 'itap of my {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'a tattoo of the {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.',]
    texts = [template.format(prompt) for template in prompt_templates] 
    return texts

def load_clip_model():
    model, preprocess = clip.load("ViT-B/32", device='cuda')
    return model, preprocess

def clip_text_features(texts,device='cuda',model=None,preprocess=None):
    
    if model is None and preprocess is None:
        model, preprocess = clip.load("ViT-B/32", device=device)
    
    tokenized_texts = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokenized_texts).float()
        
        if len(texts) > 1:
            """ From: https://github.com/chongzhou96/MaskCLIP/blob/master/tools/maskclip_utils/prompt_engineering.py """
            # This might move things around a bit.
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.mean(dim=0)
            text_features /= text_features.norm()

    return text_features,tokenized_texts

def get_mlp_matrix(state,model_config_path):
    
    model_config = OmegaConf.load(model_config_path)
    num_layers = model_config.network.num_layers

    pos_matrix = []
    mlp_matrix = []

    for key in state:
        value = state[key]
        if ('positional_encoding' in key) and ('0' not in key):
            pos_matrix.append(value)
            
        
        #remove first and last layer.
        elif ('linear' in key) and ('0' not in key) and (str(num_layers-1)) not in key:
                if 'bias' in key:
                    mlp_matrix.append(value.unsqueeze(0))
                else:
                    mlp_matrix.append(value)

    
    max_size = max(t.shape[1] for t in pos_matrix)
    padded_pos_tensors = [torch.nn.functional.pad(t, (0, 0, 0, max_size - t.shape[1])) for t in pos_matrix]

    
    pos_matrix = torch.cat(padded_pos_tensors, dim=0).T    
    mlp_matrix = torch.cat(mlp_matrix, dim=0).T  #Input dim as channels! 
    
    return pos_matrix,mlp_matrix



if __name__ == '__main__':

    #all tests running. 

    def load_img():
        import cv2 
        img = cv2.imread('../../frequency_stuff/UVG/bosphore_1080/f00001.png')
        img = img / 255.0
        img = torch.tensor(img).permute(2,0,1).float()
        return img

    patch_shape = (64,64)
    data = torch.rand((3, 95, 100))
    
    #og_img = torch.rand((3, 104, 104))
    og_img = load_img()


    def test_ch_norm():
        mean,std = ch_mean_std(og_img)
        img = (og_img - mean)/std
        inv_img = inverse_channel_norm(img,mean,std)
        print(torch.allclose(og_img,inv_img,3))


    def test_ch_norm_dct():
        #test channel norm + DCT
        og_dct = dct.batch_dct(og_img.unsqueeze(0),device=og_img.device,pad=True).squeeze()
        dct_mean,dct_std = ch_mean_std(og_dct)
        coordinates, features,shape = to_coordinates_and_features_DCT(og_img,ch_norm=True,pad=True)
        rec_dct = features.reshape(og_img.shape[1], og_img.shape[2], 3).permute(2, 0, 1)
        rec_dct = inverse_channel_norm(rec_dct,dct_mean,dct_std)
        rec_img = dct.batch_idct(rec_dct.unsqueeze(0),device=og_img.device).squeeze()
        print(torch.allclose(og_img,rec_img,3))

    def test_dct():
        #test dct 
        features = dct.batch_dct(og_img.unsqueeze(0),device=og_img.device,pad=True).squeeze()
        features = features.reshape(og_img.shape[0], -1).T	
        
        features = features.reshape(og_img.shape[1], og_img.shape[2], 3).permute(2, 0, 1)
        features = dct.batch_idct(features.unsqueeze(0),device=og_img.device).squeeze()
        print(torch.allclose(og_img,features,3))


    def test_dct_patches():
        block_size = 64

        coordinates, patched,data_shape = to_coordinates_and_features_patches_DCT(og_img,patch_shape,block_size=block_size,pad=True)
        
        #unpatch.
        patcher = patching.Patcher(patch_shape)
        unpatched_dct = patcher.unpatch(patched,data_shape)
        #cannot compare with dct. Not same shape. Unpad can be done only after inverse dct.
        out_size = (og_img.shape[1],og_img.shape[2])
        unpatched_img = dct.batch_idct(unpatched_dct.unsqueeze(0),device=unpatched_dct.device,block_size=block_size,pad=True,out_size=out_size).squeeze()
        print('DCT patched features shape:',patched.shape,' DCT coordinates shape:',coordinates.shape)
        print(torch.allclose(og_img,unpatched_img,3))
    
    def test_rgb_patches():
        print('image shape:',og_img.shape)

        img = pad_input(og_img.unsqueeze(0),patch_shape[0]).squeeze(0)
        #img = og_img
        patched_og_shape = img.shape

        coordinates, patched,data_shape = to_coordinates_and_features_patches(img,patch_shape)

        # coordinates, patched,data_shape = to_coordinates_and_features_patches(og_img,patch_shape)
        print('patched features shape:',patched.shape,' coordinates shape:',coordinates.shape)
        patcher = patching.Patcher(patch_shape)

        # unpatched_img = patcher.unpatch(patched,og_img.shape[1:])
        unpatched_img = patcher.unpatch(patched,patched_og_shape[1:])

        out_size = (og_img.shape[1],og_img.shape[2])

        unpatched_img = unpad(unpatched_img,out_shape=out_size)
        

        print(torch.allclose(og_img,unpatched_img,3))
        print('unpatched img shape:',unpatched_img.shape)
        


    def test_fourier_features():
        coordinates, patched,data_shape = to_coordinates_and_features_patches(og_img,patch_shape)
        x = GaussianFourierFeatureTransform(num_input_channels=2, mapping_size=128, scale=10)(coordinates)
        print(x.shape)


    def test_residual():
        image_list = sorted(glob.glob('bunny_data/*.png'))
        res,patch_res = get_residual(image_list,patch_size=64,plot=True,save_path='bunny_meta_data/')
        #res,patch_res = get_residual(image_list,patch_size=32,plot=True,save_path='temp_32/')


    # test_ch_norm()
    # test_ch_norm_dct()
    # test_dct()
    #test_dct_patches()
    test_rgb_patches()
    #test_fourier_features()
    #test_residual()