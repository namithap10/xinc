import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch._C import dtype
from typing import Dict

from . import patching

import torchvision.transforms as transforms

import json,os,pickle
import cv2
from PIL import Image
import glob 
import math
import sys
import importlib
from omegaconf import DictConfig
from pytorch_msssim import ssim
import compress_pickle

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
    """
    #patch size is same as block size
    patch_shape = (args.block_size,args.block_size)

    #these features are nchw hw =  patch size. 

    if args.rgb_patches: 
        coordinates,features,features_shape = to_coordinates_and_features_patches(img,patch_shape=patch_shape) 
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
            frame[i,:,random_noise_y, random_noise_x] += torch.from_numpy(noise_val).float().T.to(frame.device)

        elif noise_type =='random':
            noise_val = (np.random.rand(noise_amount, 3) * 2 - 1) * noise_amp
            frame[i,:,random_noise_y, random_noise_x] += torch.from_numpy(noise_val).float().T.to(frame.device)

        frame[i].clamp_(0,1)
    
    return frame

def process_outputs(out,features_shape,input_img_shape,patch_shape=None,**kwargs): 
  
        
    """
        For all RGB transforms, unpatch -> unpad -> crop

        Input_img_shape is NCHW. 
        
    """

    N,C,H,W = input_img_shape

    
    if kwargs['type'] == 'rgb_patches':
        patcher = patching.Patcher(patch_shape)
        out_reshape = patcher.unpatch(out,features_shape)
    
    
    elif kwargs['type'] == 'rgb_patch_volume':
        vol_patch_shape = (3,patch_shape[0],patch_shape[1])
        patcher = patching.Patcher(vol_patch_shape)
        out_reshape = patcher.unpatch(out,features_shape)

    else:
        N,C,H,W = features_shape #N=1
        out_reshape = out.reshape(H,W,C).permute(2,0,1)

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