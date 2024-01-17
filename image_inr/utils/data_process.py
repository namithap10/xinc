import imp
from sys import breakpointhook
from xml.sax.handler import feature_namespace_prefixes
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import numpy as np
from einops import rearrange

from . import patching,dct



class DataProcessor(object):
	def __init__(self, cfg_dataset,device=None):
		self.cfg_dataset = cfg_dataset
		self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# TODO Add support for fp16 and autocast. 
		self.dtype = torch.float32

	def process_inputs(self,data,return_coord=True):

		"""
			Function not called. Coordinates and features obtained separately.
		"""

		if self.cfg_dataset.patches:
			self.patch_shape = (self.cfg_dataset.block_size,self.cfg_dataset.block_size)
			data = self.pad_input(data,self.cfg_dataset.block_size).squeeze()
		else:
			self.patch_shape = None 



		
		features,features_shape = self.get_features(data,patch_shape=self.patch_shape)


		if self.cfg_dataset.data_range == [-1,1]:
			#convert from 0,1 to -1,1
			features = (features * 2) - 1

		if return_coord:
			coordinates = self.get_coordinates(data.shape,patch_shape=self.patch_shape,split=self.cfg_dataset.coord_split)	
			return coordinates, features,features_shape

		else:
			return features,features_shape


	def test(self,data):
		"""
			Testing function for the data processor. 
		"""
		coordinates, features,features_shape = self.process_inputs(data)
		out = self.process_outputs(features,features_shape,data.shape)

		print(torch.allclose(data,out,atol=1e-3))
		return out

	def process_outputs(self,out,input_img_shape,features_shape=None,patch_shape=None,group_size=None):
		
		"""
			For all DCT transorms, unpatch -> inv_dct -> unpad -> crop
			For all RGB transforms, unpatch -> unpad -> crop

			Input_img_shape is NCHW. 
			
		"""

		
		if len(input_img_shape) == 3:
			C,H,W = input_img_shape
			N = 1
		elif len(input_img_shape) == 4:
			N,C,H,W = input_img_shape


		if patch_shape is not None:
			
			#patch_shape = (self.cfg_dataset.block_size,self.cfg_dataset.block_size)			
			
			patcher = patching.Patcher(patch_shape)
			out_reshape = patcher.unpatch(out,features_shape)

			if self.cfg.dataset.dct:
				out_reshape = dct.batch_idct(out_reshape.unsqueeze(0),device=out_reshape.device,block_size=self.cfg_dataset.block_size)#.squeeze()
		
		elif self.cfg_dataset.coord_grid_mapping:
			return out #already nchw.

		else:
			out_reshape = out.reshape(N,H,W,C).permute(0,3,1,2)

		#unpad wont hurt even if there is no padding
		out_reshape = self.unpad(out_reshape,(H,W))

		#out is between 0 and 1. Else causes artifacts.
		out_reshape = out_reshape.clamp(0,1)
		

		if out_reshape.dim() == 3:
			out_reshape = out_reshape.unsqueeze(0)

		return out_reshape


	def unpad(self,img,out_shape):
		"""
			Undo padding. Since we do constant padding, we can just do a center crop.
			Expects HW.
		"""
		unpadded_tensor = transforms.CenterCrop(out_shape)(img)
		return unpadded_tensor


	def pad_input3D(self,img,volume_size):
		"""
			pad input image tensor so that it is divisible by block size
			Image - NCHW.
			Will return ** original image ** if it is already divisible by block size.
			volume_size - HWT
		"""

		img = self.pad_input(img, volume_size[0])
		t_pad_size = (volume_size[2] - img.shape[0] % volume_size[2]) % volume_size[2]

		if t_pad_size>0:
			pad_img = img[-1:,...].repeat(t_pad_size,1,1,1)
			img = torch.cat((img,pad_img),dim=0)
			assert img.size(0) % volume_size[2] == 0
		
		return img


	def pad_input(self,img,block_size):
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

		if self.cfg_dataset.rep_pad:
			pad = nn.ReplicationPad2d((w_padding[0],w_padding[1],h_padding[0],h_padding[1]))
		else:
			pad = nn.ConstantPad2d( (w_padding[0],w_padding[1],h_padding[0],h_padding[1]), value=0)

		img = pad(img)
		
		return img


	def normalize_coordinates(self,coordinates,data_shape,normalize_range=[-1,1]):

		"""
			Normalize coordinates to [-1,1] range.
			Either 2 axes separately or with respect to max dimension.
		"""

		H,W = data_shape[-2:]
		
		if self.cfg_dataset.coord_separate_norm:
			coordinates[:,0] = coordinates[:,0] / (H-1)
			coordinates[:,1] = coordinates[:,1] / (W-1)
		else:
			max_dim = max(H,W)
			coordinates = coordinates / (max_dim - 1) 

		if normalize_range == [-1,1]:
			coordinates  -= 0.5
			coordinates *= 2
		
		return coordinates

	def get_coordinates_3D(self,data_shape,patch_shape=None,split=False,normalize_range=[-1,1],dim=3):

		T,C,H,W = data_shape

		if patch_shape is not None:
			#pad before DCT when patching.
			if type(patch_shape) == int:
				patch_shape = (patch_shape,patch_shape)
		
			nh,nw = H//patch_shape[0],W//patch_shape[1]
		else:
			nh,nw = H,W

		t,h,w = torch.meshgrid(torch.arange(T),torch.arange(nh),torch.arange(nw))
		coordinates = torch.cat((h.reshape(-1,1), w.reshape(-1,1), t.reshape(-1,1)),dim=1).float() # hwt

		coordinates /= torch.Tensor([nh,nw,T]).reshape(1,-1)
		
		if normalize_range == [-1,1]:
			coordinates  -= 0.5
			coordinates *= 2
		
		return coordinates

	
	def get_coordinates(self,data_shape,patch_shape=None,split=False,normalize_range=[-1,1],dim=2):

		if dim == 2:
			H,W = data_shape[-2:]

		if (patch_shape is not None) and (split == True): raise("Split and patches cannot be together")


		if patch_shape is not None:

			if type(patch_shape) == int:
				patch_shape = (patch_shape,patch_shape)

			patcher = patching.Patcher(patch_shape)
			coordinates = torch.zeros((H,W))
			
			patch_coord,shape = patcher.patch(coordinates.unsqueeze(0))
			patch_coord = patch_coord.squeeze()
			
			for k in range(len(patch_coord)):
				patch_coord[k][patch_shape[0]//2,patch_shape[1]//2] = 1
			
			unpatched_coordinates =  patcher.unpatch(patch_coord.unsqueeze(1),shape).squeeze()
			coordinates = unpatched_coordinates.nonzero(as_tuple=False).float()
			 
		elif self.cfg_dataset.coord_grid_mapping:
			coord_x = torch.linspace(0,H-1,H)
			coord_y = torch.linspace(0,W-1,W)
			coordinates = torch.stack(torch.meshgrid(coord_x,coord_y),dim=-1).permute(2,0,1).unsqueeze(0)

		else:
			coord_range = (H,W)
			coordinates = torch.ones(coord_range).nonzero(as_tuple=False).float()

		if split:
			
			coord_x = torch.unique(coordinates[:,0])
			coord_y = torch.unique(coordinates[:,1])

			#normalize to [-1,1]
			coord_x = coord_x / (H-1) - 0.5
			coord_y = coord_y / (W-1) - 0.5
			coord_x *= 2
			coord_y *= 2			
			coordinates = [coord_x,coord_y]
			
		else:
			coordinates = self.normalize_coordinates(coordinates,data_shape,normalize_range=normalize_range) #normalize coordinates.		

		return coordinates

	def get_features(self,data,patch_shape=None):

		data_shape = data.shape

		if data.ndim == 3:
			C,H,W = data.shape
			N = 1
		else:
			N,C,H,W = data.shape


		if patch_shape is not None:
			#pad before DCT when patching.
			if type(patch_shape) == int:
				patch_shape = (patch_shape,patch_shape)
			data = self.pad_input(data,block_size = patch_shape[-1]).squeeze()

		if self.cfg_dataset.dct:
			data = dct.batch_dct(data.unsqueeze(0),device=data.device).squeeze(0)


		#final processing 
		if patch_shape is not None:
			patcher = patching.Patcher(patch_shape)
			#data_shape might change when patching.
			features, data_shape = patcher.patch(data) 
			features = features.reshape(features.shape[0], -1)

		elif not self.cfg_dataset.coord_grid_mapping:
			# Convert image to a tensor of features of shape (num_points, channels)
			features = data.reshape(data.shape[0], -1).T if data.ndim == 3 else data.reshape(N,data.shape[1], -1).permute(0,2,1)
		else:
			features = data.unsqueeze(0)

		return features,data_shape
	

	def get_features_3D(self,data,patch_shape=None,batch_size=1):
		"""
			Pass the full video. as NCWH
		"""
		N,C,H,W = data.shape
		data_shape = data.shape

		if patch_shape is not None:
			#pad before DCT when patching.
			if type(patch_shape) == int:
				patch_shape = (patch_shape,patch_shape)

			data = self.pad_input3D(data,(patch_shape[0],patch_shape[1],batch_size))
			data_shape = data.shape #might change when patching.
			self.padded_H, self.padded_W, self.padded_T = data.shape[2], data.shape[3], data.shape[0]
		
		if self.cfg_dataset.dct:
			data = dct.batch_dct(data,device=data.device).squeeze(0)

		if patch_shape is not None:
			features = rearrange(data, '(nt pt) c (nh ph) (nw pw) -> (nt nh nw) c ph pw pt', pt=batch_size, ph = patch_shape[0], pw = patch_shape[1])
		else:
			features = rearrange(data, 'n c h w -> (n h w) c')
		
			
		return features,data_shape


	def to_coordinates_and_features_full(self,data,patch_shape=None,batch_size=1):	
		
		"""
			data is NCWH. 
			Here we pass the entire video. 
		"""

		N,C,H,W = data.shape

		if patch_shape is not None:
			#pad before DCT when patching.
			if type(patch_shape) == int:
				patch_shape = (patch_shape,patch_shape)

			data = self.pad_input3D(data,(patch_shape[0],patch_shape[1],batch_size))
			self.padded_H, self.padded_W, self.padded_T = data.shape[2], data.shape[3], data.shape[0]
		
			vol_patch_shape = (3,patch_shape[0],patch_shape[1])
		
			features = rearrange(data, '(nt pt) c (nh ph) (nw pw) -> (nt nh nw) c ph pw pt', pt=batch_size, ph = patch_shape[0], pw = patch_shape[1])
			nt, nh, nw = data.size(0)//batch_size, data.size(2)//patch_shape[0], data.size(3)//patch_shape[1]
		else:
			breakpoint()
			features = rearrange(data, 'n c h w -> (n h w) c')
			nt, nh, nw = data.size(0)//batch_size, data.size(2), data.size(3)
		#nt, nh, nw = data.size(0)//batch_size, data.size(2)//patch_shape[0], data.size(3)//patch_shape[1]
		t,h,w = torch.meshgrid(torch.arange(nt),torch.arange(nh),torch.arange(nw))
		coordinates = torch.cat((h.reshape(-1,1), w.reshape(-1,1), t.reshape(-1,1)),dim=1).float() # hwt
		coordinates /= torch.Tensor([nh,nw,nt]).reshape(1,-1)

		return coordinates,features,features.shape

	def to_coordinates_and_features(self,data,patch_shape=None,split=False):
		"""Converts an image to a set of coordinates and features.

		Args:
			img (torch.Tensor): Shape (channels, height, width).
			optional: parch_shape (tuple): Shape of patch.
		"""
		
		data = data.squeeze(0) #HWC
		C,H,W = data.shape

		if (patch_shape is not None) and (split == True): raise("Split and patches cannot be together")

		if self.cfg_dataset.dct:
			data = dct.batch_dct(data.unsqueeze(0),device=data.device).squeeze(0)

		if patch_shape is not None:
			patcher = patching.Patcher(patch_shape)
			features, data_shape = patcher.patch(data) 
			coordinates = torch.zeros(data.shape[1:])
			
			patch_coord,shape = patcher.patch(coordinates.unsqueeze(0))
			patch_coord = patch_coord.squeeze()
			
			for k in range(len(patch_coord)):
				patch_coord[k][patch_shape[0]//2,patch_shape[1]//2] = 1
			
			unpatched_coordinates =  patcher.unpatch(patch_coord.unsqueeze(1),shape).squeeze()
			coordinates = unpatched_coordinates.nonzero(as_tuple=False).float()
			 

		elif self.cfg_dataset.coord_grid_mapping:
			coord_x = torch.linspace(0,H-1,H)
			coord_y = torch.linspace(0,W-1,W)
			coordinates = torch.stack(torch.meshgrid(coord_x,coord_y),dim=-1).permute(2,0,1).unsqueeze(0)
			data_shape = data.shape
			#normalization later.

		else:
			coordinates = torch.ones(data.shape[1:]).nonzero(as_tuple=False).float()
			data_shape = data.shape
		
		if split:
			
			coord_x = torch.unique(coordinates[:,0])
			coord_y = torch.unique(coordinates[:,1])

			#normalize to [-1,1]
			coord_x = coord_x / (H-1) - 0.5
			coord_y = coord_y / (W-1) - 0.5
			coord_x *= 2
			coord_y *= 2			
			coordinates = [coord_x,coord_y]
			
		else:
			coordinates = self.normalize_coordinates(coordinates,data) #normalize coordinates.

		if not self.cfg_dataset.coord_grid_mapping:
			# Convert image to a tensor of features of shape (num_points, channels)
			features = data.reshape(data.shape[0], -1).T
		else:
			features = data.unsqueeze(0)

		return coordinates, features,data_shape
	


if __name__ == '__main__':


	def load_img():
		import cv2 
		img = cv2.imread('../../frequency_stuff/vid_inr/frames/bosphore_1080/f00001.png')
		img = img / 255.0
		img = torch.tensor(img).permute(2,0,1).float().to('cuda')
		img = img.unsqueeze(0)
		return img

	from easydict import EasyDict as edict

