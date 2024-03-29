import torch 
import torch.nn as nn
import numpy as np
import math
from pydoc import locate

from . import hash_grid
#from . import model_utils
from .layers import positional_encoders


class BaseModel(nn.Module):
	def __init__(self,cfg):
		super().__init__()
		self.cfg = cfg   

		self.dim_in = self.cfg.data.dim_in
		self.dim_out = self.cfg.data.dim_out
		self.dim_hidden = self.cfg.network.layer_size
		self.num_layers = self.cfg.network.num_layers
		self.activation = self.cfg.network.activation
		
		pos_encoding_type = self.cfg.network.pos_encoding.type

		if pos_encoding_type is not None:	
			pos_encoding_opt = self.cfg.network.pos_encoding		
			if pos_encoding_type == 'nerf':
				print('Using NeRF pos encoding')
				sidelength = self.cfg.network.sidelength
				num_freq = pos_encoding_opt.num_frequencies			
				self.positional_encoding = positional_encoders.PosEncodingNeRF(in_features=self.dim_in,sidelength=sidelength,\
								fn_samples=None,use_nyquist= True,num_freq=num_freq)
		
			elif pos_encoding_type == 'hash_grid':
				print('Using hash grid encoding')
				options = self.cfg.network.pos_encoding.hash_grid_encoding
				self.positional_encoding = hash_grid.MultiResHashGrid(dim = self.dim_in,binarize=options.binarize,\
								n_levels=options.n_levels,n_features_per_level=options.n_features_per_level,\
								log2_hashmap_size=options.log2_hashmap_size,base_resolution=options.base_resolution,\
								finest_resolution=options.finest_resolution)
			
			elif pos_encoding_type == 'fourier':
				self.positional_encoding = positional_encoders.PosEncodingFourier(dim_in = self.dim_in,dim_hidden=self.dim_hidden,\
								  scale=pos_encoding_opt.fourier_noise_scale,mapping_size=pos_encoding_opt.fourier_mapping_size)

			self.in_features = self.positional_encoding.output_dim			

		else:
			self.positional_encoding = nn.Identity()
			self.in_features = self.dim_in

	def forward(self,x,**kwargs):
		raise NotImplementedError





