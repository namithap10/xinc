import torch
import torch.nn as nn

import math 
import numpy as np

def normalize_coordinates(coordinates,img):
	"""
		Normalize coordinates to [-1,1] range.
		Either 2 axes separately or with respect to max dimension.
	"""

	if img.dim() == 3:
		C,H,W = img.shape
	else:
		N,C,H,W = img.shape
	coordinates = coordinates / (max(H,W) - 1) - 0.5		
	coordinates *= 2
	return coordinates


class PosEncodingFourier(nn.Module):
	def __init__(self,dim_in,dim_hidden,scale,mapping_size):
		super().__init__()
		self.dim_in = dim_in
		self.dim_hidden = dim_hidden
		self.scale = scale
		self.mapping_size = mapping_size

		self.register_buffer('B',torch.randn((self.dim_hidden,2)) * self.scale)
		self.output_dim = self.dim_hidden*2

	def forward(self,x):
		x_proj = (2. * np.pi * x) @ self.B.t()
		return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PosEncodingNeRF(nn.Module):
	'''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
	def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True,num_freq=None,include_coord=True,freq_last=False):
		super().__init__()

		self.in_features = in_features
		self.include_coord = include_coord
		self.freq_last = freq_last

		if self.in_features == 3:
			self.num_frequencies = 10
		elif self.in_features == 2:
			assert sidelength is not None
			if isinstance(sidelength, int):
				sidelength = (sidelength, sidelength)
			self.num_frequencies = 4
			if use_nyquist and num_freq is None:
				self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
			elif num_freq is not None:
				self.num_frequencies = num_freq
			print('Num Frequencies: ',self.num_frequencies)
		elif self.in_features == 1:
			assert fn_samples is not None
			self.num_frequencies = 4
			if use_nyquist:
				self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)

		self.output_dim = in_features + 2 * in_features * self.num_frequencies

		#Fixed. Not trainable. 
		self.freq_bands = nn.parameter.Parameter(2**torch.arange(self.num_frequencies) * np.pi, requires_grad=False)


	def get_num_frequencies_nyquist(self, samples):
		nyquist_rate = 1 / (2 * (2 * 1 / samples))
		return int(math.floor(math.log(nyquist_rate, 2)))


	def forward(self, coords, single_channel=False):
		
		if single_channel:
			in_features = coords.shape[-1]
		else:
			in_features = self.in_features

		coords_pos_enc = coords.unsqueeze(-2) * self.freq_bands.reshape([1]*(len(coords.shape)-1) + [-1, 1]) #2*pi*coord
		sin = torch.sin(coords_pos_enc)
		cos = torch.cos(coords_pos_enc)

		coords_pos_enc = torch.cat([sin, cos], -1).reshape(list(coords_pos_enc.shape)[:-2] + [-1])
		
		if self.include_coord:
			coords_pos_enc = torch.cat([coords, coords_pos_enc], -1)

		if self.freq_last:
			sh = coords_pos_enc.shape[:-1]
			coords_pos_enc = coords_pos_enc.reshape(*sh, -1, in_features).transpose(-1,-2).reshape(*sh, -1)

		
		return coords_pos_enc



def to_coordinates_and_features(data):
	data = data.squeeze(0) #CHW
	coordinates = torch.ones(data.shape[1:]).nonzero(as_tuple=False).float()
	coordinates = normalize_coordinates(coordinates, data)
	features = data.reshape(data.shape[0], -1).T
	return coordinates, features


def get_activation(activation):
	
	if (activation == 'none') or (activation == 'linear') or (activation is None):
		return nn.Identity()

	elif activation.lower() == 'relu':
		return nn.ReLU()
	elif activation.lower() == 'leakyrelu':
		return nn.LeakyReLU()
	elif activation.lower() == 'tanh':
		return nn.Tanh()
	elif activation.lower() == 'sigmoid':
		return nn.Sigmoid()
	else:
		raise ValueError('Unknown activation function {}'.format(activation))