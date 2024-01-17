import numpy as np
import math 

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision



def normalize_coordinates(coordinates,img):
	"""
		Normalize coordinates to [-1,1] range.
		Either 2 axes separately or with respect to max dimension.
	"""

	if img.dim() == 3:
		C,H,W = img.shape
	else:
		N,C,H,W = img.shape
	# if self.args.coord_separate_norm:
	# 	coordinates[:,0] = coordinates[:,0] / (H-1) - 0.5
	# 	coordinates[:,1] = coordinates[:,1] / (W-1) - 0.5
	# else:
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

		self.register_buffer('B',torch.randn((self.dim_hidden,self.dim_in)) * self.scale)
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

		#removes for loop over sine and cosine.
		#bad, but optimal code. lifted from https://github.com/nexuslrf/CoordX/blob/main/modules.py
		coords_pos_enc = coords.unsqueeze(-2) * self.freq_bands.reshape([1]*(len(coords.shape)-1) + [-1, 1]) #2*pi*coord
		sin = torch.sin(coords_pos_enc)
		cos = torch.cos(coords_pos_enc)

		#is this correct ? 
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



class GaussianFourierFeatureTransform(torch.nn.Module):
	"""
	An implementation of Gaussian Fourier feature mapping.

	"Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
	   https://arxiv.org/abs/2006.10739
	   https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

	Given an input of size [batches, num_input_channels, width, height],
	 returns a tensor of size [batches, mapping_size*2, width, height].
	"""

	def __init__(self, num_input_channels, mapping_size=256, scale=10):
		super().__init__()

		self._num_input_channels = num_input_channels
		self._mapping_size = mapping_size
		self._B = torch.randn((num_input_channels, mapping_size)) * scale

	def forward(self, x):
		assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

		batches, channels, width, height = x.shape

		assert channels == self._num_input_channels,\
			"Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

		# Make shape compatible for matmul with _B.
		# From [B, C, W, H] to [(B*W*H), C].
		x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

		x = x @ self._B.to(x.device)

		# From [(B*W*H), C] to [B, W, H, C]
		x = x.view(batches, width, height, self._mapping_size)
		# From [B, W, H, C] to [B, C, W, H]
		x = x.permute(0, 3, 1, 2)

		x = 2 * np.pi * x
		return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


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


class AdaIN(nn.Module):
	def __init__(self):
		super().__init__()

	def mu(self, x):
		""" Takes a (n,c,h,w) tensor as input and returns the average across
		it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
		return torch.sum(x,(2,3))/(x.shape[2]*x.shape[3])

	def sigma(self, x):
		""" Takes a (n,c,h,w) tensor as input and returns the standard deviation
		across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
		the permutations are required for broadcasting"""
		return torch.sqrt((torch.sum((x.permute([2,3,0,1])-self.mu(x)).permute([2,3,0,1])**2,(2,3))+0.000000023)/(x.shape[2]*x.shape[3]))

	def forward(self,feature,target_mu,target_sigma):
		"""
			Feature is (N,C,H,W)
		"""
		
		feature_mu = self.mu(feature)
		feature_sigma = self.sigma(feature)
		
		#print(target_mu.shape,target_sigma.shape,feature_mu.shape,feature_sigma.shape)

		return (target_sigma*((feature.permute([2,3,0,1])-feature_mu)/feature_sigma) + target_mu).permute([2,3,0,1])


