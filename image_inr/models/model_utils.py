import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch import Tensor

import math 
import numpy as np
from collections import OrderedDict
import os,sys
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,MetaLinear,MetaDataParallel)
from torchmeta.modules.utils import get_subdict
import hyperlight as hl
import clip

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

		#removes for loop over sine and cosine.
		#bad, but optimal code. lifted from https://github.com/nexuslrf/CoordX/blob/main/modules.py
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



class Reshape_op(torch.nn.Module):
	def __init__(self, shape):
		super().__init__()
		self.shape = shape
		assert len(shape) == 3
		
	def forward(self, x):
		#breakpoint()
		x = x.view(x.size(0),self.shape[0],self.shape[1],self.shape[2])
		return x

class Sine(nn.Module):
	"""Sine activation with scaling.

	Args:
		w0 (float): Omega_0 parameter from SIREN paper.
	"""
	def __init__(self, w0=1.):
		super().__init__()
		self.w0 = w0

	def forward(self, x):
		return torch.sin(self.w0 * x)



class BatchLinear(nn.Linear, MetaModule):
	'''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a hypernetwork.'''
	__doc__ = nn.Linear.__doc__

	#weirdly all args should be before params.
	def forward(self, input,params=None,**kwargs):
		f"""
			Args: 
				input: input for the layer. 
				params: OrderedDict of parameters from hypernet.
				params can be empty when we skip a certain layer in hypernet.
		"""

		if params is None or params == {}: 
			params = OrderedDict(self.named_parameters())

		bias = params.get('bias', None)
		weight = params['weight']		

		
		if 'hypernet_output_type' in kwargs and weight.ndim > 2:
			### Need to check if the layer needs it. 

			hypernet_output_type = kwargs['hypernet_output_type']
			if hypernet_output_type == 'soft_mask':

				#here we already select the predicted weights and send. We need to send both. 
				#need to rewrite a bunch of stuff in hypernets. 
				# can get og weights by 
				og_weight = self.weight

				#activation on predicted weights. 
				weight = weight.sigmoid() - 0.5

				weight = og_weight * (weight + 1.0)
				weight = weight.to(input.dtype)

		output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
		output += bias.unsqueeze(-2)

		return output



class Encoder(nn.Module):
	#def __init__(self, input_channels, num_layers,latent_dim,kernel_size=3,conv_out=32,stride=2,padding=1,img_shape=(224,224),vae_mode=False):
	def __init__(self,input_channels, img_shape, latent_dim, num_layers=3,vae_mode=False):
		super(Encoder, self).__init__()

		layers_list = []
		in_channels = input_channels
		self.input_shape = (in_channels,*img_shape)
		self.latent_dim = latent_dim
		self.vae_mode = vae_mode

		for i in range(num_layers):
			layers_list.extend([
				nn.Conv2d(in_channels=in_channels, out_channels=32*(2**i), kernel_size=3, stride=1, padding=1),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size=2, stride=2),
			])
			in_channels = 32*(2**i)

		self.encoder = nn.Sequential(*layers_list)

		# Calculate the shape of the output feature map after passing through CNN layers
		self.output_shape = self._get_output_shape(self.input_shape)

		out_size = self.latent_dim

		if self.vae_mode:
			out_size = latent_dim*2

		self.fc = nn.Sequential(
			nn.Linear(self.output_shape[0] * self.output_shape[1] * in_channels, 256),
			nn.ReLU(),
			nn.Linear(256, out_size)  
		)

	def _get_output_shape(self, input_shape):
		with torch.no_grad():
			dummy_input = torch.zeros(1, *input_shape)
			dummy_output = self.encoder(dummy_input)
			dummy_output = F.adaptive_avg_pool2d(dummy_output, (1, 1))
			return dummy_output.shape[-2:]

	def forward(self, x):
		x = self.encoder(x)
		x = F.adaptive_avg_pool2d(x, (1, 1))  # Global average pooling to handle variable input sizes
		x = x.view(x.size(0), -1)  # Flatten the feature maps
		x = self.fc(x)

		if self.vae_mode:
			mu,logvar = torch.split(x,self.latent_dim,dim=-1)
			std = torch.exp(0.5*logvar)
			eps = torch.randn_like(std)
			z = mu + eps*std # reparameterization trick
			return z,mu,logvar

		return x


class ResNetEncoder(nn.Module):
	#def __init__(self, input_channels, num_layers,latent_dim,kernel_size=3,conv_out=32,stride=2,padding=1,img_shape=(224,224),vae_mode=False):
	def __init__(self,input_channels, img_shape, latent_dim,vae_mode=False):
		super(ResNetEncoder, self).__init__()

		layers_list = []
		in_channels = input_channels
		self.input_shape = (in_channels,*img_shape)
		self.latent_dim = latent_dim
		self.vae_mode = vae_mode

		self.features = None
		self.model = torchvision.models.resnet18(pretrained=True)
				
		for param in self.model.parameters():
			param.requires_grad = False


		out_size = self.latent_dim

		if self.vae_mode:
			out_size = latent_dim*2
		
		self.model.fc = nn.Linear(512, out_size)

		#transforms for the input image.
		self.mean=[0.48145466, 0.4578275, 0.40821073]
		self.std=[0.26862954, 0.26130258, 0.27577711]
		m = torchvision.transforms.InterpolationMode("bicubic")
		self.resize = torchvision.transforms.Resize((224, 224),m)



	def forward(self, x):

		x = self.resize(x)
		x = torchvision.transforms.Normalize(self.mean,self.std)(x)
		x = self.model(x)

		if self.vae_mode:
			mu,logvar = torch.split(x,self.latent_dim,dim=-1)
			std = torch.exp(0.5*logvar)
			eps = torch.randn_like(std)
			z = mu + eps*std # reparameterization trick
			return z,mu,logvar

		return x



class CLIPWrapper(nn.Module):
	def __init__(self,latent_dim,vae_mode=False):
		super(CLIPWrapper, self).__init__()

		
		self.latent_dim = latent_dim
		self.vae_mode = vae_mode

		self.model, self.preprocess = clip.load("ViT-B/32", device='cuda')
		self.model.eval()
		out_size = self.latent_dim

		if self.vae_mode:
			out_size = latent_dim*2
		
		self.net = nn.Sequential(nn.Linear(512, 256),nn.ReLU(),nn.Linear(256, out_size))


		#transforms for the input image.
		self.mean=[0.48145466, 0.4578275, 0.40821073]
		self.std=[0.26862954, 0.26130258, 0.27577711]
		m = torchvision.transforms.InterpolationMode("bicubic")
		self.resize = torchvision.transforms.Resize((224, 224),m)


	def forward(self, x):

		x = self.resize(x)
		x = torchvision.transforms.Normalize(self.mean,self.std)(x)
		

		with torch.no_grad():
			x = self.model.encode_image(x)
		
		x = self.net(x.float())

		if self.vae_mode:
			mu,logvar = torch.split(x,self.latent_dim,dim=-1)
			std = torch.exp(0.5*logvar)
			eps = torch.randn_like(std)
			z = mu + eps*std # reparameterization trick
			return z,mu,logvar

		return x



class ConvNet(MetaModule):
	def __init__(self, input_channels, num_layers,latent_dim,kernel_size=3,conv_out=32,stride=2,padding=1,img_shape=(224,224),vae_mode=False):
		super(ConvNet, self).__init__()
		self.num_layers = num_layers
		self.input_channels = input_channels
		self.kernel_size = kernel_size
		self.latent_dim = latent_dim
		self.img_shape = img_shape

		self.vae_mode = vae_mode

		if self.vae_mode:
			latent_dim = latent_dim*2

		# Create convolutional layers.
		self.net = []
		print('Hard coded num layers in latent network.')
		self.net.append(MetaSequential(MetaConv2d(input_channels, conv_out, kernel_size=kernel_size, stride=stride, padding=padding),nn.ReLU()))
		self.net.append(MetaSequential(MetaConv2d(conv_out, conv_out//2, kernel_size=kernel_size, stride=stride, padding=padding),nn.ReLU()))
		self.net.append(MetaSequential(MetaConv2d(conv_out//2, conv_out//4, kernel_size=kernel_size, stride=stride, padding=padding),nn.ReLU()))
		self.net.append(MetaSequential(MetaConv2d(conv_out//4, conv_out//8, kernel_size=kernel_size, stride=stride, padding=padding),nn.ReLU()))

		self.net = MetaSequential(*self.net)
		
		x = torch.rand((1,3,*self.img_shape))
		out = self.net(x)
		out = out.view(out.size(0),-1)

		#get shape of fc layer here. 
		self.fc = BatchLinear(out.size()[-1] , latent_dim)


		# # Create final linear layer to output a latent vector of specified size.
		#self.fc = BatchLinear((conv_out//8) * (conv_out//4)**2 , latent_dim)

	def forward(self, x,params=None, **kwargs):
		if params is None or params == {}:
			params = OrderedDict(self.named_parameters())
			
		out = self.net(x, params=get_subdict(params, 'net'))
		# Flatten the tensor and pass through the linear layer.
		out = out.view(out.size(0), -1)
		
		out = self.fc(out, params=get_subdict(params, 'fc'))		


		if self.vae_mode:
			mu,logvar = torch.split(out,self.latent_dim,dim=-1)
			std = torch.exp(0.5*logvar)
			eps = torch.randn_like(std)
			z = mu + eps*std # reparameterization trick

			return z,mu,logvar


		return out



class MLPBlock(hl.HyperModule):
	'''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
	Can be used just as a normal neural network though, as well.
	'''

	def __init__(self, in_features, out_features, num_layers, hidden_features,
				 outermost_linear=False, nonlinearity='relu', weight_init=None,\
					use_nerv_block=False,cfg_network=None,nerv_params=None,hyper_layers=[]):
		
		super().__init__()


		self.first_layer_init = None
		self.cfg_network = cfg_network
		self.use_nerv_block = use_nerv_block
		self.hyper_layers = hyper_layers
		self.num_layers = num_layers
		self.dim_in = in_features
		self.dim_out = out_features

		# Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
		# special first-layer initialization scheme
		nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
						 'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
						 'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
						 'tanh':(nn.Tanh(), init_weights_xavier, None),
						 'selu':(nn.SELU(inplace=True), init_weights_selu, None),
						 'softplus':(nn.Softplus(), init_weights_normal, None),
						 'elu':(nn.ELU(inplace=True), init_weights_elu, None)}

		self.nl, self.nl_weight_init, self.first_layer_init = nls_and_inits[nonlinearity]

		if weight_init is not None:  # Overwrite weight init if passed
			self.weight_init = self.weight_init
		else:
			self.weight_init = self.nl_weight_init


		self.net = []

		for i in range(self.num_layers):
			if i==0:
				in_features = in_features
				out_features = hidden_features
			elif i==self.num_layers-1:
				in_features = hidden_features
				out_features = self.dim_out
			else:
				in_features = hidden_features
				out_features = out_features
			
			if i in self.hyper_layers:
				print('Hyper linear')
				self.linear_module = hl.layers.HyperLinear
			else:
				self.linear_module = nn.Linear

			self.net.append(nn.Sequential(self.linear_module(in_features, out_features), self.nl))
			

		self.net = nn.Sequential(*self.net)
		
		if self.weight_init is not None:
			self.net.apply(self.weight_init)

		if self.first_layer_init is not None: # Apply special initialization to first layer, if applicable.
			self.net[0].apply(self.first_layer_init)

	def forward(self,x):
		output = self.net(x)
		return output



class FCBlock(MetaModule):
	'''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
	Can be used just as a normal neural network though, as well.
	'''

	def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
				 outermost_linear=False, nonlinearity='relu', weight_init=None,\
					use_nerv_block=False,cfg_network=None,nerv_params=None):
		
		super().__init__()

		# nonlinearity = 'sine'

		self.first_layer_init = None
		self.cfg_network = cfg_network
		self.use_nerv_block = use_nerv_block

		# Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
		# special first-layer initialization scheme
		nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
						 'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
						 'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
						 'tanh':(nn.Tanh(), init_weights_xavier, None),
						 'selu':(nn.SELU(inplace=True), init_weights_selu, None),
						 'softplus':(nn.Softplus(), init_weights_normal, None),
						 'elu':(nn.ELU(inplace=True), init_weights_elu, None)}

		nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

		if weight_init is not None:  # Overwrite weight init if passed
			self.weight_init = weight_init
		else:
			self.weight_init = nl_weight_init

		self.net = []
		if hidden_features is not None:
			self.net.append(MetaSequential(
				BatchLinear(in_features, hidden_features), nl
			))

			for i in range(num_hidden_layers-1): # Hidden layers
				self.net.append(MetaSequential(
					BatchLinear(hidden_features, hidden_features), nl
				))

			if outermost_linear:
				self.net.append(MetaSequential(BatchLinear(hidden_features, out_features)))
			else:
				self.net.append(MetaSequential(
					BatchLinear(hidden_features, out_features), nl
				))

		else: # No hidden layers. For bias terms in hypernetworks. Only used there.
			#print('no hidden layers for bias')
			self.net.append(MetaSequential(BatchLinear(in_features, out_features)))

		if self.use_nerv_block:
			
			self.expand_dims = nerv_params['expand_dims']
			self.expand_ch = nerv_params['expand_ch']
			self.expand_w = nerv_params['expand_w']
			self.expand_h = nerv_params['expand_h']
			self.up_sample = nerv_params['up_sample']
			self.volume_out = nerv_params['volume_out']

			self.reshape_op = Reshape_op(shape=(self.expand_ch,self.expand_h,self.expand_w))
			self.final_activation = get_activation(self.cfg_network.final_act)

			if self.cfg_network.batch_size!=1:
				self.net.append(MetaSequential(self.reshape_op,MetaConv2d(in_channels=self.expand_ch,\
					out_channels = self.volume_out*3*(self.up_sample**2),kernel_size=(3,3),stride=(1,1),padding=(1,1)),\
												nn.PixelShuffle(self.up_sample),self.final_activation))
			else:
				self.net.append(MetaSequential(self.reshape_op,MetaConv2d(in_channels=self.expand_ch, \
					out_channels=12, kernel_size=(3,3),stride=(1,1),padding=(1,1)),\
						nn.PixelShuffle(self.up_sample),self.final_activation))		

		self.net = MetaSequential(*self.net)
		
		if self.weight_init is not None:
			self.net.apply(self.weight_init)

		if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
			self.net[0].apply(first_layer_init)

	def forward(self, inputs, hyper=False,params=None, **kwargs):
		"""
			Even hypernet is built using FCblock. 
			hyper flag is used to differentiate between the two, for debugging purposes.
		"""
		if params is None or params == {}:
			params = OrderedDict(self.named_parameters())
		
		breakpoint()
		output = self.net(inputs,params=get_subdict(params, 'net'),**kwargs)
		return output


	def get_last_layer(self):
		if self.cfg_network.batch_size!=1:
			self.nerv_block = MetaSequential(MetaConv2d(in_channels=self.expand_ch,\
				out_channels = self.volume_out*3*(self.up_sample**2), kernel_size=(3,3),stride=(1,1),padding=(1,1)),\
											nn.PixelShuffle(self.up_sample))
		else:
			self.nerv_block = 	MetaSequential(MetaConv2d(in_channels=self.expand_ch, \
				out_channels=12, kernel_size=(3,3),stride=(1,1),padding=(1,1)),nn.PixelShuffle(self.up_sample))	



	def forward_with_activations(self, coords, params=None, retain_grad=False):
		'''Returns not only model output, but also intermediate activations.'''
		if params is None:
			params = OrderedDict(self.named_parameters())

		activations = OrderedDict()

		x = coords.clone().detach().requires_grad_(True)
		activations['input'] = x
		for i, layer in enumerate(self.net):
			subdict = get_subdict(params, 'net.%d' % i)
			for j, sublayer in enumerate(layer):
				if isinstance(sublayer, BatchLinear):
					print(sublayer)
					x = sublayer(x, params=get_subdict(subdict, '%d' % j))
				else:
					x = sublayer(x)

				if retain_grad:
					x.retain_grad()
				activations['_'.join((str(sublayer.__class__), "%d" % i))] = x
		return activations





class SplitFCBlock(MetaModule):
	'''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
	Can be used just as a normal neural network though, as well.
	'''

	def __init__(self, in_features, out_features, num_hidden_layers, hidden_features, outermost_linear=False, 
			nonlinearity='relu', weight_init=None, coord_dim=2, approx_layers=2, split_rule=None, fusion_operator='sum', 
			act_scale=1, fusion_before_act=False, use_atten=False, learn_code=False, last_layer_features=-1, fusion_size=1, reduced=False,
			use_nerv_block=False,cfg_network=None,nerv_params=None):

		super().__init__()
		self.first_layer_init = None

		self.cfg_network = cfg_network
		self.use_nerv_block = use_nerv_block
		self.nerv_params = nerv_params
		

		self.coord_dim = coord_dim
		feat_per_channel = in_features // coord_dim
		if split_rule is None:
			self.feat_per_channel = [feat_per_channel] * coord_dim
		else:
			self.feat_per_channel = [feat_per_channel * k for k in split_rule]
		self.split_channels = len(self.feat_per_channel)
		self.approx_layers = approx_layers
		self.num_hidden_layers = num_hidden_layers
		self.module_prefix = ""
		self.fusion_operator = fusion_operator
		self.fusion_before_act = fusion_before_act
		self.out_features = out_features
		self.use_atten = use_atten
		self.learn_code = learn_code
		self.fusion_size = 1
		self.fusion_feat_size = out_features

		if approx_layers != num_hidden_layers + 1:
			last_layer_features = 1
			self.fusion_size = fusion_size
			self.fusion_feat_size = hidden_features
		elif last_layer_features < 0:
			last_layer_features = hidden_features # Channels
		
		last_layer_features = last_layer_features * out_features

		# Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
		# special first-layer initialization scheme
		nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
						 'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
						 'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
						 'tanh':(nn.Tanh(), init_weights_xavier, None),
						 'selu':(nn.SELU(inplace=True), init_weights_selu, None),
						 'softplus':(nn.Softplus(), init_weights_normal, None),
						 'elu':(nn.ELU(inplace=True), init_weights_elu, None)}

		nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]
		split_scale = act_scale

		if weight_init is not None:  # Overwrite weight init if passed
			self.weight_init = weight_init
		else:
			self.weight_init = nl_weight_init

		s = 1 if reduced else fusion_size
		self.coord_linears = nn.ModuleList(
			[BatchLinear(feat, hidden_features*s) for feat in self.feat_per_channel]
		)
		self.coord_nl = nl
		self.coord_nl.split_scale = split_scale

		if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
			self.coord_linears.apply(first_layer_init)
		else:
			self.coord_linears.apply(self.weight_init)
 
		self.net = []
		i = -1
		for i in range(min(approx_layers, num_hidden_layers)-1):
			self.net.append(MetaSequential(
				BatchLinear(hidden_features*s, hidden_features*s), nl
			))
		i+=1
		self.net.append(MetaSequential(
				BatchLinear(hidden_features*s, hidden_features*fusion_size), nl
			))
		for j in range(i+1, num_hidden_layers):
			self.net.append(MetaSequential(
				BatchLinear(hidden_features, hidden_features), nl
			))

		if outermost_linear:
			self.net.append(MetaSequential(BatchLinear(hidden_features, last_layer_features)))
		else:
			self.net.append(MetaSequential(
				BatchLinear(hidden_features, last_layer_features), nl
			))

		self.net = MetaSequential(*self.net)
		if self.weight_init is not None:
			self.net.apply(self.weight_init)
		for i in range(self.approx_layers):
			try:
				self.net[i][0].num_input = self.split_channels
				self.net[i][1].split_scale = split_scale
			except:
				pass
		if use_atten:
			self.atten = BatchLinear(in_features, hidden_features*fusion_size)
			self.atten.apply(self.weight_init)
		if fusion_before_act and nonlinearity.endswith('elu') and \
			self.approx_layers-1 != self.num_hidden_layers + 1:
			
			self.net[self.approx_layers-1][1].inplace = False
		if learn_code:
			self.code = nn.parameter.Parameter(torch.ones(hidden_features*fusion_size))
			

	def forward(self, coords, params=None, pos_codes=None, split_coord=True, ret_feat=False, **kwargs):
		"""
		When split_coord=True, the input coords should be a list a tensor for each coord.
		the length of each coord tensor do not need to be the same. But the dimension of each coord tensor
		should be predefined for broadcasting operation.
		"""
		if split_coord:

			if params is None or params == {}:
				params = OrderedDict(self.named_parameters())

			#this gives [1,W,1] and [H,1,1] for X and Y. 			
			hs = [self.forward_channel(coord.unsqueeze(i), i, pos_codes,params=params) for i, coord in enumerate(coords)]
			h = self.forward_fusion(hs,params=params)
			sh = h.shape

			if ret_feat:
				#return (h.reshape(sh[0], -1, sh[-1]), hs)
				return (h.reshape(-1, sh[-1]), hs)
			else:
				# return h.reshape(sh[0], -1, sh[-1])
				return h.reshape(-1, sh[-1])
	
	def forward_channel(self, coord, channel_id, pos_codes=None, params=None):
		h = self.coord_linears[channel_id](coord,params=get_subdict(params, 'coord_linears.%d' % channel_id))
		h = self.coord_nl(h)

		if self.approx_layers > 0:
			for i in range(self.approx_layers-1):
				h = self.net[i](h,params=get_subdict(params, 'net.%d' % i))
			# layer before fusion
			if not self.fusion_before_act:
				# for simple fusion strategies
				h = self.net[self.approx_layers-1](h,params=get_subdict(params, 'net.%d' % (self.approx_layers-1)))
			else:
				# fusion before activation
				h = self.net[self.approx_layers-1][0](h,params=get_subdict(params, 'net.%d.0' % (self.approx_layers-1)))
			if pos_codes is not None:
				# h = (h * pos_codes)
				h = (h + pos_codes)
		return h
	
	def forward_fusion(self, hs,params=None):
		'''
		When do the fusion, it will expand the list of coord into a grid. 
		In this case, data dimension needs to be predefine. E.g.,
			X: [1,128,1], Y: [64,1,1] --> [64,128,1]
		'''
		
		# if not isinstance(hs, torch.Tensor):
		h = hs[0]
		for hi in hs[1:]:
			if self.fusion_operator == 'sum':
				h = h + hi
			elif self.fusion_operator == 'prod':
				h = h * hi

		if self.fusion_before_act:
			h = self.net[self.approx_layers-1][1](h)
		if self.learn_code:
			h = h * self.code
		# if self.approx_layers == self.num_hidden_layers + 1:
		h_sh = h.shape
		if h_sh[-1] > self.fusion_feat_size:
			h = h.reshape(*h_sh[:-1], self.fusion_feat_size, -1).sum(-1)
		

		for i in range(self.approx_layers, self.num_hidden_layers+1):
			h = self.net[i](h,params=get_subdict(params, 'net.%d' % (i+self.approx_layers))) #approx layers forward done. 
		return h


########################
# HyperNetwork modules
class HyperNetwork(nn.Module):
	#HyperNetwork(cfg=self.cfg,hyper_in_features=self.latent_dim,hypo_modules=self.hypo_modules)
	def __init__(self, cfg,hyper_in_features,hypo_modules,hyper_layers=None):
		'''
			HyperNets also use FCBlocks. Be careful while doing breakpoint. 

		Args:
			hyper_in_features: In features of hypernetwork
			hyper_hidden_layers: Number of hidden layers in hypernetwork
			hyper_hidden_features: Number of hidden units in hypernetwork
			hypo_module: MetaModule. The module whose parameters are predicted.
		'''
		super().__init__()

		hypo_parameters = hypo_modules.meta_named_parameters()

		self.cfg = cfg
		self.hyper_in_features = hyper_in_features
		self.hypo_modules = hypo_modules
		self.hyper_layers = hyper_layers

		#hypernet properties
		self.hyper_net_config = self.cfg.network.hyper_net

		self.hyper_net_output = self.hyper_net_config.output_type
		self.hyper_hidden_layers = self.hyper_net_config.num_layers
		self.hyper_hidden_features = self.hyper_net_config.hidden_dim
		
		

		self.names = []
		self.nets = nn.ModuleList()
		self.param_shapes = []
		

		
		for idx,(name, param) in enumerate(hypo_parameters):
			
			# if idx not in self.hyper_layers:
			# 	continue
			# this wont work as the idx includes bias as well.
			layer_idx = int(name.split('.')[1])
			if layer_idx not in self.hyper_layers or ('net' not in name):
				continue

			self.names.append(name)
			self.param_shapes.append(param.size())

			if self.hyper_net_output == 'weights':
				out_size = int(torch.prod(torch.tensor(param.size())))
			elif self.hyper_net_output == 'soft_mask':
				self.hypernet_mask_rank = self.hyper_net_config.mask_rank
				out_size = torch.sum(torch.tensor(param.size())) * self.hypernet_mask_rank


			if 'weight' in name:
				hn = FCBlock(in_features=hyper_in_features, out_features=out_size,
						num_hidden_layers=self.hyper_hidden_layers, hidden_features=self.hyper_hidden_features,outermost_linear=True)
				hn.net[-1].apply(lambda m: hyper_weight_init(m, param.size()[-1]))

			elif 'bias' in name:
				#bias gets a matrix. num_hidden_layers = 0
				bias_out_size = int(torch.prod(torch.tensor(param.size())))
				#bias is same for all cases.
				hn = FCBlock(in_features=hyper_in_features, out_features=bias_out_size,
						num_hidden_layers=0, hidden_features=None)
				hn.net[-1].apply(lambda m: hyper_bias_init(m))
			self.nets.append(hn)
		
		print(self.names)
		

	def forward(self, z):
		'''
		Args:-
			z: Embedding. Input to hypernetwork. Could be output of "Autodecoder" (see above)

		Returns:
			params: OrderedDict. Can be directly passed as the "params" parameter of a MetaModule.
		'''
		params = OrderedDict()
		for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
			out_w = net(z,hyper=True)
			batch_size = z.size(0)

			#bias is same for all modulations.
			if self.hyper_net_output == 'weights' or 'bias' in name:
				batch_param_shape = (-1,) + param_shape
				params[name] = out_w.reshape(batch_param_shape)

			elif self.hyper_net_output == 'soft_mask':
				rank = self.hypernet_mask_rank
				c_out,c_in = param_shape
				left_matrix = out_w[:, :c_out * rank] # [batch_size, left_matrix_size]
				right_matrix = out_w[:, c_out * rank:] # [batch_size, right_matrix_size]

				left_matrix = left_matrix.view(batch_size, c_out, rank)
				right_matrix = right_matrix.view(batch_size, rank, c_in)
				modulation = left_matrix @ right_matrix / np.sqrt(self.hypernet_mask_rank) # [batch_size, c_out, c_in]
				assert modulation.shape[1:] == param_shape
				params[name] = modulation

				"""
					activation done in forward pass. 
				"""

		return params

def fmm_modulate_linear(x: Tensor, weight: Tensor, styles: Tensor, noise=None, activation: str="demod") -> Tensor:
    """
    x: [batch_size, c_in, height, width]
    weight: [c_out, c_in, 1, 1] : Shared across all batch elements. 
    style: [batch_size, num_mod_params] : num_mod_params = c_out * rank + c_in * rank
    noise: Optional[batch_size, 1, height, width]
    """
    batch_size, c_in, h, w = x.shape
    c_out, c_in, kh, kw = weight.shape
    rank = styles.shape[1] // (c_in + c_out)

    assert kh == 1 and kw == 1
    assert styles.shape[1] % (c_in + c_out) == 0

    # Now, we need to construct a [c_out, c_in] matrix
    left_matrix = styles[:, :c_out * rank] # [batch_size, left_matrix_size]
    right_matrix = styles[:, c_out * rank:] # [batch_size, right_matrix_size]

    left_matrix = left_matrix.view(batch_size, c_out, rank) # [batch_size, c_out, rank]
    right_matrix = right_matrix.view(batch_size, rank, c_in) # [batch_size, rank, c_in]

    # Imagine, that the output of `self.affine` (in SynthesisLayer) is N(0, 1)
    # Then, std of weights is sqrt(rank). Converting it back to N(0, 1)
    modulation = left_matrix @ right_matrix / np.sqrt(rank) # [batch_size, c_out, c_in]

    if activation == "tanh":
        modulation = modulation.tanh()
    elif activation == "sigmoid":
        modulation = modulation.sigmoid() - 0.5

    W = weight.squeeze(3).squeeze(2).unsqueeze(0) * (modulation + 1.0) # [batch_size, c_out, c_in]
    if activation == "demod":
        W = W / (W.norm(dim=2, keepdim=True) + 1e-8) # [batch_size, c_out, c_in]
    W = W.to(dtype=x.dtype)

    # out = torch.einsum('boi,bihw->bohw', W, x)
    x = x.view(batch_size, c_in, h * w) # [batch_size, c_in, h * w]
    out = torch.bmm(W, x) # [batch_size, c_out, h * w]
    out = out.view(batch_size, c_out, h, w) # [batch_size, c_out, h, w]

    if not noise is None:
        out = out.add_(noise)

    return out



############################### Other Functions #########################################################

############################
# Initialization scheme
def hyper_weight_init(m, in_features_main_net, siren=False):
	if hasattr(m, 'weight'):
		nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
		m.weight.data = m.weight.data / 1e1

	# if hasattr(m, 'bias') and siren:
	#     with torch.no_grad():
	#         m.bias.uniform_(-1/in_features_main_net, 1/in_features_main_net)


def hyper_bias_init(m, siren=False):
	if hasattr(m, 'weight'):
		nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
		m.weight.data = m.weight.data / 1.e1

	# if hasattr(m, 'bias') and siren:
	#     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
	#     with torch.no_grad():
	#         m.bias.uniform_(-1/fan_in, 1/fan_in)




########################
# Initialization methods
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
	# Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
	# grab from upstream pytorch branch and paste here for now
	def norm_cdf(x):
		# Computes standard normal cumulative distribution function
		return (1. + math.erf(x / math.sqrt(2.))) / 2.

	with torch.no_grad():
		# Values are generated by using a truncated uniform distribution and
		# then using the inverse CDF for the normal distribution.
		# Get upper and lower cdf values
		l = norm_cdf((a - mean) / std)
		u = norm_cdf((b - mean) / std)

		# Uniformly fill tensor with values from [l, u], then translate to
		# [2l-1, 2u-1].
		tensor.uniform_(2 * l - 1, 2 * u - 1)

		# Use inverse cdf transform for normal distribution to get truncated
		# standard normal
		tensor.erfinv_()

		# Transform to proper mean, std
		tensor.mul_(std * math.sqrt(2.))
		tensor.add_(mean)

		# Clamp to ensure it's in the proper range
		tensor.clamp_(min=a, max=b)
		return tensor


def init_weights_trunc_normal(m):
	# Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
	if type(m) == BatchLinear or type(m) == nn.Linear:
		if hasattr(m, 'weight'):
			fan_in = m.weight.size(1)
			fan_out = m.weight.size(0)
			std = math.sqrt(2.0 / float(fan_in + fan_out))
			mean = 0.
			# initialize with the same behavior as tf.truncated_normal
			# "The generated values follow a normal distribution with specified mean and
			# standard deviation, except that values whose magnitude is more than 2
			# standard deviations from the mean are dropped and re-picked."
			_no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)


def init_weights_normal(m):
	if type(m) == BatchLinear or type(m) == nn.Linear:
		if hasattr(m, 'weight'):
			nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):
	if type(m) == BatchLinear or type(m) == nn.Linear:
		if hasattr(m, 'weight'):
			num_input = m.weight.size(-1)
			nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
	if type(m) == BatchLinear or type(m) == nn.Linear:
		if hasattr(m, 'weight'):
			num_input = m.weight.size(-1)
			nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
	if type(m) == BatchLinear or type(m) == nn.Linear:
		if hasattr(m, 'weight'):
			nn.init.xavier_normal_(m.weight)


def sine_init(m):
	with torch.no_grad():
		if hasattr(m, 'weight'):
			num_input = m.weight.size(-1)
			# See supplement Sec. 1.5 for discussion of factor 30
			m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
	with torch.no_grad():
		if hasattr(m, 'weight'):
			num_input = m.weight.size(-1)
			# See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
			m.weight.uniform_(-1 / num_input, 1 / num_input)


###################
# Complex operators
def compl_conj(x):
	y = x.clone()
	y[..., 1::2] = -1 * y[..., 1::2]
	return y


def compl_div(x, y):
	''' x / y '''
	a = x[..., ::2]
	b = x[..., 1::2]
	c = y[..., ::2]
	d = y[..., 1::2]

	outr = (a * c + b * d) / (c ** 2 + d ** 2)
	outi = (b * c - a * d) / (c ** 2 + d ** 2)
	out = torch.zeros_like(x)
	out[..., ::2] = outr
	out[..., 1::2] = outi
	return out


def compl_mul(x, y):
	'''  x * y '''
	a = x[..., ::2]
	b = x[..., 1::2]
	c = y[..., ::2]
	d = y[..., 1::2]

	outr = a * c - b * d
	outi = (a + b) * (c + d) - a * c - b * d
	out = torch.zeros_like(x)
	out[..., ::2] = outr
	out[..., 1::2] = outi
	return out




if __name__ == '__main__':
		

	target_mean = torch.zeros(2, 3)
	target_std = torch.ones(2, 3)
	ada = AdaIN()

	x = torch.randn(2,3,64,64)

	out = ada(x,target_mean,target_std)
	print(out.shape)
	
	print('og stats: ',x.mean([0,2,3]),x.std([0,2,3]))
	print('new stats: ',out.mean([0,2,3]),out.std([0,2,3]))

	#Need to align the stats of outputs to closely match stats of patch. 


