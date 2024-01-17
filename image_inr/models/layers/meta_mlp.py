
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

from . import layer_utils

def fmm_modulate_linear(og_weight: Tensor,modulation:Tensor,activation: str="demod") -> Tensor:

	if activation == "tanh":
		modulation = modulation.tanh()
	elif activation == "sigmoid":
		modulation = modulation.sigmoid() - 0.5

	
	W = og_weight * (modulation + 1.0)  # [c_out, c_in]

	if activation == "demod":
		W = W / (W.norm(dim=2, keepdim=True) + 1e-8) # [batch_size, c_out, c_in]
	return W


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

			"""
				Should only be applied to weights where needed. 
			"""

			hypernet_output_type = kwargs['hypernet_output_type']
			if hypernet_output_type == 'soft_mask':				
				og_weight = self.weight
				#assert the matrices arent the same. 
				assert not torch.equal(og_weight,weight)

				modulation = weight #we get the modulation from hypernet.
				weight = fmm_modulate_linear(og_weight,modulation,activation=kwargs['activation'])
				weight = weight.to(input.dtype)
				

		output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
		output += bias.unsqueeze(-2)

		return output


class FCBlock(MetaModule):
	'''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
	Can be used just as a normal neural network though, as well.
	 
	Using these layers allows us to perform meta-learning on the weights of the network.
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
		nls_and_inits = {'sine':(layer_utils.Sine(), layer_utils.sine_init, layer_utils.first_layer_sine_init),
						 'relu':(nn.ReLU(inplace=True), layer_utils.init_weights_normal, None),
						 'sigmoid':(nn.Sigmoid(), layer_utils.init_weights_xavier, None),
						 'tanh':(nn.Tanh(), layer_utils.init_weights_xavier, None),
						 'leakyrelu':(nn.LeakyReLU(inplace=True), layer_utils.init_weights_normal, None),
						 'selu':(nn.SELU(inplace=True), layer_utils.init_weights_selu, None),
						 'softplus':(nn.Softplus(), layer_utils.init_weights_normal, None),
						 'elu':(nn.ELU(inplace=True), layer_utils.init_weights_elu, None)}

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

			self.reshape_op = layer_utils.Reshape_op(shape=(self.expand_ch,self.expand_h,self.expand_w))
			self.final_activation = layer_utils.get_activation(self.cfg_network.final_act)

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
	
if __name__ == '__main__':
	from .. import hyper_net

	net = FCBlock(in_features=3,out_features=3,num_hidden_layers=3,\
		hidden_features=256,outermost_linear=True,nonlinearity='leakyrelu')

	inputs = torch.randn(5,100,2)	
	weights = torch.randn(5,256,256)
	bias = torch.randn(5,)
	params = {'net.0.0.weight':weights,'net.0.0.bias':bias}

	out = net(inputs,params=params,**{'hypernet_output_type':'soft_mask','activation':'demod'})

