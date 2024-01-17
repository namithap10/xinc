import torch
import torch.nn as nn
from . import layer_utils
from collections import OrderedDict

class MLPBlock(nn.Module):
	'''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
	Can be used just as a normal neural network though, as well.
	'''

	def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
				 outermost_linear=True, nonlinearity='relu', weight_init=None,\
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
						 'leaky_relu':(nn.LeakyReLU(inplace=True), layer_utils.init_weights_normal, None),
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
			self.net.append(nn.Sequential(
				nn.Linear(in_features, hidden_features), nl
			))

			for i in range(num_hidden_layers-1): # Hidden layers
				self.net.append(nn.Sequential(
					nn.Linear(hidden_features, hidden_features), nl
				))

			if outermost_linear:
				self.net.append(nn.Sequential(nn.Linear(hidden_features, out_features)))
			else:
				self.net.append(nn.Sequential(
					nn.Linear(hidden_features, out_features), nl
				))

		else: # No hidden layers. For bias terms in hypernetworks. Only used there.
			#print('no hidden layers for bias')
			self.net.append(nn.Sequential(nn.Linear(in_features, out_features)))

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
				self.net.append(nn.Sequential(self.reshape_op,nn.Conv2d(in_channels=self.expand_ch,\
					out_channels = self.volume_out*3*(self.up_sample**2),kernel_size=(3,3),stride=(1,1),padding=(1,1)),\
												nn.PixelShuffle(self.up_sample),self.final_activation))
			else:
				self.net.append(nn.Sequential(self.reshape_op,nn.Conv2d(in_channels=self.expand_ch, \
					out_channels=12, kernel_size=(3,3),stride=(1,1),padding=(1,1)),\
						nn.PixelShuffle(self.up_sample),self.final_activation))		

		self.net = nn.Sequential(*self.net)
		
		if self.weight_init is not None:
			self.net.apply(self.weight_init)

		if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
			self.net[0].apply(first_layer_init)


	def forward(self, inputs):
		output = self.net(inputs)
		return output


if __name__ == '__main__':
	net = MLPBlock(in_features=3,out_features=3,num_hidden_layers=3,\
		hidden_features=256,outermost_linear=True,nonlinearity='leakyrelu')

	inputs = torch.randn(1,3)	

	out = net(inputs)

	breakpoint()
