import torch
import torch.nn as nn

from .layers import mlp, layer_utils

from collections import OrderedDict
import numpy as np


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
				hn = mlp.MLPBlock(in_features=hyper_in_features, out_features=out_size,
						num_hidden_layers=self.hyper_hidden_layers,nonlinearity=self.hyper_net_config.nl, \
						hidden_features=self.hyper_hidden_features,outermost_linear=True)
				
				hn.net[-1].apply(lambda m: layer_utils.hyper_weight_init(m, param.size()[-1]))

			elif 'bias' in name:
				#bias gets a matrix. num_hidden_layers = 0
				bias_out_size = int(torch.prod(torch.tensor(param.size())))
				#bias is same for all cases.
				hn = mlp.MLPBlock(in_features=hyper_in_features, out_features=bias_out_size,
						num_hidden_layers=0, hidden_features=None)
				hn.net[-1].apply(lambda m: layer_utils.hyper_bias_init(m))
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
			out_w = net(z)
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


