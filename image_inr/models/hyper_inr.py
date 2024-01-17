import torch 
import torch.nn as nn
import numpy as np
import math
from pydoc import locate


from models import BaseModel
from . import simple_vit
from . import hyper_net

from .layers import layer_utils,meta_mlp,encoders

import loralib as lora

"""
	Uses concepts from https://arxiv.org/pdf/2210.08942.pdf for INR models. 
"""


class hyper_inr(BaseModel):
	
	def __init__(self,cfg):
		super().__init__(cfg)

		self.final_activation = layer_utils.get_activation(self.cfg.network.final_act)
		self.fc_block = meta_mlp.FCBlock(in_features=self.in_features,out_features=self.dim_out, 
						num_hidden_layers=self.num_layers-1, #absolute. 
						hidden_features=self.dim_hidden, outermost_linear=True, 
						nonlinearity=self.activation,use_nerv_block=False,cfg_network=self.cfg.network)

		self.hypo_modules = self.fc_block 
		
		self.latent_dim = self.cfg.network.latent_dim
		self.latent_net = None
		self.random_latent = False
		self.image_shape = self.cfg.data.image_shape


		########## Setup Latent Network ##########

		if self.cfg.network.latent_network.layers is not None:
			
			latent_net_config = self.cfg.network.latent_network

			if latent_net_config.type == 'vit':
				self.latent_net = simple_vit.SimpleViT(image_size=self.cfg.network.sidelength,\
					   patch_size=latent_net_config.patch_size,num_classes=None,dim=self.latent_dim,\
					depth=latent_net_config.layers,heads=latent_net_config.heads,mlp_dim=latent_net_config.mlp_dim)
				
				print('Using ViT latent network')

			elif self.cfg.network.latent_network.type == 'cnn':
				#fixed layers.
				self.latent_net = encoders.ConvNet(input_channels=3,latent_dim=self.latent_dim,\
						num_layers=self.cfg.network.latent_network.layers,kernel_size=3,img_shape=self.image_shape)
				print('Using CNN latent network')
			else:
				self.latent_net = nn.Identity()

		######## Hyper Network ########################
		hyper_layers = self.cfg.network.hyper_net.layers if self.cfg.network.hyper_net.layers is not None else list(range(self.cfg.network.num_layers))
		
		self.hyper_net_args = {'hypernet_output_type':self.cfg.network.hyper_net.output_type,'activation':self.cfg.network.hyper_net.mask_act}

		self.hyper_net = hyper_net.HyperNetwork(cfg=self.cfg,hyper_in_features=self.latent_dim,\
					    hypo_modules=self.hypo_modules,hyper_layers=hyper_layers)
		self.hypernet_output_type = self.cfg.network.hyper_net.output_type



		####################### Weight encoder #######################

		"""
			Weight encoder ensures that the predicted weights from the hypernet 
			correspond to CLIP. 
		"""
		
		param_size = []
		for name,param in self.fc_block.meta_named_parameters():
			if name in self.hyper_net.names:
				param_size.append(np.prod(param.shape))

		if self.cfg.network.weight_encoder.num_layers > 0:
			self.weight_encoder = meta_mlp.FCBlock(in_features=sum(param_size),out_features=self.latent_dim,\
											num_hidden_layers=self.cfg.network.weight_encoder.num_layers,\
											hidden_features=self.cfg.network.weight_encoder.hidden_dim,nonlinearity='softplus',outermost_linear=True)

		else:
			self.weight_encoder = None


	def run_hyper_net(self,x,z,inference=False):
		params = self.hyper_net(z)
		params = {k: v.squeeze(0) for k, v in params.items()}
		out = self.fc_block(x,params=params,**self.hyper_net_args) #hypernet_output_type=self.hypernet_output_type)
		out = self.final_activation(out)

		if (not inference) and (self.weight_encoder is not None):
			#hyperclip operations. 
			all_params = []
			for k,v in params.items():
				all_params.append(v.view(-1))
			all_params = torch.cat(all_params,dim=0).unsqueeze(0)
			
			weight_enc_out = self.weight_encoder(all_params)
			output = {'predicted':out,'weight_enc_out':weight_enc_out}

		output = {'predicted':out}
		return output

	def forward(self,coords,img=None,z=None,inference=False,**kwargs):
		"""
			x: Nx2
		"""
		
		coords = self.positional_encoding(coords)

		if z is None:
			if self.latent_net is not None and img is not None:
				z = self.latent_net(img)
			elif self.random_latent:
				z = torch.randn((1,self.latent_dim)).cuda()
		
		outputs = self.run_hyper_net(coords,z,inference=inference)

		return outputs


if __name__ == '__main__':
	import easydict,time 


	#test fourier features. 
	args = easydict.EasyDict({'network':{'layer_size':128,'num_layers':4,'w0':30.0,'w0_initial':30.0,'final_act':None,\
		'batch_size':1,'pos_encoding':True,'sidelength':224,'hash_grid_encoding':{'enabled':True},\
			'weight_encoder':{'num_layers':0},'num_frequencies':100,'latent_dim':128,\
			'hyper_net':{'layers':[1],'num_layers':1,'hidden_dim':256},\
			'latent_network':{'layers':2,'random_latent':False,'type':'vit','patch_size':8,'heads':1,'mlp_dim':512}},\
				'image_shape':(224,224), 'dataset':{'image_size':224}  })

				# self.latent_net = simple_vit.SimpleViT(image_size=self.cfg.network.sidelength,\
				# 	   patch_size=latent_net_config.patch_size,num_classes=None,dim=self.latent_dim,\
				# 	depth=latent_net_config.layers,heads=latent_net_config.heads,mlp_dim=latent_net_config.mlp_dim)


	img_shape = 224
	
	#Nx2
	x = torch.randn((4,3,224,224)).cuda()
	coord = torch.randn((224*224,2)).cuda()

	args.network.pos_encoding = True
	args.network.hash_grid_encoding.enabled = False

	model = hyper_inr(dim_in=2, dim_out=3,cfg=args)
	model = model.cuda()

	# from torchmeta.modules import MetaDataParallel
	# #model = nn.DataParallel(model)
	# model.latent_net = MetaDataParallel(model.latent_net)
	# model = MetaDataParallel(model)
	# device = torch.device('cuda')
	# model.to(device)

	out = model(coord,img=x)
	breakpoint()
	

	print(out['predicted'].shape)
	