import torch
import torch.nn as nn
import torchvision

from collections import OrderedDict
import clip

class CLIP_latent_encoder(nn.Module):
	def __init__(self,latent_dim,vae_mode=False):
		super(CLIP_latent_encoder, self).__init__()

		
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


class CLIP_encoder(nn.Module):
	def __init__(self):
		super(CLIP_encoder, self).__init__()		

		self.model, self.preprocess = clip.load("ViT-B/32", device='cuda')
		self.model.eval()

	def forward(self, x):

		with torch.no_grad():
			x = self.model.encode_image(x)

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




class ConvNet(nn.Module):
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
		self.net.append(nn.Sequential(nn.Conv2d(input_channels, conv_out, kernel_size=kernel_size, stride=stride, padding=padding),nn.ReLU()))
		self.net.append(nn.Sequential(nn.Conv2d(conv_out, conv_out//2, kernel_size=kernel_size, stride=stride, padding=padding),nn.ReLU()))
		self.net.append(nn.Sequential(nn.Conv2d(conv_out//2, conv_out//4, kernel_size=kernel_size, stride=stride, padding=padding),nn.ReLU()))
		self.net.append(nn.Sequential(nn.Conv2d(conv_out//4, conv_out//8, kernel_size=kernel_size, stride=stride, padding=padding),nn.ReLU()))

		self.net = nn.Sequential(*self.net)
		
		x = torch.rand((1,3,*self.img_shape))
		out = self.net(x)
		out = out.view(out.size(0),-1)

		#get shape of fc layer here. 
		self.fc = nn.Linear(out.size()[-1] , latent_dim)


		# # Create final linear layer to output a latent vector of specified size.
		#self.fc = nn.Linear((conv_out//8) * (conv_out//4)**2 , latent_dim)

	def forward(self, x):
		out = self.net(x )
		# Flatten the tensor and pass through the linear layer.
		out = out.view(out.size(0), -1)
		out = self.fc(out)		

		if self.vae_mode:
			mu,logvar = torch.split(out,self.latent_dim,dim=-1)
			std = torch.exp(0.5*logvar)
			eps = torch.randn_like(std)
			z = mu + eps*std # reparameterization trick

			return z,mu,logvar


		return out

