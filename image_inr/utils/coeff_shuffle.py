import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
from . import dct


class CoefficientShuffler(torch.nn.Module):
	def __init__(self, channels, direction='channels', block_size=8,remove_zigzag=False):
		super(CoefficientShuffler, self).__init__()
		self._channels = channels
		self._direction = direction
		self.block_size = block_size
		self.remove_zigzag = remove_zigzag

		self.schema = self.get_schema((block_size,block_size))

	def forward(self, x, pad=None):
		if self._direction == 'channels':
			return self.channels(x)
		elif self._direction == 'blocks':
			return self.blocks(x, pad)
		
	def channels(self, x):
		"""
			Returns N,num_freqs,channels,block_size,block_size
		"""
		blocks = torch.nn.functional.unfold(x, kernel_size=self.block_size, stride=self.block_size)
		blocks = blocks.transpose(1, 2).contiguous().view(-1, x.shape[2] // self.block_size, x.shape[3] // self.block_size, self._channels, self.block_size*self.block_size)
		blocks = blocks.transpose(2, 3).transpose(1, 2)
		blocks = blocks.transpose(3, 4).transpose(2, 3).transpose(1, 2)
		#blocks = blocks.contiguous().view(-1, (self.block_size*self.block_size) * self._channels, x.shape[2] // self.block_size, x.shape[3] // self.block_size)

		#keep channels separate?
		blocks = blocks.contiguous().view(-1, (self.block_size*self.block_size) , self._channels, x.shape[2] // self.block_size, x.shape[3] // self.block_size)

		if self.remove_zigzag:
			blocks = blocks[:,self.schema,:,:,:]

		return blocks
	
	def blocks(self, x, pad):
		"""
			x needs to be (N,num_freqs,channels,block_size,block_size)
		"""

		if self.remove_zigzag:
			x = x.index_copy(1,self.schema,x.clone()) #1 because num_freqs is dim 1

		# This is just the inverse procedure from channels		
		blocks = x
		#blocks = x.view(-1, self.block_size**2 , self._channels, x.shape[2], x.shape[3])
		
		blocks = blocks.transpose(1, 2).transpose(2, 3).transpose(3, 4)
		blocks = blocks.transpose(1, 2).transpose(2, 3)
		blocks = blocks.contiguous().view(-1, x.shape[-2] * x.shape[-1], self._channels * (self.block_size*self.block_size))
		blocks = blocks.transpose(1, 2)
	
		blocks = torch.nn.functional.fold(blocks, kernel_size=self.block_size, stride=self.block_size, output_size=(x.shape[-2] * self.block_size, x.shape[-1] * self.block_size))
		if pad is not None:
			diffY = pad.shape[2] - blocks.shape[2]
			diffX = pad.shape[3] - blocks.shape[3]
			blocks = torch.nn.functional.pad(blocks, pad=(diffX // 2, diffX - diffX // 2,
														  diffY // 2, diffY - diffY // 2))
		return blocks

	def get_schema(self,patch_shape):
		schema = np.arange(patch_shape[0]*patch_shape[1])
		schema = schema.reshape(patch_shape[0],patch_shape[1])
		schema = dct.zigzag(schema)
		schema = torch.tensor(schema).long()
		return schema



class CoeffShuffle(object):
	def __init__(self,block_size=8,remove_zigzag=False):
		self.coeff_convert = CoefficientShuffler(3,direction='channels',block_size=block_size,remove_zigzag=remove_zigzag)
	def __call__(self, tensor):
		device = tensor.device 
		return self.coeff_convert(tensor)

class CoeffShuffleBack(object):
	def __init__(self,block_size=8,remove_zigzag=False):
		self.coeff_reconvert = CoefficientShuffler(3,direction='blocks',block_size=block_size,remove_zigzag=remove_zigzag)
	def __call__(self, tensor):
		device = tensor.device 
		return self.coeff_reconvert(tensor)


if __name__ == '__main__':

	from PIL import Image 
	import numpy as np
	from utils import dct

	def load_tensor(path):
		img = Image.open(path)
		#resize to 256
		#img = img.resize((128,128))
		img = np.array(img)
		img = img.transpose(2,0,1)
		img = torch.from_numpy(img).float() / 255.0
		img = img.unsqueeze(0)
		return img

	def get_schema(patch_shape):
		schema = np.arange(patch_shape[0]*patch_shape[1])
		schema = schema.reshape(patch_shape[0],patch_shape[1])
		schema = dct.zigzag(schema)
		schema = torch.tensor(schema).long()
		return schema

	for bs in [8,16,32,48,64]:

		img = load_tensor('bunny_debug/f00001.png')
		img = dct.batch_dct(img,block_size=bs)

		#pad if not divisible by block size
		if img.shape[2] % bs != 0 or img.shape[3] % bs != 0:
			print('padding for block size: ',bs)
			# img = torch.zeros(img.shape[0],img.shape[1],img.shape[2] + (bs - img.shape[2] % bs),img.shape[3] + (bs - img.shape[3] % bs))
			pad = torch.zeros(img.shape[0],img.shape[1],img.shape[2] + (bs - img.shape[2] % bs),img.shape[3] + (bs - img.shape[3] % bs))
			pad[:,:,:img.shape[2],:img.shape[3]] = img
			img = pad

		coeff_shuffle = CoeffShuffle(block_size=bs,remove_zigzag=True)
		shuffled = coeff_shuffle(img)
		coeff_reconvert = CoeffShuffleBack(block_size=bs,remove_zigzag=True)
		shuffled_back = coeff_reconvert(shuffled)

		print(shuffled.shape,shuffled_back.shape)

		print(torch.allclose(img,shuffled_back))
