import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F

import math
import numpy as np
import cv2
from PIL import Image 


#Fold and unfold help in applying a function in a sliding window fashion.
def blockify(im, size):
	bs = im.shape[0]
	ch = im.shape[1]
	h = im.shape[2]
	w = im.shape[3]

	im = im.view(bs * ch, 1, h, w)
	im = torch.nn.functional.unfold(im, kernel_size=size, stride=size)
	im = im.transpose(1, 2)
	im = im.view(bs * ch, -1, size, size)

	return im


def deblockify(blocks, ch, size):
	bs = blocks.shape[0] // ch
	block_size = blocks.shape[2]

	blocks = blocks.reshape(bs * ch, -1, block_size**2)
	blocks = blocks.transpose(1, 2)
	blocks = torch.nn.functional.fold(blocks, output_size=size, kernel_size=block_size, stride=block_size)
	blocks = blocks.reshape(bs, ch, size[0], size[1])

	return blocks


def normalize(N):
	n = torch.ones((N, 1))
	n[0, 0] = 1 / math.sqrt(2) #works.
	return (n @ n.t())


def harmonics(N):
	spatial = torch.arange(float(N)).reshape((N, 1))
	spectral = torch.arange(float(N)).reshape((1, N))

	spatial = 2 * spatial + 1
	spectral = (spectral * math.pi) / (2 * N)

	return torch.cos(spatial @ spectral)


def block_dct(im, device=None):
	N = im.shape[3] # dimension of block. Eg 8

	n = normalize(N) #seems fine.
	h = harmonics(N)

	if device is not None:
		n = n.to(device)
		h = h.to(device)
	#coeff = (1 / math.sqrt(2 * N)) * n * (h.t() @ im @ h)
	coeff = (2 / math.sqrt(N*N)) * n * (h.t() @ im @ h) #added

	return coeff


def block_idct(coeff, device=None):
	N = coeff.shape[3]

	n = normalize(N)
	h = harmonics(N)

	if device is not None:
		n = n.to(device)
		h = h.to(device)
	#im = (1 / math.sqrt(2 * N)) * (h @ (n * coeff) @ h.t())
	im = (2/math.sqrt(N*N)) * (h @ (n * coeff) @ h.t()) #added 
	return im

def convert_to_zigzag(dct_blocks,device=None,n_freq=None,block_freq=None):
	"""
	n_freq is number of freq components to include in the compression.
	Rest are set to zero. 
	The order of the freq in each block is obtained by zigzag function.
	n_freq is a tuple which says inclusively which all freqs to include. 
		range: 0 - 63 (inclusive)
	"""
	#changed to RERUN2
	# if n_freq is not None:
	# 	assert n_freq[0] >= 0 and n_freq[-1] <= 63

	size = (dct_blocks.shape[2],dct_blocks.shape[3])

	if n_freq is None:
		#n_freq = (0,63) #include all - adapting this for varying bsizes.
		n_freq = (0, (size[0]*size[1])-1 )
		
		

	ch = dct_blocks.shape[0]
	N = dct_blocks.shape[1] #Number of 8X8 blocks in the image.


	schema = np.arange(size[0]*size[1])
	schema = schema.reshape(size)
	schema = zigzag(schema)
	schema = torch.tensor(schema).long()
	left_out_indices = schema[:n_freq[0]].tolist() + schema[n_freq[1]+1:].tolist()

	if block_freq is not None:
		if type(block_freq) ==int:
			left_out_indices.append(block_freq)
		elif type(block_freq) == list:
			left_out_indices += block_freq

	schema_mask = torch.ones(size).flatten().to(device)
	schema_mask[left_out_indices] = 0 
	schema_mask = schema_mask.reshape(size,size)
	#Create an 8X8 mask with selected indices and multiply it with the blocks.
	

	#How to reorder without loop.
	dct_blocks = dct_blocks * schema_mask

	return dct_blocks

def convert_inverse_zigzag(im_blocks,device=None):
	ch = im_blocks.shape[0]
	N = im_blocks.shape[1] #Number of 8X8 blocks in the image.

	size = (im_blocks.shape[2],im_blocks.shape[3])

	#pdb.set_trace()

	for i in range(N):
		block = im_blocks[:,i,:,:]
		#channelwise.
		for c in range(ch):
			b = block[c].flatten()
			#pdb.set_trace()
			b = inverse_zigzag(b,size[0],size[1],device)
			#b = torch.tensor(b).to(device)

			im_blocks[c,i,:,:] = b
	return im_blocks	


def batch_dct(im, device=None,n_freq=None,block_freq=None,block_size=8,pad=False):

	# if device is not None:
	# 	im = im.to(device)

	"""
		n_freq is inclusive of both frequencies to include.

	"""
	#if pad:
	#	im = pad_input(im,block_size) 
		
	ch = im.shape[1]
	size = (im.shape[2], im.shape[3])

	im_blocks = blockify(im, block_size)
	dct_blocks = block_dct(im_blocks, device=device)

	#Select relevant top frequencies from zigzag orders.
	#We dont rearrage elements in zigzag. We just select the top freq based on zigzag order and
	#then do IDCT. Earlier we were doing zigzag reorder,selection followed by inverse!!

	dct_blocks = convert_to_zigzag(dct_blocks,device=device,n_freq=n_freq,block_freq=block_freq)

	dct = deblockify(dct_blocks, ch, size)

	return dct


def batch_idct(dct, device=None,block_size=8,pad=False):
	ch = dct.shape[1]
	size = (dct.shape[2], dct.shape[3])

	dct_blocks = blockify(dct, block_size)

	#dct_blocks = convert_inverse_zigzag(dct_blocks,device=device)
	#Remove zigzag conversion.

	im_blocks = block_idct(dct_blocks, device=device)
	im = deblockify(im_blocks, ch, size)

	#if pad and out_size is not None:
	#	im = unpad(im,out_size) #padding, unpadding moved to data_process.py

	return im

def unpad(img,out_shape):
	"""
		Undo padding. Since we do constant padding, we can just do a center crop.
	"""
	unpadded_tensor = transforms.CenterCrop(out_shape)(img)
	return unpadded_tensor



def to_ycbcr(x, device=None):
	ycbcr_from_rgb = torch.Tensor([
		0.29900, 0.58700, 0.11400,
		-0.168735892, -0.331264108, 0.50000,
		0.50000, -0.418687589, -0.081312411
	]).view(3, 3).transpose(0, 1).to(device)

	b = torch.Tensor([0, 128, 128]).view(1, 3, 1, 1).to(device)

	x = torch.einsum('cv,bcxy->bvxy', [ycbcr_from_rgb, x])
	x += b

	return x.contiguous()


def to_rgb(x, device=None):
	rgb_from_ycbcr = torch.Tensor([
		1, 0, 1.40200,
		1, -0.344136286, -0.714136286,
		1, 1.77200, 0
	]).view(3, 3).transpose(0, 1).to(device)

	b = torch.Tensor([0, 128, 128]).view(1, 3, 1, 1).to(device)

	x -= b
	x = torch.einsum('cv,bcxy->bvxy', [rgb_from_ycbcr, x])

	return x.contiguous()


def prepare_dct(dct, stats, device=None, type=None):
	ch = []

	for i in range(dct.shape[1]):
		dct_blocks = blockify(dct[:, i:(i+1), :, :], 8)

		t = ['y', 'cb', 'cr'][i] if type is None else type
		dct_blocks = stats.forward(dct_blocks, device=device, type=t)

		ch.append(deblockify(dct_blocks, 1, dct.shape[2:]))

	return torch.cat(ch, dim=1)


def unprepare_dct(dct, stats, device=None, type=None):
	ch = []

	for i in range(dct.shape[1]):
		dct_blocks = blockify(dct[:, i:(i+1), :, :], 8)

		t = ['y', 'cb', 'cr'][i] if type is None else type
		dct_blocks = stats.backward(dct_blocks, device=device, type=t)

		ch.append(deblockify(dct_blocks, 1, dct.shape[2:]))

	return torch.cat(ch, dim=1)


def batch_to_images(dct, stats, device=None, scale_freq=True, crop=None, type=None):
	if scale_freq:
		dct = unprepare_dct(dct, stats, device=device, type=type)

	spatial = batch_idct(dct, device=device) + 128

	if spatial.shape[1] == 3:
		spatial = to_rgb(spatial, device)

	spatial = spatial.clamp(0, 255)
	spatial = spatial / 255

	if crop is not None:
		while len(crop.shape) > 1:
			crop = crop[0]

		cropY = crop[-2]
		cropX = crop[-1]

		spatial = spatial[:, :, :cropY, :cropX]

	return spatial


def images_to_batch(spatial, stats, device=None, type=None):
	spatial *= 255

	if spatial.shape[1] == 3:
		spatial = to_ycbcr(spatial, device)

	spatial -= 128

	frequency = batch_dct(spatial, device=device)
	return prepare_dct(frequency, stats, device=device, type=type)
  


import numpy as np

def zigzag(input):
	#initializing the variables
	#----------------------------------
	h = 0
	v = 0
	vmin = 0
	hmin = 0
	vmax = input.shape[0]
	hmax = input.shape[1]
	#print(vmax ,hmax )
	i = 0
	output = np.zeros(( vmax * hmax))
	#----------------------------------

	while ((v < vmax) and (h < hmax)):
		if ((h + v) % 2) == 0:                 # going up
			if (v == vmin):
				output[i] = input[v, h]        # if we got to the first line
				if (h == hmax):
					v = v + 1
				else:
					h = h + 1                        

				i = i + 1

			elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
				#print(2)
				output[i] = input[v, h] 
				v = v + 1
				i = i + 1

			elif ((v > vmin) and (h < hmax -1 )):    # all other cases
				#print(3)
				output[i] = input[v, h] 
				v = v - 1
				h = h + 1
				i = i + 1        
		else:                                    # going down
			if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
				#print(4)
				output[i] = input[v, h] 
				h = h + 1
				i = i + 1
			elif (h == hmin):                  # if we got to the first column
				#print(5)
				output[i] = input[v, h] 

				if (v == vmax -1):
					h = h + 1
				else:
					v = v + 1

				i = i + 1

			elif ((v < vmax -1) and (h > hmin)):     # all other cases
				#print(6)
				output[i] = input[v, h] 
				v = v + 1
				h = h - 1
				i = i + 1

		if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
			#print(7)        	
			output[i] = input[v, h] 
			break

	return output




# Inverse zigzag scan of a matrix
# Arguments are: a 1-by-m*n array, 
# where m & n are vertical & horizontal sizes of an output matrix.
# Function returns a two-dimensional matrix of defined sizes,
# consisting of input array items gathered by a zigzag method.
#
# Matlab Code:
# Alexey S. Sokolov a.k.a. nICKEL, Moscow, Russia
# June 2007
# alex.nickel@gmail.com


def inverse_zigzag(input, vmax, hmax,device=None):

	# initializing the variables
	#----------------------------------
	h = 0
	v = 0
	vmin = 0
	hmin = 0
	#output = np.zeros((vmax, hmax))
	output = torch.zeros((vmax, hmax)).to(device)
	i = 0
	#----------------------------------
	while ((v < vmax) and (h < hmax)): 
		#print ('v:',v,', h:',h,', i:',i)   	
		if ((h + v) % 2) == 0:                 # going up            
			if (v == vmin):
				#print(1)				
				output[v, h] = input[i]        # if we got to the first line
				if (h == hmax):
					v = v + 1
				else:
					h = h + 1                        

				i = i + 1

			elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
				#print(2)
				output[v, h] = input[i] 
				v = v + 1
				i = i + 1

			elif ((v > vmin) and (h < hmax -1 )):    # all other cases
				#print(3)
				output[v, h] = input[i] 
				v = v - 1
				h = h + 1
				i = i + 1        
		else:                                    # going down
			if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
				#print(4)
				output[v, h] = input[i] 
				h = h + 1
				i = i + 1
			elif (h == hmin):                  # if we got to the first column
				#print(5)
				output[v, h] = input[i] 
				if (v == vmax -1):
					h = h + 1
				else:
					v = v + 1
				i = i + 1
								
			elif((v < vmax -1) and (h > hmin)):     # all other cases
				output[v, h] = input[i] 
				v = v + 1
				h = h - 1
				i = i + 1

		if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
			#print(7)        	
			output[v, h] = input[i] 
			break

	return output


def save_tensor(a,filename):
	a = a.cpu().detach().numpy()
	a = np.squeeze(a)
	try:
		a = np.transpose(a,(1,2,0))
	except:
		pass

	cv2.imwrite(filename,a*255.0)


def pad_input(img,block_size):
	"""
		pad input image tensor so that it is divisible by block size
		Image - NCHW
	"""
	w_pad_size = h_pad_size = 0
	w_padding = h_padding = (0,0)

	if img.shape[3] % block_size != 0:
		w_pad_size = (block_size - img.shape[3] % block_size) % block_size
	
	if img.shape[2] % block_size != 0:
		h_pad_size = (block_size - img.shape[2] % block_size) % block_size

	if w_pad_size == 0 and h_pad_size == 0:
		return img

	if w_pad_size!=0:
		if w_pad_size%2==0:
			w_padding = (w_pad_size//2,w_pad_size//2)
		else:
			w_padding = (w_pad_size//2,w_pad_size//2+1)


	if h_pad_size!=0:
		if h_pad_size%2==0:
			h_padding = (h_pad_size//2,h_pad_size//2)
		else:
			h_padding = (h_pad_size//2,h_pad_size//2+1)

	pad = nn.ConstantPad2d( (w_padding[0],w_padding[1],h_padding[0],h_padding[1]), value=0)

	img = pad(img)
	
	return img



if __name__ == '__main__':


	img = cv2.imread('../data/bunny/f00001.png')
	img = torch.tensor(img).float() / 255.0
	img = img.permute(2,0,1).unsqueeze(0)
	img = img.cuda().float()

	block_size = 64

	img = pad_input(img,block_size)

	dct_rep = batch_dct(img,block_size=block_size,device=img.device)
	rec_img = batch_idct(dct_rep,device=dct_rep.device,block_size=block_size)

	#need to pad for 32 blocks!
	rec_img.clamp_(0,1)

	print(torch.allclose(img,rec_img,3))

	cv2.imwrite('rec.png',rec_img.squeeze().permute(1,2,0).cpu().numpy()*255 )
	cv2.imwrite('og.png',img.squeeze().permute(1,2,0).cpu().numpy()*255 )

	# mean = torch.tensor([0.5,0.5,0.5]).reshape(1,3,1,1)
	# std = torch.tensor([0.5,0.5,0.5]).reshape(1,3,1,1)
	# inv_mean = torch.tensor([-1,-1,-1]).reshape(1,3,1,1)
	# inv_std = torch.tensor([2,2,2]).reshape(1,3,1,1)
	# y_img = to_ycbcr(img)
	# y_img_norm = y_img.sub_(mean).div_(std)
	# y_img_dct = batch_dct(y_img_norm,device='cpu')
	# #Reverse.
	# rgb_img = to_rgb(y_img)


