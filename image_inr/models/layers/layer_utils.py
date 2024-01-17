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
	#if type(m) == BatchLinear or type(m) == nn.Linear:
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
	if hasattr(m, 'weight'):
		nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):	
	if hasattr(m, 'weight'):
		num_input = m.weight.size(-1)
		nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
	#if type(m) == BatchLinear or type(m) == nn.Linear:
	if hasattr(m, 'weight'):
		num_input = m.weight.size(-1)
		nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
	#if type(m) == BatchLinear or type(m) == nn.Linear:
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