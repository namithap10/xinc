import math

import numpy as np
import torch
import torch.nn as nn


class Reshape_op(torch.nn.Module):
	def __init__(self, shape):
		super().__init__()
		self.shape = shape
		assert len(shape) == 3
		
	def forward(self, x):
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



########################
# Initialization methods

def init_weights_normal(m):
	if hasattr(m, 'weight'):
		nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):	
	if hasattr(m, 'weight'):
		num_input = m.weight.size(-1)
		nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
	if hasattr(m, 'weight'):
		num_input = m.weight.size(-1)
		nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
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