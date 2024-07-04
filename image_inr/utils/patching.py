import torch


class Patcher:
	"""Class to patch and unpatch data.

	Args:
		patch_shape (tuple of ints). Patch size of audio or image. For example
			(200,) for audio, or (64, 64) for an image.
	"""

	def __init__(self, patch_shape):
		assert len(patch_shape) in (1, 2, 3)
		self.patch_shape = patch_shape
		self.patch_dims = len(patch_shape)

	def patch(self, data):
		"""Splits data into patches. If the patch shape doesn't divide the data
		shape, use reflection padding.

		Args:
			data (torch.Tensor): Shape (channels, width) or (channels, height, width) or
				(channels, depth, height, width). Note that there should not be
				a batch dimension.

		Returns:
			Patched data of shape (num_patches, channels, {depth, height,} width)
			and a tuple ({depth, height,} width) specifiying the original shape
			of the data (this is required to reconstruct the data).
		"""
		if self.patch_dims == 1:
			assert data.ndim == 2, "Incorrect data shape for 1d audio."

			# Extract shapes
			channels = data.shape[0]
			spatial_shape = data.shape[1:]

			# Pad data so it can be divided into equally sized patches
			pad_width = get_padding(spatial_shape, self.patch_shape)
			padding = (0, pad_width)
			
			padded = torch.nn.functional.pad(data, padding, mode="reflect")

			# padded has shape (channels, padded_width)
			# Reshape to (num_patches, channels, patch_width)
			return padded.reshape(-1, channels, self.patch_shape[0]), spatial_shape
		elif self.patch_dims == 2:
			assert data.ndim == 3, "Incorrect data shape for images."

			# Extract shapes
			channels = data.shape[0]
			spatial_shape = data.shape[1:]
			patch_height, patch_width = self.patch_shape

			# Pad data so it can be divided into equally sized patches
			pad_height, pad_width = get_padding(spatial_shape, self.patch_shape)
			# Note that padding operates from last to first in terms of dimension
			# i.e. (left, right, top, bottom)            
			padding = (0, pad_width, 0, pad_height)
			try:
				padded = torch.nn.functional.pad(data, padding, mode="reflect")
			except:
				# pytorch bug
				padded = torch.nn.functional.pad(data.unsqueeze(0), padding, mode="reflect").squeeze(0)

			# padded has shape (channels, padded_height, padded_width),
			# unsqueeze this to add a batch dimension (expected by unfold)
			patches = torch.nn.functional.unfold(
				padded.unsqueeze(0),
				stride=self.patch_shape,
				kernel_size=self.patch_shape,
			)
			# patches has shape (1, channels * patch_height * patch_width, num_patches).
			# Reshape to (num_patches, channels, patch_height, patch_width)
			patches = patches.reshape(channels, patch_height, patch_width, -1).permute(
				3, 0, 1, 2
			)
			# Return patches and data shape, so data can be reconstructed from patches
			return patches, spatial_shape

	def unpatch(self, patches, spatial_shape):
		"""
		Args:
			patches (torch.Tensor): Shape (num_patches, channels, {patch_depth,
				patch_height,} patch_width).
			spatial_shape (tuple of ints): Tuple describing spatial dims of
				original unpatched data, i.e. ({depth, height,} width).
		"""
		if self.patch_dims == 1:
			# Calculate padded shape (required to reshape)
			width = spatial_shape[0]
			pad_width = get_padding(spatial_shape, self.patch_shape)
			padded_width = width + pad_width

			# Reshape patches tensor from shape (num_patches, channels, patch_width),
			# to (channels, padded_width) and remove padding to get tensor of shape
			# (channels, width)
			return patches.reshape(-1, padded_width)[:, :width]
		elif self.patch_dims == 2:
			# Calculate padded shape (required by fold function)
			height, width = spatial_shape
			pad_height, pad_width = get_padding(spatial_shape, self.patch_shape)
			padded_shape = (height + pad_height, width + pad_width)

			# Reshape patches tensor from (num_patches, channels, patch_height, patch_width)
			# to (1, channels * patch_height * patch_width, num_patches)
			num_patches, channels, patch_height, patch_width = patches.shape
			patches = patches.permute(1, 2, 3, 0).reshape(1, -1, num_patches)
			# Fold data to return a tensor of shape (1, channels, padded_height, padded_width)
			padded_data = torch.nn.functional.fold(
				patches,
				output_size=padded_shape,
				kernel_size=self.patch_shape,
				stride=self.patch_shape,
			)

			# Remove padding to get tensor of shape (channels, height, width)
			return padded_data[0, :, :height, :width]


def get_padding(spatial_shape, patch_shape):
	"""Returns padding required to make patch_shape divide data_shape into equal
	patches.

	Args:
		spatial_shape (tuple of ints): Shape ({depth, height,} width).
		patch_shape (tuple of ints): Shape ({patch_depth, patch_height,} patch_width).
	"""
	if len(patch_shape) == 1:
		patch_width = patch_shape[0]
		width = spatial_shape[0]
		excess_width = width % patch_width
		pad_width = patch_width - excess_width if excess_width else 0
		return pad_width
	if len(patch_shape) == 2:
		patch_height, patch_width = patch_shape
		height, width = spatial_shape
		excess_height = height % patch_height
		excess_width = width % patch_width
		pad_height = patch_height - excess_height if excess_height else 0
		pad_width = patch_width - excess_width if excess_width else 0
		return pad_height, pad_width


if __name__ == "__main__":

	# 2D tests - images.
	patch_shape = (20, 10)
	data = torch.rand((3, 95, 100))
	patcher = Patcher(patch_shape)
	patched, data_shape = patcher.patch(data)
	print(data.shape)
	print(patched.shape)

	data_unpatched = patcher.unpatch(patched, data_shape)
	print(data_unpatched.shape)

	print((data - data_unpatched).abs().sum())
