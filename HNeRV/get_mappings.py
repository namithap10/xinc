import torch
import torch.nn as nn
import torch.nn.functional as F


class ComputeContributions:

    def __init__(self, model, args, decoder_results, img_out, stride=1, padding=1):
        super(ComputeContributions, self).__init__()

        self.img_out = img_out
        self.decoder_results = decoder_results
        self.model = model
        self.args = args
        self.out_shape = img_out.shape
        self.stride = stride
        self.padding = padding

    def shuffle_channels(self, contrib_map, groups):
        b, c, h, w = contrib_map.size()
        channels_per_grp = c // groups
        contrib_map = contrib_map.view(b, groups, channels_per_grp, h, w)
        contrib_map = torch.transpose(contrib_map, 1, 2).contiguous()
        contrib_map = contrib_map.view(b, -1, h, w)
        return contrib_map

    def apply_box_filter(self, contrib_map, kernel_size, stride, padding):
        # Apply a kernel_size x kernel_size box filter to the contribution map
        # to simulate downstream conv layer's receptive field
        num_groups = contrib_map.size(1)
        box_filter_kernel = (
            torch.ones(num_groups, 1, kernel_size, kernel_size, dtype=torch.float32)
            / kernel_size**2
        )
        conv_layer = nn.Conv2d(
            num_groups,
            num_groups,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            bias=False,
        )
        conv_layer.weight.data = box_filter_kernel
        with torch.no_grad():
            filtered_contrib_map = conv_layer(contrib_map)
        return filtered_contrib_map

    def blk_2_sanity_check(
        self, nerv_blk_2_repeated_contrib_before_gelu, nerv_blk_3_input, pshuffle_stride
    ):

        # Number of sequential (grouped) channels to add - pshuffle_stride**2
        # Compute the new number of channels after grouping
        new_num_channels = (
            nerv_blk_2_repeated_contrib_before_gelu.size(1) // pshuffle_stride**2
        )

        # Reshape the input tensor by grouping channels
        c1, c2, h, w = nerv_blk_2_repeated_contrib_before_gelu.size()
        grouped_tensor = nerv_blk_2_repeated_contrib_before_gelu.view(
            c1, new_num_channels, pshuffle_stride**2, h, w
        )
        # Sum along the channel dimension (axis 2) to add pshuffle_stride**2 channels
        # Only one of the pshuffle_stride**2 channels in a group will have non-zero values
        sanity_check_output = grouped_tensor.sum(dim=2)
        sanity_check_output = sanity_check_output.sum(dim=0)
        sanity_check_output = nn.GELU()(sanity_check_output)

        return (
            torch.isclose(sanity_check_output, nerv_blk_3_input.cpu(), atol=1e-2)
            .all()
            .item()
        )

    def blk_1_sanity_check(
        self, nerv_blk_1_repeated_contrib_before_gelu, nerv_blk_2_input, pshuffle_stride
    ):

        # Number of sequential (grouped) channels to add - pshuffle_stride**2
        # Compute the new number of channels after grouping
        new_num_channels = (
            nerv_blk_1_repeated_contrib_before_gelu.size(1) // pshuffle_stride**2
        )
        
        # Reshape the input tensor by grouping channels
        c1, c2, h, w = nerv_blk_1_repeated_contrib_before_gelu.size()
        grouped_tensor = nerv_blk_1_repeated_contrib_before_gelu.view(
            c1, new_num_channels, pshuffle_stride**2, h, w
        )

        # Sum along the channel dimension (axis 2) to add pshuffle_stride**2 channels
        # Only one of the pshuffle_stride**2 channels in a group will have non-zero values
        sanity_check_output = grouped_tensor.sum(dim=2)
        sanity_check_output = sanity_check_output.sum(dim=0)
        sanity_check_output = nn.GELU()(sanity_check_output)

        return (
            torch.isclose(sanity_check_output, nerv_blk_2_input.cpu(), atol=1e-4)
            .all()
            .item()
        )

    def compute_head_mappings(self):
        head_layer_input = self.decoder_results[-1]
        # Take the first sample/frame (batch size 1)
        head_layer_input = head_layer_input[0, :]
        head_layer_weights = self.model.head_layer.weight

        # Get the dimensions of the head layer weights and inputs/outputs
        output_channels, input_channels, kernel_size, _ = head_layer_weights.shape
        input_channels, height, width = head_layer_input.shape

        # Initialize a map to store the intermediate results (without adding across channels)
        head_layer_output_contrib = torch.zeros(
            (input_channels, output_channels, height, width)
        )

        with torch.no_grad():
            # Iterate through input channels
            for in_ch in range(input_channels):
                # Extract the current input channel and corresponding weights
                current_input_channel = (
                    head_layer_input[in_ch, :, :].unsqueeze(0).unsqueeze(0)
                )
                current_weights = head_layer_weights[:, in_ch, :, :].unsqueeze(1)

                # Create a convolutional layer for the current input channel
                conv_layer = nn.Conv2d(
                    1, output_channels, kernel_size, stride=1, padding=1
                )

                # Set the weights of the convolutional layer
                conv_layer.weight.data = current_weights
                conv_layer.bias.data = self.model.head_layer.bias / input_channels

                # Apply the convolution operation for the current input channel
                # Store the result
                head_layer_output_contrib[in_ch, :, :, :] = conv_layer(
                    current_input_channel
                ).squeeze(0)

        # OutImg bias can be applied

        return head_layer_output_contrib

    def compute_last_nerv_block_mappings(self):
        # Get the input to various layers using decoder_results
        nerv_blk_3_input = self.decoder_results[-2].clone()
        head_layer_input = self.decoder_results[-1].clone()
        # Take the first sample/frame (batch size 1)
        nerv_blk_3_input = nerv_blk_3_input[0, :]
        head_layer_input = head_layer_input[0, :]

        # Obtain the weights of various layers
        nerv_blk_3_conv_weights = self.model.decoder[3].conv.upconv[0].weight
        nerv_blk_3_pshuffle_upscale_factor = (
            self.model.decoder[3].conv.upconv[1].upscale_factor
        )
        head_layer_weights = self.model.head_layer.weight

        nerv_blk_3_conv_out_ch, nerv_block_3_in_ch, kernel_size_nerv_blk_3, _ = (
            nerv_blk_3_conv_weights.shape
        )
        nerv_blk_3_input_size = nerv_blk_3_input.shape

        img_out_ch, nerv_block_3_out_ch, kernel_size_head_layer, _ = (
            head_layer_weights.shape
        )
        head_layer_input_size = head_layer_input.shape

        img_out_shape = self.img_out.shape

        assert (img_out_shape[-2], img_out_shape[-1]) == (
            head_layer_input_size[-2],
            head_layer_input_size[-1],
        )

        # Create maps to store the intermediate results
        # NeRV Block 3 Input -> PShuffle Input
        nerv_blk_conv_pshuffle_contrib = torch.zeros(
            (
                nerv_block_3_in_ch,
                nerv_blk_3_conv_out_ch,
                nerv_blk_3_input_size[-2],
                nerv_blk_3_input_size[-1],
            )
        )
        # NeRV Block 3 Input -> NeRV Block 3 Output
        nerv_blk_3_head_layer_contrib = torch.zeros(
            (
                nerv_block_3_in_ch,
                nerv_block_3_out_ch,
                head_layer_input_size[-2],
                head_layer_input_size[-1],
            )
        )
        # NeRV Block 3 Output -> Head Layer Output
        # We want a mapping from the c_1 x c_2 kernels in NeRV conv layer to
        # every spatial location in the output image
        nerv_blk_3_output_contrib = torch.zeros(
            (
                nerv_block_3_in_ch,
                nerv_blk_3_conv_out_ch,
                img_out_shape[-2],
                img_out_shape[-1],
            )
        )

        # Conv2D
        with torch.no_grad():

            for in_ch in range(nerv_block_3_in_ch):
                current_input_channel = (
                    nerv_blk_3_input[in_ch, :, :].unsqueeze(0).unsqueeze(0)
                )
                current_weights = nerv_blk_3_conv_weights[:, in_ch, :, :].unsqueeze(1)
                conv_layer = nn.Conv2d(
                    1,
                    nerv_blk_3_conv_out_ch,
                    kernel_size_nerv_blk_3,
                    stride=1,
                    padding=1,
                )

                conv_layer.weight.data = current_weights
                conv_layer.bias.data = (
                    self.model.decoder[3].conv.upconv[0].bias / nerv_block_3_in_ch
                )

                # Apply the convolution operation for the current input channel
                # and store the result
                nerv_blk_conv_pshuffle_contrib[in_ch, :, :, :] = conv_layer(
                    current_input_channel
                ).squeeze(0)

        # PixelShuffle
        pshuffle_stride = nerv_blk_3_pshuffle_upscale_factor
        # Simulate rearranging the contribution map when it passes through PixelShuffle
        # This gives contribution to output of Nerv Block 3
        pshuffle_layer = nn.PixelShuffle(pshuffle_stride)
        nerv_blk_3_head_layer_contrib = pshuffle_layer(nerv_blk_conv_pshuffle_contrib)

        pshuffle_stride = nerv_blk_3_pshuffle_upscale_factor
        nerv_blk_3_output_contrib = nerv_blk_3_head_layer_contrib.clone()
        nerv_blk_3_output_contrib = nerv_blk_3_output_contrib.repeat(
            1, pshuffle_stride**2, 1, 1
        )

        # Interleave the repeated feature maps such that sequential channels
        # of group size pshuffle_stride**2 belong to different pshuffle strides
        # but are part of the same feature maps
        channels_per_group = pshuffle_stride**2
        nerv_blk_3_output_contrib = self.shuffle_channels(
            nerv_blk_3_output_contrib, channels_per_group
        )

        # Mask contributions according to PixelShuffle stride map mask
        pshuffle_index_map = torch.zeros(
            nerv_blk_3_output_contrib.shape[2:], dtype=torch.int8
        )
        for stride_1 in range(pshuffle_stride):
            for stride_2 in range(pshuffle_stride):
                pshuffle_index_map[
                    stride_1::pshuffle_stride, stride_2::pshuffle_stride
                ] = (stride_1 * pshuffle_stride + stride_2) + 1

        # Set the non-corresponding elements in each distinct feature map
        # to 0 based on pshuffle stride map
        for i in range(0, pshuffle_stride**2):
            mask = pshuffle_index_map != (i + 1)
            in_channels_per_stride = nerv_blk_3_conv_out_ch // pshuffle_stride**2  # 21

            mask = (
                mask.unsqueeze(0)
                .unsqueeze(0)
                .repeat(nerv_block_3_in_ch, in_channels_per_stride, 1, 1)
            )

            nerv_blk_3_output_contrib[:, i :: pshuffle_stride**2][mask] = 0

        # Apply GELU and Box Filter to get the final contribution map
        # Apply GELU
        nerv_blk_3_output_contrib_before_gelu = nerv_blk_3_output_contrib.clone()
        nerv_blk_3_output_contrib = nn.GELU()(nerv_blk_3_output_contrib)

        # Apply box filter - to each channel independently
        # Purpose is to spatially pool contributions in a 3x3 window
        # which is the size of the window observed by the head layer
        nerv_blk_3_output_contrib = self.apply_box_filter(
            contrib_map=nerv_blk_3_output_contrib,
            kernel_size=kernel_size_head_layer,
            stride=1,
            padding=1,
        )

        return nerv_blk_3_output_contrib, nerv_blk_3_output_contrib_before_gelu

    def forward_contrib_map_through_nerv_blk_3(
        self, nerv_blk_2_contrib, kernel_size_nerv_blk_3
    ):
        """
            This function also simulates passing through head layer
        """
        
        # Apply box filter
        nerv_blk_2_contrib = self.apply_box_filter(
            contrib_map=nerv_blk_2_contrib,
            kernel_size=kernel_size_nerv_blk_3,
            stride=1,
            padding=1,
        )

        # Treat nerv_blk_2_contrib as the input map to nerv_blk_3. (However,
        # the number of channels would be greater)
        blk_3_input = nerv_blk_2_contrib.clone()

        blk_3_pshuffle_upscale_factor = (
            self.model.decoder[3].conv.upconv[1].upscale_factor
        )
        head_layer_weights = self.model.head_layer.weight
        blk_3_input_size = blk_3_input.shape

        kernel_size_head_layer = head_layer_weights.shape[2]
        img_out_shape = self.img_out.shape

        assert (
            blk_3_input_size[-2] * blk_3_pshuffle_upscale_factor,
            blk_3_input_size[-1] * blk_3_pshuffle_upscale_factor,
        ) == (img_out_shape[-2], img_out_shape[-1])

        # Nearest neighbor upsampling in place of PixelShuffle
        blk_3_pshuffle_stride = blk_3_pshuffle_upscale_factor
        blk_2_repeated_contrib = F.interpolate(
            blk_3_input, scale_factor=blk_3_pshuffle_stride, mode="nearest"
        )

        # Apply GELU and Box Filter to obtain final contribution map
        nerv_blk_2_output_contrib = nn.GELU()(blk_2_repeated_contrib)
        nerv_blk_2_output_contrib = self.apply_box_filter(
            contrib_map=nerv_blk_2_output_contrib,
            kernel_size=kernel_size_head_layer,
            stride=1,
            padding=1,
        )

        return nerv_blk_2_output_contrib

    def forward_contrib_map_through_nerv_blk_2(
        self, nerv_blk_1_contrib, nerv_blk_3_input_shape, kernel_size_nerv_blk_2
    ):
        # Apply box filter
        nerv_blk_1_contrib = self.apply_box_filter(
            contrib_map=nerv_blk_1_contrib,
            kernel_size=kernel_size_nerv_blk_2,
            stride=1,
            padding=1,
        )
        blk_2_input = nerv_blk_1_contrib.clone()

        blk_2_pshuffle_upscale_factor = (
            self.model.decoder[2].conv.upconv[1].upscale_factor
        )
        blk_2_input_size = blk_2_input.shape

        blk_3_conv_weights = self.model.decoder[3].conv.upconv[0].weight
        kernel_size_blk_3 = blk_3_conv_weights.shape[2]

        assert (
            blk_2_input_size[-2] * blk_2_pshuffle_upscale_factor,
            blk_2_input_size[-1] * blk_2_pshuffle_upscale_factor,
        ) == (nerv_blk_3_input_shape[-2], nerv_blk_3_input_shape[-1])

        # Simulate passing through Block 2 operations
        # Nearest neighbor upsampling in place of PixelShuffle
        blk_2_pshuffle_stride = blk_2_pshuffle_upscale_factor
        blk_1_repeated_contrib = F.interpolate(
            blk_2_input, scale_factor=blk_2_pshuffle_stride, mode="nearest"
        )

        # Apply GELU, then pass through the downstream layers
        nerv_blk_1_contrib = nn.GELU()(blk_1_repeated_contrib)
        nerv_blk_1_output_contrib = self.forward_contrib_map_through_nerv_blk_3(
            nerv_blk_1_contrib, kernel_size_blk_3
        )

        return nerv_blk_1_output_contrib

    def compute_nerv_block_2_mappings(self):
        # Extract the input to various layers from the dissected decoder_results
        # Take the first frame (batch size 1)
        nerv_blk_2_input = self.decoder_results[-3].clone()[0, :]
        nerv_blk_3_input = self.decoder_results[-2].clone()[0, :]

        # Obtain the weights for current and subsequent layers
        nerv_blk_2_conv_weights = self.model.decoder[2].conv.upconv[0].weight
        nerv_blk_2_pshuffle_upscale_factor = (
            self.model.decoder[2].conv.upconv[1].upscale_factor
        )
        nerv_blk_3_conv_weights = self.model.decoder[3].conv.upconv[0].weight
        nerv_blk_3_pshuffle_upscale_factor = (
            self.model.decoder[3].conv.upconv[1].upscale_factor
        )

        nerv_blk_2_conv_out_ch, nerv_blk_2_in_ch, kernel_size_nerv_blk_2, _ = (
            nerv_blk_2_conv_weights.shape
        )
        nerv_blk_2_input_size = nerv_blk_2_input.shape
        _, nerv_blk_3_in_ch, kernel_size_nerv_blk_3, _ = nerv_blk_3_conv_weights.shape

        nerv_blk_3_input_size = nerv_blk_3_input.shape
        nerv_blk_2_out_ch = nerv_blk_3_in_ch

        img_out_shape = self.img_out.shape

        assert (nerv_blk_2_input_size[-2], nerv_blk_2_input_size[-1]) == (
            img_out_shape[-2]
            // (
                nerv_blk_3_pshuffle_upscale_factor * nerv_blk_2_pshuffle_upscale_factor
            ),
            img_out_shape[-1]
            // (
                nerv_blk_3_pshuffle_upscale_factor * nerv_blk_2_pshuffle_upscale_factor
            ),
        )

        # Create maps to store the intermediate results (without summing across channels)
        # NeRV Block 2 Conv Input -> PShuffle Input (Conv Output)
        nerv_blk_2_conv_pshuffle_contrib = torch.zeros(
            (
                nerv_blk_2_in_ch,
                nerv_blk_2_conv_out_ch,
                nerv_blk_2_input_size[-2],
                nerv_blk_2_input_size[-1],
            )
        )
        # NeRV Block 2 Conv Input -> NeRV Block 2 Output
        nerv_blk_2_blk_3_contrib = torch.zeros(
            (
                nerv_blk_2_in_ch,
                nerv_blk_2_out_ch,
                nerv_blk_3_input_size[-2],
                nerv_blk_3_input_size[-1],
            )
        )
        # We want a mapping from the c_1 x c_2 kernels in NeRV block 2 conv layer
        # to every spatial location in the output image. Thus, we need to pass
        # Block 2 (all kernels) -> Block 3 contributions through the subsequent layer
        # simulation (NeRV Block 3, Head Layer) to get the final map
        nerv_blk_2_repeated_contrib = torch.zeros(
            (
                nerv_blk_2_in_ch,
                nerv_blk_2_conv_out_ch,
                img_out_shape[-2],
                img_out_shape[-1],
            )
        )

        # Conv2D
        with torch.no_grad():
            for in_ch in range(nerv_blk_2_in_ch):
                # Extract the current input channel and weights
                current_input_channel = (
                    nerv_blk_2_input[in_ch, :, :].unsqueeze(0).unsqueeze(0)
                )
                current_weights = nerv_blk_2_conv_weights[:, in_ch, :, :].unsqueeze(1)
                conv_layer = nn.Conv2d(
                    1,
                    nerv_blk_2_conv_out_ch,
                    kernel_size_nerv_blk_2,
                    stride=self.stride,
                    padding=self.stride,
                )
                # Set the weights of the created convolutional layer using the weights extracted
                conv_layer.weight.data = current_weights
                conv_layer.bias.data = (
                    self.model.decoder[2].conv.upconv[0].bias / nerv_blk_2_in_ch
                )

                # Apply the convolution operation for the current input channel
                # and store the result in the contribution map
                nerv_blk_2_conv_pshuffle_contrib[in_ch, :, :, :] = conv_layer(
                    current_input_channel
                ).squeeze(0)

        # PixelShuffle
        pshuffle_stride = nerv_blk_2_pshuffle_upscale_factor
        # Simulate rearranging the contribution map when it passes through PixelShuffle
        # This gives overall contribution to output of NeRV Block 2
        pshuffle_layer = nn.PixelShuffle(pshuffle_stride)
        nerv_blk_2_blk_3_contrib = pshuffle_layer(nerv_blk_2_conv_pshuffle_contrib)

        channels_per_group = pshuffle_stride**2
        nerv_blk_2_repeated_contrib = nerv_blk_2_blk_3_contrib.clone().repeat(
            1, pshuffle_stride**2, 1, 1
        )

        # Interleave the repeated feature maps such that sequential channels
        # of group size pshuffle_stride**2 belong to different pshuffle strides
        # but are part of the same feature maps
        nerv_blk_2_repeated_contrib = self.shuffle_channels(
            nerv_blk_2_repeated_contrib, channels_per_group
        )

        # Mask contributions according to PixelShuffle stride map mask
        pshuffle_index_map = torch.zeros(
            nerv_blk_2_repeated_contrib.shape[2:], dtype=torch.int8
        )
        for stride_1 in range(pshuffle_stride):
            for stride_2 in range(pshuffle_stride):
                pshuffle_index_map[
                    stride_1::pshuffle_stride, stride_2::pshuffle_stride
                ] = (stride_1 * pshuffle_stride + stride_2) + 1

        # Set the non-corresponding elements in each distinct feature map
        # to 0 based on pshuffle stride map
        for i in range(0, pshuffle_stride**2):
            mask = pshuffle_index_map != (i + 1)
            in_channels_per_stride = nerv_blk_2_conv_out_ch // pshuffle_stride**2  # 21

            mask = (
                mask.unsqueeze(0)
                .unsqueeze(0)
                .repeat(nerv_blk_2_in_ch, in_channels_per_stride, 1, 1)
            )
            nerv_blk_2_repeated_contrib[:, i :: pshuffle_stride**2][mask] = 0

        nerv_blk_2_output_contrib_before_gelu = nerv_blk_2_repeated_contrib.clone()
        # Perform sanity check on nerv_blk_2_output_contrib_before_gelu 
        # After summing it group-wise, summing across all in_channels and applying GELU
        # it should be equal to the original nerv_blk_3_input
        assert (
            self.blk_2_sanity_check(
                nerv_blk_2_output_contrib_before_gelu, nerv_blk_3_input, pshuffle_stride
            )
            == True
        ), "Sanity check failed for NeRV Block 2 - Does not match Block 3 Input"

        # Apply GELU before passing through downstream layers (NeRV Block 3, Head Layer)
        # Box filter will be applied in the subsequent layer simulation (NeRV Block 3, Head Layer)
        nerv_blk_2_contrib = nn.GELU()(nerv_blk_2_repeated_contrib)

        nerv_blk_2_output_contrib = self.forward_contrib_map_through_nerv_blk_3(
            nerv_blk_2_contrib, kernel_size_nerv_blk_3
        )

        return nerv_blk_2_output_contrib, nerv_blk_2_output_contrib_before_gelu

    def compute_nerv_block_1_mappings(self):
        # Extract the input to various layers from the dissected decoder_results
        # Take the first frame (batch size 1)
        nerv_blk_1_input = self.decoder_results[-4].clone()[0, :]
        nerv_blk_2_input = self.decoder_results[-3].clone()[0, :]
        nerv_blk_3_input = self.decoder_results[-2].clone()[0, :]

        # Obtain the weights for current and subsequent layers
        nerv_blk_1_conv_weights = self.model.decoder[1].conv.upconv[0].weight
        nerv_blk_1_pshuffle_upscale_factor = (
            self.model.decoder[1].conv.upconv[1].upscale_factor
        )
        nerv_blk_2_conv_weights = self.model.decoder[2].conv.upconv[0].weight
        nerv_blk_1_conv_out_ch, nerv_blk_1_in_ch, kernel_size_nerv_blk_1, _ = (
            nerv_blk_1_conv_weights.shape
        )
        nerv_blk_1_input_size = nerv_blk_1_input.shape

        _, nerv_blk_2_in_ch, kernel_size_nerv_blk_2, _ = nerv_blk_2_conv_weights.shape
        nerv_blk_2_input_size = nerv_blk_2_input.shape
        nerv_blk_1_out_ch = nerv_blk_2_in_ch

        assert (nerv_blk_1_input_size[-2], nerv_blk_1_input_size[-1]) == (
            nerv_blk_2_input_size[-2] // (nerv_blk_1_pshuffle_upscale_factor),
            nerv_blk_2_input_size[-1] // (nerv_blk_1_pshuffle_upscale_factor),
        )

        # Create maps to store the intermediate results (without summing across channels)
        # NeRV Block 1 Conv Input -> PShuffle Input (Conv Output)
        nerv_blk_1_conv_pshuffle_contrib = torch.zeros(
            (
                nerv_blk_1_in_ch,
                nerv_blk_1_conv_out_ch,
                nerv_blk_1_input_size[-2],
                nerv_blk_1_input_size[-1],
            )
        )
        # NeRV Block 1 Conv Input -> NeRV Block 2 Output (Ignoring GELU)
        nerv_blk_1_blk_2_contrib = torch.zeros(
            (
                nerv_blk_1_in_ch,
                nerv_blk_1_out_ch,
                nerv_blk_2_input_size[-2],
                nerv_blk_2_input_size[-1],
            )
        )

        # We want a mapping from the c_1 x c_2 kernels in NeRV block 1 conv layer
        # to every spatial location in the output image. Thus, we need to pass
        # Block 1 -> Block 2 contributions through the subsequent layers
        # (NeRV Block 2, NeRV Block 3, Head Layer) to get the final map
        nerv_blk_1_repeated_contrib = torch.zeros(
            (
                nerv_blk_1_in_ch,
                nerv_blk_1_conv_out_ch,
                nerv_blk_2_input_size[-2],
                nerv_blk_2_input_size[-1],
            )
        )

        # Conv2D
        with torch.no_grad():
            for in_ch in range(nerv_blk_1_in_ch):
                # Extract the current input channel amd weights
                current_input_channel = (
                    nerv_blk_1_input[in_ch, :, :].unsqueeze(0).unsqueeze(0)
                )
                current_weights = nerv_blk_1_conv_weights[:, in_ch, :, :].unsqueeze(1)

                conv_layer = nn.Conv2d(
                    1,
                    nerv_blk_1_conv_out_ch,
                    kernel_size_nerv_blk_1,
                    stride=self.stride,
                    padding=self.stride,
                )
                # Set the weights of the created convolutional layer using the weights extracted
                conv_layer.weight.data = current_weights
                conv_layer.bias.data = (
                    self.model.decoder[1].conv.upconv[0].bias / nerv_blk_1_in_ch
                )

                nerv_blk_1_conv_pshuffle_contrib[in_ch, :, :, :] = conv_layer(
                    current_input_channel
                ).squeeze(0)

        # PixelShuffle
        pshuffle_stride = nerv_blk_1_pshuffle_upscale_factor
        # Simulate rearranging the contribution map when it passes through PixelShuffle
        # This gives overall contribution to output of Nerv Block 1
        pshuffle_layer = nn.PixelShuffle(pshuffle_stride)
        nerv_blk_1_blk_2_contrib = pshuffle_layer(nerv_blk_1_conv_pshuffle_contrib)

        channels_per_group = pshuffle_stride**2
        nerv_blk_1_repeated_contrib = nerv_blk_1_blk_2_contrib.clone().repeat(
            1, pshuffle_stride**2, 1, 1
        )

        # Interleave the repeated feature maps such that sequential channels
        # of group size pshuffle_stride**2 belong to different pshuffle strides
        # but are part of the same feature maps
        nerv_blk_1_repeated_contrib = self.shuffle_channels(
            nerv_blk_1_repeated_contrib, channels_per_group
        )

        # Mask contributions according to PixelShuffle Stride map mask
        pshuffle_index_map = torch.zeros(
            nerv_blk_1_repeated_contrib.shape[2:], dtype=torch.int8
        )
        for stride_1 in range(pshuffle_stride):
            for stride_2 in range(pshuffle_stride):
                pshuffle_index_map[
                    stride_1::pshuffle_stride, stride_2::pshuffle_stride
                ] = (stride_1 * pshuffle_stride + stride_2) + 1

        # Set the non-corresponding elements in each distinct feature map
        # to 0 based on pshuffle stride map
        for i in range(0, pshuffle_stride**2):
            mask = pshuffle_index_map != (i + 1)
            in_channels_per_stride = nerv_blk_1_conv_out_ch // pshuffle_stride**2

            mask = (
                mask.unsqueeze(0)
                .unsqueeze(0)
                .repeat(nerv_blk_1_in_ch, in_channels_per_stride, 1, 1)
            )
            nerv_blk_1_repeated_contrib[:, i :: pshuffle_stride**2][mask] = 0

        nerv_blk_1_output_contrib_before_gelu = nerv_blk_1_repeated_contrib.clone()
        # Perform sanity check on nerv_blk_1_output_contrib_before_gelu 
        # After summing it group-wise, summing across all in_channels and applying GELU
        # it should be equal to the original nerv_blk_2_input
        assert (
            self.blk_1_sanity_check(
                nerv_blk_1_output_contrib_before_gelu, nerv_blk_2_input, pshuffle_stride
            )
            == True
        ), "Sanity check failed for NeRV Block 1 - Does not match Block 2 Input"

        # Apply GELU before passing through downstream layers (NeRV Block 2, NeRV Block 3, Head Layer)
        # Box filter will be applied in the subsequent layer simulation
        nerv_blk_1_contrib = nn.GELU()(nerv_blk_1_repeated_contrib)

        nerv_blk_1_output_contrib = self.forward_contrib_map_through_nerv_blk_2(
            nerv_blk_1_contrib, nerv_blk_3_input.shape[1:], kernel_size_nerv_blk_2
        )

        return nerv_blk_1_output_contrib, nerv_blk_1_output_contrib_before_gelu


if __name__ == "__main__":
    pass
