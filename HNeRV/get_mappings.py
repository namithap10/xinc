# Do regular torch imports
import torch
import torch.nn as nn


class ComputeContributions():
    
    def __init__(self, model, args, decoder_results, img_out):
        super(ComputeContributions, self).__init__()
        
        self.img_out = img_out
        self.decoder_results = decoder_results
        self.model = model
        self.args = args
        self.out_shape = img_out.shape
        
        
    def compute_head_mappings(self):
        # input
        head_layer_input = self.decoder_results[-1]
        # Take just the first sample/frame for now, ignoring the rest of the batch
        head_layer_input = head_layer_input[0, :]
        
        # weights
        head_layer_weights = self.model.head_layer.weight
        
        # Get the dimensions of the head layer weights and inputs/outputs
        output_channels, input_channels, kernel_size, _ = head_layer_weights.shape
        input_channels, height, width = head_layer_input.shape

        # Initialize a tensor to store the intermediate results (without adding across conv channels)
        head_layer_output_contrib = torch.zeros((input_channels, output_channels, height, width))

        with torch.no_grad():
            # Iterate through input channels
            for in_ch in range(input_channels):
                # Extract the current input channel
                current_input_channel = head_layer_input[in_ch, :, :].unsqueeze(0).unsqueeze(0)
            
                # Extract the weights for the current input channel
                current_weights = head_layer_weights[:, in_ch, :, :].unsqueeze(1)
                
                # Create a convolutional layer for the current input channel
                conv_layer = nn.Conv2d(1, output_channels, kernel_size, stride=1, padding=1)
            
                # Set the weights of the convolutional layer
                conv_layer.weight.data = current_weights
                conv_layer.bias.data = self.model.head_layer.bias / input_channels
            
                # Apply the convolution operation for the current input channel
                # Store the result
                head_layer_output_contrib[in_ch, :, :, :] = conv_layer(current_input_channel).squeeze(0)
        
        return head_layer_output_contrib
    
        # Note: OutImg bias has not been applied to the contribution mapping
        
    def shuffle_channels(self, contrib_map, groups):
        b, c, h, w = contrib_map.size()
        channels_per_grp = c // groups 
        contrib_map = contrib_map.view(b, groups, channels_per_grp, h, w)
        contrib_map = torch.transpose(contrib_map, 1, 2).contiguous()
        contrib_map = contrib_map.view(b, -1, h, w)
        return contrib_map
        
    def compute_last_nerv_block_mappings(self):
        # Get the input to various layers using decoder_results
        nerv_blk_3_input = self.decoder_results[-2].clone()
        head_layer_input = self.decoder_results[-1].clone()
        # Take just the first sample/frame for now, ignoring the rest of the batch
        nerv_blk_3_input = nerv_blk_3_input[0, :]
        head_layer_input = head_layer_input[0, :]
        
        # Obtain the weights of various layers
        nerv_blk_3_conv_weights = self.model.decoder[3].conv.upconv[0].weight
        nerv_blk_3_pshuffle_upscale_factor = self.model.decoder[3].conv.upconv[1].upscale_factor
        head_layer_weights = self.model.head_layer.weight
        
        nerv_blk_3_conv_out_ch, nerv_block_3_in_ch, kernel_size_nerv_blk_3, _ = nerv_blk_3_conv_weights.shape
        nerv_blk_3_input_size = nerv_blk_3_input.shape

        img_out_ch, nerv_block_3_out_ch, kernel_size_head_layer, _ = head_layer_weights.shape
        head_layer_input_size = head_layer_input.shape

        img_out_shape = self.img_out.shape

        assert (img_out_shape[-2], img_out_shape[-1]) == (head_layer_input_size[-2], head_layer_input_size[-1])


        # Create tensors to store the intermediate results (without adding across conv channels)
        # NeRV Block 3 Input -> PShuffle Input (Conv Output)
        nerv_blk_conv_pshuffle_contrib = torch.zeros((nerv_block_3_in_ch, nerv_blk_3_conv_out_ch, nerv_blk_3_input_size[-2], nerv_blk_3_input_size[-1]))
        # NeRV Block 3 Input -> NeRV Block 3 Output          (Ignoring GELU for now)
        nerv_blk_3_head_layer_contrib = torch.zeros((nerv_block_3_in_ch, nerv_block_3_out_ch, head_layer_input_size[-2], head_layer_input_size[-1]))
        # NeRV Block 3 Output -> Head Layer Output (Conv Output)
        # We want a mapping from the c_1 x c_2 kernels in NeRV coonv layer to every spatial location in the output image
        nerv_blk_3_output_contrib = torch.zeros((nerv_block_3_in_ch, nerv_blk_3_conv_out_ch, img_out_shape[-2], img_out_shape[-1]))
        # Head layer to output???
        
        with torch.no_grad():

            # Conv2D
            for in_ch in range(nerv_block_3_in_ch):
                # Extract the current input channel
                current_input_channel = nerv_blk_3_input[in_ch, :, :].unsqueeze(0).unsqueeze(0)
                # Extract the weights for the current input channel
                current_weights = nerv_blk_3_conv_weights[:, in_ch, :, :].unsqueeze(1)
                # Create a convolutional layer for the current input channel
                conv_layer = nn.Conv2d(1, nerv_blk_3_conv_out_ch, kernel_size_nerv_blk_3, stride=1, padding=1)
            
                # Set the weights of the convolutional layer
                conv_layer.weight.data = current_weights
                conv_layer.bias.data = self.model.decoder[3].conv.upconv[0].bias / nerv_block_3_in_ch
            
                # Apply the convolution operation for the current input channel and store the result
                nerv_blk_conv_pshuffle_contrib[in_ch, :, :, :] = conv_layer(current_input_channel).squeeze(0)

        ### PixelShuffle
        pshuffle_stride = nerv_blk_3_pshuffle_upscale_factor
        # Simulate rearranging the contribution map when it passes through PixelShuffle
        # This gives overall contribution to output of Nerv Block 3
        pshuffle_layer = nn.PixelShuffle(pshuffle_stride)
        nerv_blk_3_head_layer_contrib = pshuffle_layer(nerv_blk_conv_pshuffle_contrib)
    
        pshuffle_stride = nerv_blk_3_pshuffle_upscale_factor
        nerv_blk_3_output_contrib = nerv_blk_3_head_layer_contrib.clone()
        nerv_blk_3_output_contrib = nerv_blk_3_output_contrib.repeat(1, pshuffle_stride**2, 1, 1)
        
        ### Interleave the repeated feature maps such that sequential channels of group size pshuffle_stride**2
        # belong to different pshuffle strides but the same feature maps
        channels_per_group = pshuffle_stride**2
        nerv_blk_3_output_contrib = self.shuffle_channels(nerv_blk_3_output_contrib, channels_per_group)
        
        ### Mask contributions according to PixelShuffle Stride map mask
        pshuffle_index_map = torch.zeros(nerv_blk_3_output_contrib.shape[2:], dtype=torch.int8)

        # Add 1s, 2s, 3s etc at corresponding locations
        for stride_1 in range(pshuffle_stride):
            for stride_2 in range(pshuffle_stride):
                pshuffle_index_map[stride_1::pshuffle_stride, stride_2::pshuffle_stride] = (stride_1 * pshuffle_stride + stride_2 ) + 1

        # Set the non-corresponding elements in each distinct feature map to 0 based 
        # on pshuffle stride map
        for i in range(0, pshuffle_stride**2):
            mask = pshuffle_index_map != ( i + 1 )
            in_channels_per_stride = nerv_blk_3_conv_out_ch // pshuffle_stride**2 # 21

            mask = mask.unsqueeze(0).unsqueeze(0).repeat(nerv_block_3_in_ch, in_channels_per_stride, 1, 1)
            
            nerv_blk_3_output_contrib[:, i::pshuffle_stride**2][mask] = 0
            
        ### Apply GELU and Box Filter for final contribution map 
        # Apply GELU
        nerv_blk_3_output_contrib_before_gelu = nerv_blk_3_output_contrib.clone()
        nerv_blk_3_output_contrib = nn.GELU()(nerv_blk_3_output_contrib)

        # Apply Box Filter - to each channel independently
        # Purpose is to spatially pool contributions in a 3x3 window
        # which is the size of the window observed by the head layer
        num_groups = nerv_blk_3_output_contrib.size(1)
        box_filter_kernel = torch.ones(nerv_blk_3_output_contrib.size(1), 1, 3, 3, dtype=torch.float32) / 9
        conv_layer = nn.Conv2d(nerv_blk_3_output_contrib.size(1), nerv_blk_3_output_contrib.size(1), kernel_size=3, 
                            stride=1, padding=1, groups=num_groups, bias=False)
        conv_layer.weight.data = box_filter_kernel 

        with torch.no_grad():
            nerv_blk_3_output_contrib = conv_layer(nerv_blk_3_output_contrib)
            
        return nerv_blk_3_output_contrib, nerv_blk_3_output_contrib_before_gelu

if __name__ == '__main__':
    pass