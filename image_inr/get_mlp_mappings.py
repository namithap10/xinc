import einops
import torch
import torch.nn as nn


class ComputeMLPContributions():
    
    def __init__(self, ffn_model, intermediate_results, img_hw):
        super(ComputeMLPContributions, self).__init__()
        
        self.intermediate_results = intermediate_results
        self.ffn_model = ffn_model
        self.img_hw = img_hw
        
    def compute_all_layer_mappings(self):
        
        H, W = self.img_hw
        layer_1 = self.ffn_model.net[0]
        layer_1_weight, layer_1_bias = layer_1.linear.weight, layer_1.linear.bias
        layer_2 = self.ffn_model.net[1]
        layer_2_weight, layer_2_bias = layer_2.linear.weight, layer_2.linear.bias
        layer_3 = self.ffn_model.net[2]
        layer_3_weight, layer_3_bias = layer_3.linear.weight, layer_3.linear.bias

        layer_1_input, layer_2_input, layer_3_input = self.intermediate_results[-4], self.intermediate_results[-3], self.intermediate_results[-2]
        
        in_feats = [layer_1_weight.shape[1], layer_2_weight.shape[1], layer_3_weight.shape[1]]
        out_feats = layer_3_weight.shape[0]
                
        layer_1_output_contrib = torch.zeros(in_feats[0], H*W) # PE_dim (e.g. 26) x hw
        layer_1_output_contrib_to_layer2infeat = torch.zeros(in_feats[0], in_feats[1], H*W)
        
        layer_2_output_contrib = torch.zeros(in_feats[1], H*W) # 128 x hw
        layer_2_output_contrib_to_layer3infeat = torch.zeros(in_feats[1], in_feats[2], H*W) # 128 x 128 x hw
        
        layer_3_output_contrib = torch.zeros(in_feats[2], H*W) # 128 x hw
        layer_3_output_contrib_to_rgb = torch.zeros(in_feats[2], out_feats, H*W) # 128 x 3 x hw
        
        with torch.no_grad():
            # Layer 3
            for in_feat in range(layer_3_output_contrib.shape[0]):
                # hw x 1 matmul with 1 x 3
                contrib = torch.mm(layer_3_input[:, in_feat].unsqueeze(1), layer_3_weight[:, in_feat].unsqueeze(1).T) # hw x 3
                # Add bias
                contrib += layer_3_bias / layer_3_output_contrib.shape[0]
                
                layer_3_output_contrib_to_rgb[in_feat,:,:] = contrib.T 
                layer_3_output_contrib[in_feat,:] = torch.sum(contrib, dim=-1)
                    
            # Layer 2
            for in_feat in range(layer_2_output_contrib.shape[0]):
                # hw x 1 matmul with 1 x out_feat
                contrib = torch.mm(layer_2_input[:, in_feat].unsqueeze(1), layer_2_weight[:, in_feat].unsqueeze(1).T)
                # Add bias
                contrib += layer_2_bias / layer_2_output_contrib.shape[0]
        
                layer_2_output_contrib_to_layer3infeat[in_feat,:,:] = contrib.T
                layer_2_output_contrib[in_feat,:] = torch.sum(contrib, dim=-1)
        
            layer_2_output_contrib = nn.ReLU()(layer_2_output_contrib)
        
            # Layer 1
            for in_feat in range(layer_1_output_contrib.shape[0]):
                # hw x 1 matmul with 1 x out
                contrib = torch.mm(layer_1_input[:, in_feat].unsqueeze(1), layer_1_weight[:, in_feat].unsqueeze(1).T)
                # Add bias
                contrib += layer_1_bias / layer_1_output_contrib.shape[0]
        
                layer_1_output_contrib_to_layer2infeat[in_feat,:,:] = contrib.T
                layer_1_output_contrib[in_feat,:] = torch.sum(contrib, dim=-1)
        
            layer_1_output_contrib = nn.ReLU()(layer_1_output_contrib)

        # Do all the contrib reshaping at the end
        layer_1_output_contrib = einops.rearrange(layer_1_output_contrib, 'f (h w) -> f h w', h=H, w=W)
        layer_2_output_contrib = einops.rearrange(layer_2_output_contrib, 'f (h w) -> f h w', h=H, w=W)
        layer_3_output_contrib = einops.rearrange(layer_3_output_contrib, 'f (h w) -> f h w', h=H, w=W)
        layer_3_output_contrib_to_rgb = einops.rearrange(layer_3_output_contrib_to_rgb, 'f c (h w) -> f c h w', h=H, w=W)

        return layer_1_output_contrib, layer_2_output_contrib, layer_3_output_contrib, layer_1_output_contrib_to_layer2infeat, layer_2_output_contrib_to_layer3infeat, layer_3_output_contrib_to_rgb 