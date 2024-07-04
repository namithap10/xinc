import easydict
import numpy as np
import torch
import torch.nn as nn
from models import BaseModel

from . import model_utils


class MLPLayer(nn.Module):
    """Implements a single SIREN layer.

    Args:
        dim_in (int): Dimension of input.
        dim_out (int): Dimension of output.
        w0 (float):
        c (float): c value from SIREN paper used for weight initialization.
        is_first (bool): Whether this is first layer of model.
        use_bias (bool):
        activation (torch.nn.Module): Activation function. If None, defaults to
            ReLU activation.
    """
    def __init__(self, dim_in, dim_out,is_first=False,
                 use_bias=True, activation=None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)

        self.activation = nn.ReLU() if activation is None else activation

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        return out


class ffn(BaseModel):
    """
        FFN model - Feature fourier networks. 
    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        use_bias (bool):
        final_activation (torch.nn.Module): Activation function.
    """
    
    def __init__(self,cfg):        
        super().__init__(cfg)

        self.activation = nn.ReLU()
        self.final_activation = model_utils.get_activation(self.cfg.network.final_act)
        self.use_bias = self.cfg.network.use_bias
        
        layers = []
        for ind in range(self.num_layers-1):
            is_first = ind == 0
            
            layer_dim_in = self.in_features if is_first else self.dim_hidden

            layers.append(MLPLayer(
                dim_in=layer_dim_in,
                dim_out=self.dim_hidden,
                use_bias=self.use_bias,
                is_first=is_first,
                activation=self.activation
            ))

        self.final_activation = nn.Identity() if self.final_activation is None else self.final_activation
        
        if cfg.data.patch_shape is not None:
            self.dim_out = 3 * np.prod(cfg.data.patch_shape)
        
        layers.append(MLPLayer(dim_in=self.dim_hidden, dim_out=self.dim_out,
                                use_bias=self.use_bias, activation=self.final_activation))

        self.net = nn.Sequential(*layers)


    def forward(self, x, **kwargs):
        mapped_x = self.positional_encoding(x)
        predicted = self.net(mapped_x)
        outputs = {'predicted':predicted}
        return outputs
    

if __name__ == '__main__':

    #test fourier features. 
    args = easydict.EasyDict({'network':{'layer_size':512,'num_layers':5,'w0':30.0,'w0_initial':30.0,'final_act':None,\
        'batch_size':1,'fourier_mapping':True,'fourier_noise_scale':1.0}})
    img_shape = 224
    
    #Nx2
    x = torch.randn((224*224,2)).cuda()

    model = ffn(dim_in=2, dim_out=3,cfg=args,use_bias=True)
    model = model.cuda()
    out = model(x)

    print(out.shape)
    