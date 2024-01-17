import torch 
import torch.nn as nn
import numpy as np
import math
from . import model_utils

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


class SirenLayer(nn.Module):
    """Implements a single SIREN layer.

    Args:
        dim_in (int): Dimension of input.
        dim_out (int): Dimension of output.
        w0 (float):
        c (float): c value from SIREN paper used for weight initialization.
        is_first (bool): Whether this is first layer of model.
        use_bias (bool):
        activation (torch.nn.Module): Activation function. If None, defaults to
            Sine activation.
    """
    def __init__(self, dim_in, dim_out, w0=30., c=6., is_first=False,
                 use_bias=True, activation=None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)

        # Initialize layers following SIREN paper
        w_std = (1 / dim_in) if self.is_first else (np.sqrt(c / dim_in) / w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if use_bias:
            nn.init.uniform_(self.linear.bias, -w_std, w_std)

        self.activation = Sine(w0) if activation is None else activation

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        return out

class siren(nn.Module):
    """SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool):
        final_activation (torch.nn.Module): Activation function.
    """
    
    def __init__(self,cfg,dim_in=2,dim_out=3,use_bias=True,patch_out=None):        

        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.use_bias = use_bias

        self.cfg = cfg

        self.dim_hidden = self.cfg.network.layer_size
        self.num_layers = self.cfg.network.num_layers

        self.w0 = self.cfg.network.w0
        self.w0_initial = self.cfg.network.w0_initial
        
        self.activation = Sine(self.w0) # Defaults to Sine
        self.final_activation = model_utils.get_activation(self.cfg.network.final_act)


        layers = []
        for ind in range(self.num_layers-1):
            is_first = ind == 0
            layer_w0 = self.w0_initial if is_first else self.w0
            layer_dim_in = self.dim_in if is_first else self.dim_hidden

            layers.append(SirenLayer(
                dim_in=layer_dim_in,
                dim_out=self.dim_hidden,
                w0=layer_w0,
                use_bias=self.use_bias,
                is_first=is_first,
                activation=self.activation
            ))

        self.final_activation = nn.Identity() if self.final_activation is None else self.final_activation
        layers.append(SirenLayer(dim_in=self.dim_hidden, dim_out=self.dim_out,
                                use_bias=self.use_bias, activation=self.final_activation))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':
    import easydict,time 
    args = easydict.EasyDict({'network':{'layer_size':512,'num_layers':5,'w0':30.0,'w0_initial':30.0,'final_act':None,\
        'batch_size':1,'fourier_mapping':False}})


    model = siren(dim_in=2, dim_out=3,cfg=args,use_bias=True)
    model = model.cuda()
    input = torch.rand(224*224,2).cuda()
    out = model(input)

    print(out.shape)

