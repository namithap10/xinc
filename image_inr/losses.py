"""
    File that defines all losses. 
"""
import arguments
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image.lpip import \
    LearnedPerceptualImagePatchSimilarity as LPIPS
from utils import helper


class kl_loss(nn.Module):

    def __init__(self,cfg):
        super(kl_loss, self).__init__()
        self.cfg = cfg

    def forward(self, mu, logvar):
        # Calculate KL divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = torch.mean(kl_loss)
        return kl_loss

class lpips_loss(nn.Module):
    def __init__(self,cfg):
        super(lpips_loss,self).__init__()
        self.cfg = cfg
        self.perceptual_criterion = LPIPS(net_type='vgg').eval().cuda()

    def forward(self, input, target):
        """
            input and target are NCHW between 0 and 1
        """
        #Need to transform to -1 to 1 from [0,1]
        input = input.clamp(0,1)
        perceptual_loss = self.perceptual_criterion(input*2-1,target*2-1)
        return perceptual_loss

class l1(nn.Module):
    def __init__(self,cfg,**kwargs):
        super(l1, self).__init__()
        self.loss = {}
        self.cfg = cfg

    def forward(self, input, target):
        self.loss = F.l1_loss(input, target)
        return self.loss
        
class mse(nn.Module):
    def __init__(self,cfg,**kwargs):
        super(mse, self).__init__()
        self.loss = {}
        self.cfg = cfg

    def forward(self, input, target):
        self.loss = F.mse_loss(input, target)
        return self.loss

class weight_loss(nn.Module):
    def __init__(self,cfg):
        super(weight_loss, self).__init__()
        self.loss = {}
        self.cfg = cfg

    def forward(self,current_model,previous_model_state_dict,w_lambda = None):

        losses = []
        mse_loss = nn.MSELoss()

        for name,param in current_model.named_parameters():
            if name in previous_model_state_dict:
                try:
                    losses.append(mse_loss(param.data,previous_model_state_dict[name]).requires_grad_(True))
                except:
                    continue
        
        w_loss = torch.sum(torch.stack(losses))
        if w_lambda is not None:
            w_loss = w_lambda * w_loss

        return w_loss


if __name__ == '__main__':

    wloss = weight_loss()
    
    cfg = arguments.get_cfg()

    model = helper.get_model(cfg)
    weight = model.state_dict()

    loss = wloss(model,weight)

    print(loss)