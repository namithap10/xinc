"""
    File that defines all losses. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
from pytorch_msssim import ssim
from torchmetrics.image.lpip import \
    LearnedPerceptualImagePatchSimilarity as LPIPS
from torchvision import transforms


class DreamSimLoss(nn.Module):
    """
        From https://github.com/ssundaram21/dreamsim
    """
    def __init__(self,cfg):
        super(DreamSimLoss, self).__init__()
        self.cfg = cfg
        self.model,self.preprocess = dreamsim(pretrained=True)

    def forward(self, input, target):
        pass


class vae_loss(nn.Module):

    def __init__(self,cfg):
        super(vae_loss, self).__init__()
        self.cfg = cfg

    def kl_divergence_loss(self,mu, logvar):
        # Calculate KL divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = torch.mean(kl_loss)  # Take the mean across the batch

        return kl_loss


    def vae_loss(self,decoded,data,mu,logvar,lambdas):
        mse = torch.nn.functional.mse_loss(decoded,data)
        kl_loss = self.kl_divergence_loss(mu,logvar)
        loss = mse * lambdas[0] + kl_loss * lambdas[1]
        return loss
        

    def forward(self, predicted, target,mu,logvar,lambdas=[1,1]):
        loss = self.vae_loss(predicted,target,mu,logvar,lambdas=lambdas)
        return loss


class kl_loss(nn.Module):

    def __init__(self,cfg):
        super(kl_loss, self).__init__()
        self.cfg = cfg

    def forward(self, mu, logvar):
        # Calculate KL divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = torch.mean(kl_loss)
        return kl_loss



class clip_loss(torch.nn.Module):
    """
        From https://github.com/cassiePython/CLIPNeRF/blob/main/supplyment/criteria/clip_loss.py
    """

    def __init__(self,cfg,**kwargs):
        super(clip_loss, self).__init__()
        self.cfg = cfg

        self.mean=[0.48145466, 0.4578275, 0.40821073]
        self.std=[0.26862954, 0.26130258, 0.27577711]
        m = transforms.InterpolationMode("bicubic")
        self.resize = transforms.Resize((224, 224),m)

        
        self.clip_model = kwargs['clip_model']
        self.clip_model = self.clip_model.float()
        self.clip_model.eval()
        

    def forward(self, predicted, clip_feat):
        """
            Takes in generated image and corresponding text features to compute clip loss.
            predicted is of shape (B,C,H,W)

            clip_feats: (B,512) 
            normalized. 
        """
        predicted_resize = self.resize(predicted.clamp(0,1))
        predicted_resize = predicted_resize.clamp(0,1)
        predicted_resize = transforms.Normalize(self.mean,self.std)(predicted_resize)
        predicted_clip_img_feats = self.clip_model.encode_image(predicted_resize)

        predicted_clip_img_feats = predicted_clip_img_feats / predicted_clip_img_feats.norm(dim=-1, keepdim=True)

        #normalize clip_feat
        clip_feat = clip_feat / clip_feat.norm(dim=-1, keepdim=True)

        clip_loss = 1 - torch.cosine_similarity(predicted_clip_img_feats,clip_feat,dim=-1).mean()


        return clip_loss        


class clip_loss_contrastive(torch.nn.Module):
    """
        From https://arxiv.org/pdf/2212.08070.pdf
    """

    def __init__(self,args):
        super(clip_loss_contrastive, self).__init__()
        self.device = 'cuda'
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.args = args
        
        self.og_clip_img = self.preprocess(Image.open(args.input)).unsqueeze(0).cuda()

        with torch.no_grad():
            self.og_clip_img_features = self.model.encode_image(self.og_clip_img).float()

        #_,tokenized_og_prompt = helper.clip_text_features(texts,device=self.device)
        texts = helper.prompt_to_template(args.input_prompt)
        self.text_features_og_prompt,tokenized_og_prompt = helper.clip_text_features(texts,device=self.device)
        print('obtained text features for ', args.input_prompt)

        texts = helper.prompt_to_template(args.target_prompt)
        self.text_features_target_prompt,tokenized_target_prompt = helper.clip_text_features(texts,device=self.device)

        print('obtained text features for ', args.target_prompt)

        self.gen_preprocess = tv.transforms.Compose([tv.transforms.Resize((224,224)),tv.transforms.CenterCrop((224,224)),tv.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


    def forward(self, image):
        
        """
            Takes in generated image and tokenized target text. 

            PS: Also does edge loss for now.
        """
        
        generated_img = self.gen_preprocess(image.permute(2,0,1).unsqueeze(0)).cuda()
        generated_img_clip_features = self.model.encode_image(generated_img).float()

        
        #similarity loss between the differences of the image and text features, between generated and og. 
        img_diff = generated_img_clip_features - self.og_clip_img_features
        text_diff = self.text_features_target_prompt - self.text_features_og_prompt

        
        dis_similarity = 1 - nn.CosineSimilarity()(img_diff, text_diff.unsqueeze(0))
        
        return dis_similarity

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
        # self.loss = {'mse': F.mse_loss(input, target)}
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

        #return torch.mean(torch.stack(losses))        
        #return torch.sum(torch.stack(losses))
        
        w_loss = torch.sum(torch.stack(losses))
        if w_lambda is not None:
            w_loss = w_lambda * w_loss

        return w_loss
        

class nerv_loss(nn.Module):
    def __init__(self,cfg,**kwargs):
        super(nerv_loss, self).__init__()
        self.loss = {}
        self.cfg = cfg

    def forward(self, input, target):
        """
            Hard coding it for Full-HD. 
        """
        H = 1088 #hard coded. 
        num_patches_h = H//self.cfg.network.block_size
        input_reshape = rearrange(input,'(nh nw) b c h w -> b c (nh h) (nw w)', nh=num_patches_h)
        target_reshape = rearrange(target,'(nh nw) b c h w -> b c (nh h) (nw w)', nh=num_patches_h)
        loss = 0.7 * torch.mean(torch.abs(input_reshape - target_reshape)) + \
            0.3 * (1 - ssim(input_reshape, target_reshape, data_range=1, size_average=True))

        #loss = 0.7 * torch.mean(torch.abs(input - target)) + 0.3 * (1 - ssim(input, target, data_range=1, size_average=True))
        return loss


class mse_total_variation(nn.Module):
    def __init__(self, cfg,**kwargs):
        super(mse_total_variation, self).__init__()
        self.loss = {}
        self.cfg = cfg
        self.lamb = self.cfg.lambda_param

    def forward(self, input, target):
        self.mse = F.mse_loss(input, target)
        self.tv = total_variation(input,reduction='mean').mean()

        self.loss = self.mse + self.tv * self.lamb
        return self.loss


# def kl_loss(mu, log_var, std=0.01):
#     std = log_var.mul(0.5).exp_()

#     p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std) * std)
#     q = torch.distributions.Normal(mu, std)

#     return 0.05 * torch.distributions.kl_divergence(p, q).mean()


# Define the VGG-based loss function
class content_style_loss(nn.Module):
    def __init__(self,cfg,target=None,lamb=0.8):
        super(content_style_loss, self).__init__()
        self.loss = {}
        self.vgg = torchvision.models.vgg19(pretrained=True).features[:36].cuda().eval() # use vgg19 until layer 36
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalize input to match vgg input
        ])

        
        if target is not None:
            self.target = self.transform(target).detach()
            self.target_features = self.vgg(self.target)
            self.target_gram = self.gram_matrix(self.target_features)
            

        self.lamb = lamb

    def gram_matrix(self,input):
        a, b, c, d = input.size()
        try:
            features = input.view(a * b, c * d)
        except:
            features = input.reshape(a * b, c * d)

        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)


    def forward(self, input_tensor, target=None,lamb=None):
        
        input_tensor = self.transform(input_tensor)
        input_features = self.vgg(input_tensor)
        input_gram = self.gram_matrix(input_features)

        if target is not None:
            self.target = self.transform(target).detach()
            self.target_features = self.vgg(target)
            self.target_gram  = self.gram_matrix(self.target_features)
        
        if lamb is not None:
            self.lamb = lamb
        
        
        # calculate content loss
        self.content_loss = nn.MSELoss()(input_features, self.target_features)
        self.style_loss = nn.MSELoss()(input_gram, self.target_gram)
        self.loss = {'content_style_loss':  self.lamb * self.content_loss + (1 - self.lamb) * self.style_loss}

        return self.loss


if __name__ == '__main__':
    
    import arguments
    from utils import helper

    wloss = weight_loss()
    
    cfg = arguments.get_cfg()

    model = helper.get_model(cfg)
    weight = model.state_dict()

    loss = wloss(model,weight)

    print(loss)

# def weight_loss(self,current_model,base_weights):
#     """
#         Calculate weight loss. 
#         This ensures that the weights are close to the base weights at initialization. 
#     """
#     #return torch.sum(torch.abs(weight_1-weight_2))
#     losses = []
#     loss = nn.MSELoss()
#     for name,param in self.model.named_parameters():
#         if name in base_weights:
#             param.data.requires_grad = True
#             losses.append(loss(param.data,base_weights[name]).requires_grad_(True))
#     return torch.sum(torch.stack(losses))


