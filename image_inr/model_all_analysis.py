from pydoc import locate

import lightning as L
import torch
import torch.nn as nn
from models import BaseModel, model_utils
from models.ffn import MLPLayer
from utils import data_process, helper


class lightning_model(L.LightningModule):
    def __init__(self,cfg,ffn_model):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        
        # model_name = self.cfg.network.model_name
        # self.model_class = locate('models.'+ model_name + '.'+ model_name)
        # Modify lightning_model to take preinitialized ffn_model
        self.model = ffn_model
        self.coords = None
        
        self.proc = data_process.DataProcessor(self.cfg.data,device='cpu') #dataloading on cpu
        self.size = self.cfg.network.sidelength
        self.load_loss()

        self.log_dir = self.cfg.logging.checkpoint.logdir

        self.img_save_folder = self.cfg.logging.checkpoint.logdir+'/' + 'reconstructions/'
        helper.make_dir(self.img_save_folder)


    def load_loss(self):
        loss_cfg = self.cfg.trainer.losses
        loss_list = loss_cfg.loss_list
        self.loss_functions = {}
        self.loss_weights = {}

        for k in loss_list:
            if 'vae' in k:
                loss_list.remove(k)
                loss_list += ['mse','kl_loss']
                break
        
        for idx,loss in enumerate(loss_list):
            loss_name = loss.lower()
            loss_class = locate('losses.'+loss_name)

            if 'clip' in loss_name:
                self.clip_model,_ = helper.load_clip_model()
                loss_func = loss_class(self.cfg,**{'clip_model':self.clip_model})
            else:
                loss_func = loss_class(self.cfg)
            self.loss_weights[loss_name] = loss_cfg.loss_weights[idx]
            self.loss_functions[loss_name] = loss_func

            print('Using loss function : ',loss_name,' with weight: ',self.loss_weights[loss_name])        

    def apply_loss(self,x,pred,**kwargs):

        loss = 0
        for k,v in self.loss_functions.items():            
            loss_weight = self.loss_weights[k]
            if 'clip' in k:
                loss += v(pred,clip_feat=kwargs['clip_feat']) * loss_weight
            else:
                loss += v(x,pred) * loss_weight
        return loss


    def forward(self,batch,kwargs,reshape=False):
        data = batch['data']
        features = batch['features'].squeeze()
        features_shape = batch['features_shape'].squeeze().tolist()

        #Image not used in most architectures. check model file.
        out = self.model(self.coords,img=data,**kwargs)
        pred = out['predicted']
        intermediate_results = out['intermediate_results']
        
        loss = self.apply_loss(features,pred.squeeze(),**kwargs)
        psnr = helper.get_clamped_psnr(features,pred)
        
        if reshape:
            pred = self.proc.process_outputs(pred,input_img_shape=batch['data_shape'].squeeze().tolist(),\
                                             features_shape=features_shape,\
                                            patch_shape=self.cfg.data.patch_shape)

        return loss,psnr,pred,intermediate_results
    

    def training_step(self, batch, batch_idx):
        x = batch['data']
        features_shape = batch['features_shape'].squeeze().tolist()
        
        if 'clip_feat' in batch:
            clip_feat = batch['clip_feat']
            clip_feat = clip_feat.to(x)
            kwargs = {'clip_feat':clip_feat}
        else:
            kwargs = {}

        
        if batch_idx == 0 or self.coords is None:
            self.coords = self.proc.get_coordinates(data_shape=features_shape,patch_shape=self.cfg.data.patch_shape,\
                                                    split=self.cfg.data.coord_split,normalize_range=self.cfg.data.coord_normalize_range)
            self.coords = self.coords.to(x)

        loss,psnr,pred,intermediate_results = self.forward(batch,kwargs)

        self.log_dict({'train_loss':loss,'train_psnr':psnr},\
                      on_step=False,on_epoch=True,prog_bar=True,logger=True,sync_dist=True)

        return loss
    
    def validation_step(self,batch,batch_idx):
        
        x = batch['data']
        features_shape = batch['features_shape'].squeeze().tolist()

        if 'clip_feat' in batch:
            clip_feat = batch['clip_feat']
            kwargs = {'clip_feat':clip_feat}
        else:
            kwargs = {}

        if batch_idx == 0 or self.coords is None:
            
            self.coords = self.proc.get_coordinates(data_shape=features_shape,patch_shape=self.cfg.data.patch_shape,\
                                                    split=self.cfg.data.coord_split,normalize_range=self.cfg.data.coord_normalize_range)
            self.coords = self.coords.to(x)
        
        
        if ((self.current_epoch+1) % self.cfg.logging.checkpoint.save_every ==0) and (batch_idx==0):
            loss,psnr,pred,intermediate_results = self.forward(batch,kwargs,reshape=True)
            
            pred = pred.squeeze().clamp(0,1)
            pred_img = helper.tensor_to_cv(pred)
            helper.save_tensor_img(pred,self.img_save_folder+'/reconstruction_epoch_'+str(self.current_epoch)+'.png')

            if self.cfg.logging.wandb.enabled:
                self.logger.log_image(key='predictions_val',images=[pred_img],step=self.current_epoch)
        else:
            loss,psnr,pred,intermediate_results = self.forward(batch,kwargs,reshape=False)
        
        self.log_dict({'val_loss':loss,'val_psnr':psnr},\
                        on_step=False,on_epoch=True,prog_bar=True,logger=True,sync_dist=True)

    def on_fit_end(self):
        if self.current_epoch < self.cfg.trainer.eval_every:
            print('return')
            return

        best_score = self.trainer.checkpoint_callback.best_model_score.item()
        params = sum(p.numel() for p in self.parameters())
        info = {'best_psnr':best_score,'params':params}
        helper.dump_to_json(info,self.log_dir+'/info.json')


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.trainer.lr,betas=self.cfg.trainer.optimizer.betas)
        return optimizer


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
        layers.append(MLPLayer(dim_in=self.dim_hidden, dim_out=self.dim_out,
                                use_bias=self.use_bias, activation=self.final_activation))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x, **kwargs):
        intermediate_results = []
        intermediate_results.append(x)
        
        mapped_x = self.positional_encoding(x)
        intermediate_results.append(mapped_x)
        
        predicted = self.net(mapped_x)
        # We want to break down the forward pass into layers
        # The last intermediate result is just the output of the network
        intermediate_x = mapped_x.clone()
        for layer in self.net:
            intermediate_x = layer(intermediate_x)
            intermediate_results.append(intermediate_x)
        outputs = {'predicted': predicted, 'intermediate_results': intermediate_results}
        return outputs    