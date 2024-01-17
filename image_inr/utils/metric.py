from email.policy import default
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from enum import Enum
from collections import defaultdict

from . import helper

class Metric():
    def __init__(self):
        #keys=['fp_psnr','encoding_time','quantized_psnr','storage_optimize_time']
        self.results = defaultdict(dict)

    def add_metric(self,metric_name,summary_type):
        self.results[metric_name] = AverageMeter(metric_name,summary_type=summary_type)
    

    def update(self,res,epoch):
        for key in res:
            self.results[key].update(res[key])

    def get_results(self,vals=False,epoch=None):
        

        if vals:
            res = {k:v.values for k,v in self.results.items()}
        else:                
            res = {k:v.values[-1] if v.values!=[] else 0  for k,v in self.results.items()}

        res['epoch'] = epoch
        return res

    def save_to_json(self,filename,epoch=None):
        self.compute_results = self.get_results(epoch=epoch)
        helper.dump_to_json(self.compute_results,filename)

    def total_time(self):
        total_time = self.results['encoding_time'].sum
        return total_time

    def plot(self,filename='metrics.png',epoch=None):
        self.plot_results = self.get_results(epoch=epoch,vals=True)
        
        #we plot psnr for each frame for given epoch.

        #we need epochwise numbers for each. 
        for key in self.plot_results:
            if key=='epoch':
                continue
            title = (key + ' ') + ' vs Epoch'
            if self.plot_results[key]!=[]:
                plt.title(title)
                plt.plot(self.plot_results[key],label=key,marker='o')
                plt.legend()
                plt.savefig(filename.replace('.png',f'_{key}.png'))
                plt.close('all')


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.values = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.values.append(val)

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def all_gather(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")        
        
        tensor = torch.tensor(self.values,dtype=torch.float32,device=device)
        out = helper.all_gather([tensor])[0]
        return out


    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
            val = None
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
            val = self.avg 
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
            val = self.sum
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
            val = self.count
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        print(fmtstr.format(**self.__dict__))

        return val
        #return fmtstr.format(**self.__dict__)


class quant_bit_tracker():
    def __init__(self):
        self.psnr_tracker = defaultdict(list)
        self.comp_size_tracker = defaultdict(list)
        
        
    def get_average(self):
        for key in self.psnr_tracker:
            self.psnr_tracker[key] = np.mean(self.psnr_tracker[key])
        
        for key in self.comp_size_tracker:
            self.comp_size_tracker[key] = np.sum(self.comp_size_tracker[key])
       
        self.psnr_vals = list(self.psnr_tracker.values())
        self.comp_size_vals = list(self.comp_size_tracker.values())
    
        self.psnr_mean = torch.tensor(self.psnr_vals).tolist()
        self.comp_size = torch.tensor(self.comp_size_vals).tolist()

        self.bit_list = list(self.psnr_tracker.keys())
        self.bit_list.sort()
        

    def all_reduce(self):

        """
            All mean and sum happen for Each bit. 
            Hence need to maintain a tensor of size ``q_bits". 
        """

        self.world_size = dist.get_world_size()
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        self.psnr_mean = torch.tensor(self.psnr_vals,dtype=torch.float32,device=device)
        self.comp_size = torch.tensor(self.comp_size_vals,dtype=torch.float32,device=device)
        
        dist.all_reduce(self.comp_size, dist.ReduceOp.SUM, async_op=False)
        dist.all_reduce(self.psnr_mean, dist.ReduceOp.SUM, async_op=False)

        #Mean acrtoss all workers, for each q_bit.
        self.psnr_mean = (self.psnr_mean / self.world_size).tolist()

        #No average for this. We need total.
        self.comp_size = self.comp_size.tolist()




    def all_gather(self):
        pass

    def save_to_json(self,filename):
        helper.dump_to_json(self.results,filename)

    def plot(self,filename='metrics.png'):
        plt.plot(self.results['bit'],self.results['psnr'],label='psnr',marker='o')
        plt.title('psnr')
        plt.legend()
        plt.savefig(filename.replace('.png',f'_psnr.png'))
        plt.close('all')

