import argparse
import glob
import json
import os
import pickle
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from get_mappings import ComputeContributions
from model_all_analysis import HNeRV
from torch.utils.data import Subset
from torchvision.io import read_image
from torchvision.utils import save_image
from vps_datasets import CityscapesVPSVideoDataSet, VIPSegVideoDataSet


def worker_init_fn(worker_id):
        """
        Re-seed each worker process to preserve reproducibility
        """
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        return

################## split one video into seen/unseen frames ##################
def data_split(img_list, split_num_list, shuffle_data, rand_num=0):
    valid_train_length, total_train_length, total_data_length = split_num_list
    # assert total_train_length < total_data_length
    temp_train_list, temp_val_list = [], []
    if shuffle_data:
        random.Random(rand_num).shuffle(img_list)
    for cur_i, frame_id in enumerate(img_list):
        if (cur_i % total_data_length) < valid_train_length:
            temp_train_list.append(frame_id)
        elif (cur_i % total_data_length) >= total_train_length:
            temp_val_list.append(frame_id)
    return temp_train_list, temp_val_list

def load_model_checkpoint(model, args):
    checkpoint_path = args.weight
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    orig_ckt = checkpoint['state_dict']
    new_ckt={k.replace('blocks.0.',''):v for k,v in orig_ckt.items()} 
    if 'module' in list(orig_ckt.keys())[0] and not hasattr(model, 'module'):
        new_ckt={k.replace('module.',''):v for k,v in new_ckt.items()}
        missing = model.load_state_dict(new_ckt, strict=False)
    elif 'module' not in list(orig_ckt.keys())[0] and hasattr(model, 'module'):
        missing = model.module.load_state_dict(new_ckt, strict=False)
    else:
        missing = model.load_state_dict(new_ckt, strict=False)
    print(missing)
    
    for param in model.parameters():
        param.requires_grad = False
        
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
        
    model.cuda()

    return model

class Args:
    pass

def contribs_for_single_config(base_model_dir, out_contribs_dir, dataset_name, vidname, gt_img_path, resized_img_path):
    
    args = Args()
    
    args.embed = 'pe_1.25_80'
    args.ks = '0_3_3'
    args.num_blks = '1_1'
    args.enc_dim = '64_16'
    args.enc_strds = []
    args.dec_strds = [4, 2, 2]
    args.fc_dim = 37 # 36 # 256 
    args.fc_hw = '8_16'
    args.norm = 'none'
    args.act = 'gelu'
    args.reduce = 1.2
    args.lower_width = 6
    args.conv_type = ['convnext', 'pshuffel']
    args.b = 1
    args.out_bias = 0.0

    args.resize_list = '128_256'
    args.modelsize = 1.0

    args.distributed = False
    args.batchSize = args.b
    args.workers = 4
    args.data_split='1_1_1'
    args.shuffle_data = False
    args.dump_images=False
    
    args.vidname = vidname
    
    if dataset_name == "cityscapes":
        # Add cityscapes VPS paths
        args.cityscapes_vps_root = "../inr_analysis/data/cityscapes_vps"
        args.split = "val"
        args.panoptic_video_mask_dir = os.path.join(args.cityscapes_vps_root, args.split, "panoptic_video")
        args.panoptic_inst_mask_dir = os.path.join(args.cityscapes_vps_root, args.split, "panoptic_inst")
        
        args.data_path = os.path.join(args.cityscapes_vps_root, args.split, "img_all")
        args.anno_path = '../inr_analysis/data/cityscapes_vps/panoptic_gt_val_city_vps.json' # panoptic_gt_val_city_vps_path

        args.crop_list = '-1'

    elif dataset_name == "vipseg":
        args.VIPSeg_720P_root = '../inr_analysis/data/VIPSeg-Dataset/VIPSeg/VIPSeg_720P'
        args.panomasks_dir = os.path.join(args.VIPSeg_720P_root, "panomasks")
        args.panomasksRGB_dir = os.path.join(args.VIPSeg_720P_root, "panomasksRGB")
        args.data_path = os.path.join(args.VIPSeg_720P_root, "images")
        args.anno_path = '../inr_analysis/data/VIPSeg-Dataset/VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg.json'

        args.crop_list = '640_1280'
    
    args.weight = os.path.join(base_model_dir, f'{args.vidname}')
    print("Loading checkpoint from ", args.weight)
    exp_id = glob.glob(args.weight + '/*')[0]
    
    args.weight = exp_id + '/model_best.pth'
    if not os.path.exists(args.weight):
        args.weight = exp_id + '/model_latest.pth'
    
    model = HNeRV(args)
    model = load_model_checkpoint(model, args)

    ############# Create dataloader #############
    if dataset_name == "cityscapes":
        full_dataset = CityscapesVPSVideoDataSet(args)
    elif dataset_name == "vipseg":
        full_dataset = VIPSegVideoDataSet(args)
        
    args.final_size = full_dataset.final_size
    args.full_data_length = len(full_dataset)
    split_num_list = [int(x) for x in args.data_split.split('_')]
    train_ind_list, args.val_ind_list = data_split(list(range(args.full_data_length)), split_num_list, args.shuffle_data, 0)

    #  Make sure the testing dataset is fixed for every run
    train_dataset =  Subset(full_dataset, train_ind_list)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False,#train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, worker_init_fn=worker_init_fn)

    # Get contributions for single image (first frame)
    with torch.no_grad():
        for batch in train_dataloader:
            img_data, norm_idx, img_idx = batch['img'].to('cuda'), batch['norm_idx'].to('cuda'), batch['idx'].to('cuda')
                
            _, _, _, decoder_results, img_out = model(norm_idx)
            
            # Break after finding img_idx=0 (first annotated frame)
            if img_idx == 0:
                break
        
    compute_contrib_obj = ComputeContributions(
        model, args, decoder_results, img_out.detach().clone()[0]
    )
    
    head_layer_output_contrib = compute_contrib_obj.compute_head_mappings()
    nerv_blk_3_output_contrib, _ = compute_contrib_obj.compute_last_nerv_block_mappings() 
        
    # kernel to pixel contributions
    contribs = {
        1: {
            0: {
            "head_layer_output_contrib": head_layer_output_contrib,
            "nerv_blk_3_output_contrib": nerv_blk_3_output_contrib,
        }}
    }
    
    # Save dictionary
    contrib_filepath = out_contribs_dir
    os.makedirs(contrib_filepath, exist_ok=True)
    with open(f'{contrib_filepath}/{vidname}_contribs.pickle', 'wb') as handle:
        pickle.dump(contribs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Dumped contribs to {contrib_filepath}/{vidname}_contribs.pkl")
    
    # Load and resize (and crop) GT image using torchvision
    gt_img = read_image(gt_img_path)
    gt_img = gt_img.float() / 255.0
    if dataset_name == "vipseg":
        gt_img = transforms.CenterCrop((640, 1280))(gt_img)
    gt_img = transforms.Resize((128, 256))(gt_img)
    
    save_image(gt_img, resized_img_path)    

def multi_seeds_for_single_frame(base_model_dir, out_contribs_dir, dataset_name, vidname, gt_img_path, resized_img_path):
    args = Args()
    
    args.embed = 'pe_1.25_80'
    args.ks = '0_3_3'
    args.num_blks = '1_1'
    args.enc_dim = '64_16'
    args.enc_strds = []
    args.dec_strds = [4, 2, 2]
    args.fc_dim = 37 # 36 # 256 
    args.fc_hw = '8_16'
    args.norm = 'none'
    args.act = 'gelu'
    args.reduce = 1.2
    args.lower_width = 6
    args.conv_type = ['convnext', 'pshuffel']
    args.b = 1
    args.out_bias = 0.0

    args.resize_list = '128_256'
    args.modelsize = 1.0

    args.distributed = False
    args.batchSize = args.b
    args.workers = 4
    args.data_split='1_1_1'
    args.shuffle_data = False
    args.dump_images=False
    
    args.vidname = vidname
    
    if dataset_name == "cityscapes":
        # Add cityscapes VPS paths
        args.cityscapes_vps_root = "../inr_analysis/data/cityscapes_vps"
        args.split = "val"
        args.panoptic_video_mask_dir = os.path.join(args.cityscapes_vps_root, args.split, "panoptic_video")
        args.panoptic_inst_mask_dir = os.path.join(args.cityscapes_vps_root, args.split, "panoptic_inst")
        
        args.data_path = os.path.join(args.cityscapes_vps_root, args.split, "img_all")
        args.anno_path = '../inr_analysis/data/cityscapes_vps/panoptic_gt_val_city_vps.json' # panoptic_gt_val_city_vps_path
        
        args.crop_list = '-1'

    elif dataset_name == "vipseg":
        args.VIPSeg_720P_root = '../inr_analysis/data/VIPSeg-Dataset/VIPSeg/VIPSeg_720P'
        args.panomasks_dir = os.path.join(args.VIPSeg_720P_root, "panomasks")
        args.panomasksRGB_dir = os.path.join(args.VIPSeg_720P_root, "panomasksRGB")
        args.data_path = os.path.join(args.VIPSeg_720P_root, "images")
        args.anno_path = '../inr_analysis/data/VIPSeg-Dataset/VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg.json'

        args.crop_list = '640_1280'
                
    #### Load models with different seeds from respective stored weights
    seeds = [1, 10, 20, 30, 40]#, 50, 60, 70, 80, 90]
    models = {}

    for seed in seeds:
        # Point to checkpoint for current seed
        # example base_model_dir = '../inr_analysis/HNeRV/outputs/similarity/'
        args.weight = base_model_dir + f'{args.vidname}/seed{str(seed)}_{args.vidname}_128_256_modelsize1.0/{args.vidname}/'
        exp_id = glob.glob(args.weight + '/*')[0]
        args.weight = exp_id + '/model_best.pth'
        if not os.path.exists(args.weight):
            args.weight = exp_id + '/model_latest.pth'
        
        model = HNeRV(args)
        
        models[seed] = load_model_checkpoint(model, args)

    ############# Create dataloader #############
    if dataset_name == "cityscapes":
        full_dataset = CityscapesVPSVideoDataSet(args)
    elif dataset_name == "vipseg":
        full_dataset = VIPSegVideoDataSet(args)

    args.final_size = full_dataset.final_size
    args.full_data_length = len(full_dataset)
    split_num_list = [int(x) for x in args.data_split.split('_')]
    train_ind_list, args.val_ind_list = data_split(list(range(args.full_data_length)), split_num_list, args.shuffle_data, 0)

    #  Make sure the testing dataset is fixed for every run
    train_dataset =  Subset(full_dataset, train_ind_list)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False,#train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, worker_init_fn=worker_init_fn)

    # Create a dictionary to store the intermediate decoder_results from each seeded model, over time.
    inference_results = {}
    for seed in seeds:
        inference_results[seed] = {}

    with torch.no_grad():
        for batch in train_dataloader:
            img_data, norm_idx, img_idx = batch['img'].to('cuda'), batch['norm_idx'].to('cuda'), batch['idx'].to('cuda')

            # Get output and intermediate decoder results from each (differently seeded) model
            for seed in seeds:
                _, _, _, decoder_results, img_out = models[seed](norm_idx)
                
                # Save all input-output information related to annotated images                    
                inference_results[seed][img_idx.item()] = {
                    "decoder_results": decoder_results,
                    "img_out": img_out,
                }
                
            # Break after finding img_idx=0 (first annotated frame)
            if img_idx == 0:
                break
    
    # Construct contributions dictionary
    contribs_from_seeded_models = {}
    for seed in seeds:
        contribs_from_seeded_models[seed] = {}
        
        img_out = inference_results[seed][0]["img_out"]
        decoder_results = inference_results[seed][0]["decoder_results"]
        
        compute_contrib_obj = ComputeContributions(
            # pass the correct current seed's model
            models[seed], args, decoder_results, img_out.detach().clone()[0]
        )
        
        head_layer_output_contrib = compute_contrib_obj.compute_head_mappings()
        nerv_blk_3_output_contrib, _ = compute_contrib_obj.compute_last_nerv_block_mappings()
        
        
        # kernel to pixel contributions
        contribs_from_seeded_models[seed][0] = {
            "head_layer_output_contrib": head_layer_output_contrib,
            "nerv_blk_3_output_contrib": nerv_blk_3_output_contrib,
        }
    
    # Save dictionary
    # sample out_contribs_dir - output/VIPSeg_models/similarity/contributions/2350_sCLtK1a2GGc
    contrib_filepath = out_contribs_dir #+ f'/{vidname}'
    os.makedirs(contrib_filepath, exist_ok=True)
    with open(f'{contrib_filepath}/seeded_{vidname}_contribs.pickle', 'wb') as handle:
        pickle.dump(contribs_from_seeded_models, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Dumped contribs to {contrib_filepath}/seeded_{vidname}_contribs.pickle")
    
    # Load and resize (and crop) GT image
    gt_img = read_image(gt_img_path)
    gt_img = gt_img.float() / 255.0
    if dataset_name == "vipseg":
        gt_img = transforms.CenterCrop((640, 1280))(gt_img)
    gt_img = transforms.Resize((128, 256))(gt_img)
    
    save_image(gt_img, resized_img_path) 

def contribs_for_multi_seeds_epochs(base_model_dir, out_contribs_dir, vidname, epochs_list, seed_list, dataset_name):
    
    args = Args()
    
    args.embed = 'pe_1.25_80'
    args.ks = '0_3_3'
    args.num_blks = '1_1'
    args.enc_dim = '64_16'
    args.enc_strds = []
    args.dec_strds = [4, 2, 2]
    args.fc_dim = 37 # 36 # 256 
    args.fc_hw = '8_16'
    args.norm = 'none'
    args.act = 'gelu'
    args.reduce = 1.2
    args.lower_width = 6
    args.conv_type = ['convnext', 'pshuffel']
    args.b = 1
    args.out_bias = 0.0

    if dataset_name == "cityscapes":
        args.crop_list = '-1'
    else:
        args.crop_list = '640_1280'
    args.resize_list = '128_256'
    args.modelsize = 1.0

    args.distributed = False
    args.batchSize = args.b
    args.workers = 4
    args.data_split='1_1_1'
    args.shuffle_data = False
    args.dump_images=False
    
    args.vidname = vidname
    
    if dataset_name == "cityscapes":
        # Add cityscapes VPS paths
        args.cityscapes_vps_root = "../inr_analysis/data/cityscapes_vps"
        args.split = "val"
        args.panoptic_video_mask_dir = os.path.join(args.cityscapes_vps_root, args.split, "panoptic_video")
        args.panoptic_inst_mask_dir = os.path.join(args.cityscapes_vps_root, args.split, "panoptic_inst")
        
        args.data_path = os.path.join(args.cityscapes_vps_root, args.split, "img_all")
        args.anno_path = '../inr_analysis/data/cityscapes_vps/panoptic_gt_val_city_vps.json' # panoptic_gt_val_city_vps_path

        with open(args.anno_path, 'r') as f:
            panoptic_gt_val_city_vps = json.load(f)
                    
        panoptic_categories = panoptic_gt_val_city_vps['categories']

    elif dataset_name == "vipseg":
        args.VIPSeg_720P_root = '../inr_analysis/data/VIPSeg-Dataset/VIPSeg/VIPSeg_720P'
        args.panomasks_dir = os.path.join(args.VIPSeg_720P_root, "panomasks")
        args.panomasksRGB_dir = os.path.join(args.VIPSeg_720P_root, "panomasksRGB")
        args.data_path = os.path.join(args.VIPSeg_720P_root, "images")
        args.anno_path = '../inr_analysis/data/VIPSeg-Dataset/VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg.json'

        args.crop_list = '640_1280'

        with open(args.anno_path, 'r') as f:
            panoptic_gt_VIPSeg = json.load(f)
                    
            panoptic_categories = panoptic_gt_VIPSeg['categories']

    models = {}

    for seed in seed_list:
        models[seed] = {}
        for num_epochs in epochs_list:
            # Point to checkpoint for current seed
            args.weight = os.path.join(base_model_dir, f'{args.vidname}/seed{str(seed)}_E{num_epochs}_128_256_modelsize1.0/{args.vidname}')
            # get the name of the folder in the above path
            print(args.weight)
            exp_id = glob.glob(args.weight + '/*')[0]
            args.weight = exp_id + '/model_best.pth'
            
            model = HNeRV(args)
            
            models[seed][num_epochs] = load_model_checkpoint(model, args)

    ############# Create dataloader #############
    if dataset_name == "cityscapes":
        full_dataset = CityscapesVPSVideoDataSet(args)
    elif dataset_name == "vipseg":
        full_dataset = VIPSegVideoDataSet(args)
        
    args.final_size = full_dataset.final_size
    args.full_data_length = len(full_dataset)
    split_num_list = [int(x) for x in args.data_split.split('_')]
    train_ind_list, args.val_ind_list = data_split(list(range(args.full_data_length)), split_num_list, args.shuffle_data, 0)

    #  Make sure the testing dataset is fixed for every run
    train_dataset =  Subset(full_dataset, train_ind_list)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False,#train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, worker_init_fn=worker_init_fn)

    # Create a dictionary to store the intermediate decoder_results from each seeded model, each num_epochs, over time.
    inference_results = {}
    for seed in seed_list:
        inference_results[seed] = {}
        for num_epochs in epochs_list:
            inference_results[seed][num_epochs] = {}    
        
    # If VIPSeg, sample a few image (6-10) indices for analysis
    if dataset_name == "vipseg":
        num_indices = len(train_dataloader) * args.b
        num_samples = 6
        sampled_img_indices = [i * (num_indices - 1) // (num_samples - 1) for i in range(num_samples)]

    with torch.no_grad():
        for batch in train_dataloader:
            img_data, norm_idx, img_idx = batch['img'].to('cuda'), batch['norm_idx'].to('cuda'), batch['idx'].to('cuda')
            
            if dataset_name == "vipseg" and img_idx not in sampled_img_indices:
                continue

            # Get output and intermediate decoder results from each (differently seeded) model
            for seed in seed_list:
                for num_epochs in epochs_list:
                    
                    _, _, _, decoder_results, img_out = models[seed][num_epochs](norm_idx)
                    
                    # Save all input-output information related to annotated images                    
                    inference_results[seed][num_epochs][img_idx.item()] = {
                        "decoder_results": decoder_results,
                        "img_out": img_out,
                    }
    
    # Construct contributions dictionary
    contribs = {}
    for seed in seed_list:
        contribs[seed] = {}
        for num_epochs in epochs_list:
            contribs[seed][num_epochs] = {}
            
            for img_idx, value in inference_results[seed][num_epochs].items():
                img_out = value["img_out"]
                decoder_results = value["decoder_results"]
                
                compute_contrib_obj = ComputeContributions(
                    # pass the correct current seed's model
                    models[seed][num_epochs], args, decoder_results, img_out.detach().clone()[0]
                )
                
                head_layer_output_contrib = compute_contrib_obj.compute_head_mappings()
                nerv_blk_3_output_contrib, _ = compute_contrib_obj.compute_last_nerv_block_mappings()
                
                
                # kernel to pixel contributions
                contribs[seed][num_epochs][img_idx] = {
                    "head_layer_output_contrib": head_layer_output_contrib,
                    "nerv_blk_3_output_contrib": nerv_blk_3_output_contrib,
                }
    
    # Save dictionary
    contrib_filepath = out_contribs_dir + f'/{vidname}'
    os.makedirs(contrib_filepath, exist_ok=True)
    with open(f'{contrib_filepath}/contribs_multi_seed_epoch.pickle', 'wb') as handle:
        pickle.dump(contribs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Dumped contribs to {contrib_filepath}/contribs_multi_seed_epoch.pickle")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model-dir', default=None, required=True, help="base model directory for seeded models")
    parser.add_argument('--out-contribs-dir', default=None, required=True)
    parser.add_argument('--dataset-name', default=None, help="cityscapes or vipseg")
    parser.add_argument('--vidname', default=None, help="cityscpaes or vipseg videoname")
    parser.add_argument('--multi-seed-epoch', action='store_true', help="whether to run for multiple seeds")
    # add params for multiple seeds and epochs
    parser.add_argument('--seeds', default=None, required=False, nargs='+', type=int, help="comma separated list of seeds")
    parser.add_argument('--epochs', default=None, required=False, nargs='+', type=int, help="comma separated list of epochs")
    
    parser.add_argument('--single-config', action='store_true', help="whether to run for a single config")
    parser.add_argument('--resized-img-path', type=str, help="path to dump resized (and cropped) image")
    parser.add_argument('--img-path', type=str, help="path to ground truth frame (first annotated frame)")
    
    parser.add_argument('--multi-seeds-for-single-frame', action='store_true', help="whether to run for multiple seeds but only first frame")
    
    
    args = parser.parse_args()
    print(args.seeds, args.epochs)
    
    if args.multi_seed_epoch:
        contribs_for_multi_seeds_epochs(args.base_model_dir, args.out_contribs_dir, args.vidname, args.epochs, args.seeds, args.dataset_name)
    elif args.single_config:
        contribs_for_single_config(args.base_model_dir, args.out_contribs_dir, args.dataset_name, args.vidname, args.img_path, args.resized_img_path)
    elif args.multi_seeds_for_single_frame:
        multi_seeds_for_single_frame(args.base_model_dir, args.out_contribs_dir, args.dataset_name, args.vidname, args.img_path, args.resized_img_path)