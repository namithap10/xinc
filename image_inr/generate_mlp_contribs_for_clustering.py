import argparse
import os
import pickle

import torch
import torchvision.transforms as transforms
from get_mlp_mappings import ComputeMLPContributions
from image_vps_datasets import (single_image_cityscape_vps_dataset,
                                single_image_vipseg_dataset)
from model_all_analysis import ffn, lightning_model
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.utils import save_image
from utils import data_process, helper


def load_cfg(model_ckpt_dir, dataset_name, vidname):
    
    if dataset_name == "cityscapes":
        # Add cityscapes VPS paths
        # vidname = "0005"
        exp_config_path = os.path.join(model_ckpt_dir, 'exp_config.yaml')
        
        cfg = OmegaConf.load(exp_config_path)
        
        cfg.data.cityscapes_vps_root = "../data/cityscapes_vps"
        cfg.data.split = "val"
        cfg.data.panoptic_video_mask_dir = os.path.join(cfg.data.cityscapes_vps_root, cfg.data.split, "panoptic_video")
        cfg.data.panoptic_inst_mask_dir = os.path.join(cfg.data.cityscapes_vps_root, cfg.data.split, "panoptic_inst")
        
        cfg.data.vidname = vidname
        # We will work with the first annotated frame in the given video
        cfg.data.frame_num_in_video = 0
        
        cfg.data.data_path = os.path.join(cfg.data.cityscapes_vps_root, cfg.data.split, "img_all")
        cfg.data.anno_path = '../data/cityscapes_vps/panoptic_gt_val_city_vps.json'

    elif dataset_name == "vipseg":
        # vidname = "26_cblDl5vCZnw" #14_TzgdlZ2CZ3g, 20_o-wWIdQ1H98, 
        exp_config_path = os.path.join(model_ckpt_dir, 'exp_config.yaml')
        
        
        cfg = OmegaConf.load(exp_config_path)
        
        cfg.data.VIPSeg_720P_root = '../data/VIPSeg-Dataset/VIPSeg/VIPSeg_720P'
        cfg.data.panomasks_dir = os.path.join(cfg.data.VIPSeg_720P_root, "panomasks")
        cfg.data.panomasksRGB_dir = os.path.join(cfg.data.VIPSeg_720P_root, "panomasksRGB")
        
        cfg.data.vidname = vidname
        # We will work with the first annotated frame in the given video
        cfg.data.frame_num_in_video = 0
        
        cfg.data.data_path = data_path = os.path.join(cfg.data.VIPSeg_720P_root, "images")
        cfg.data.anno_path = '../data/VIPSeg-Dataset/VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg.json'
        
        # Crop for VIPSeg to match NeRV
        cfg.data.crop=[640,1280]
        
    return cfg


def load_model(cfg):
    save_dir = cfg.logging.checkpoint.logdir
    ckpt_path = helper.find_ckpt(save_dir)
    print(f'Loading checkpoint from {ckpt_path}')

    checkpoint = torch.load(ckpt_path)

    # Load checkpoint into this wrapper model cause that is what is stored in disk :)
    model = lightning_model(cfg, ffn(cfg))
    model.load_state_dict(checkpoint['state_dict'])
    ffn_model = model.model
    
    return ffn_model.cuda()

def get_loader(cfg,dataset_name,val=False):
    # use the dataloader which returns image along with annotations
    if dataset_name == "cityscapes":
        img_dataset = single_image_cityscape_vps_dataset(cfg)
    else:
        img_dataset = single_image_vipseg_dataset(cfg)
    #create torch dataset for one image.
    loader = DataLoader(img_dataset, batch_size=1, shuffle = False ,num_workers=0)
    return loader

def contribs_for_single_config(base_model_dir, out_contribs_dir, dataset_name, vidname, gt_img_path, resized_img_path):
    
    
    model_ckpt_dir = os.path.join(base_model_dir, f'{args.vidname}_framenum_0_128_256')
    
    cfg = load_cfg(model_ckpt_dir, dataset_name, vidname)

    save_dir = cfg.logging.checkpoint.logdir
    ckpt_path = helper.find_ckpt(save_dir)
    checkpoint = torch.load(ckpt_path)
            
    # Load checkpoint into this wrapper model cause that is what is stored in disk :)
    model = lightning_model(cfg, ffn(cfg))
    model.load_state_dict(checkpoint['state_dict'])
    
    ffn_model = model.model
            
    single_image_dataloader = get_loader(cfg, dataset_name, val=True)
    with torch.no_grad():
        batch = next(iter(single_image_dataloader))

    # Same image is used by all seeds
    data = batch['data']
    N,C,H,W = data.shape
    features_shape = batch['features_shape'].squeeze().tolist()

    proc = data_process.DataProcessor(cfg.data, device='cpu')
    x = batch['data']
    coords = proc.get_coordinates(data_shape=features_shape,patch_shape=cfg.data.patch_shape,\
                                    split=cfg.data.coord_split,normalize_range=cfg.data.coord_normalize_range)
    coords = coords.to(x)

    
    out = ffn_model(coords, img=data)
    
    intermediate_results = out["intermediate_results"]

    # Construct contributions dictionary       
    compute_contrib_obj = ComputeMLPContributions(
        # pass the correct current seed's model
        ffn_model, intermediate_results, (H,W)
    )
            
    layer_1_output_contrib, layer_2_output_contrib, layer_3_output_contrib, _, _, _ = compute_contrib_obj.compute_all_layer_mappings()
    
    
    # kernel to pixel contributions
    contribs = {
        1: {
            "layer_1_output_contrib": layer_1_output_contrib,
            "layer_2_output_contrib": layer_2_output_contrib,
            "layer_3_output_contrib": layer_3_output_contrib
        }
    }
    
    # Save dictionary
    contrib_filepath = out_contribs_dir
    os.makedirs(contrib_filepath, exist_ok=True)
    with open(f'{contrib_filepath}/{vidname}_contribs.pkl', 'wb') as handle:
        pickle.dump(contribs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Dumped contribs to {contrib_filepath}/{vidname}_contribs.pkl")
    
    # Load and resize (and crop) GT image using torchvision
    gt_img = read_image(gt_img_path)
    gt_img = gt_img.float() / 255.0
    if dataset_name == "vipseg":
        gt_img = transforms.CenterCrop((640, 1280))(gt_img)
    gt_img = transforms.Resize((128, 256))(gt_img)
    
    # gt_img = transforms.ToTensor()(gt_img)
    
    save_image(gt_img, resized_img_path)

def multi_seeds_for_single_frame(base_model_dir, out_contribs_dir, dataset_name, vidname, gt_img_path, resized_img_path):
    
    models = {}
    seed_list = [1, 10, 20, 30, 40]
    
    for seed in seed_list:
        seed_model_ckpt_dir = os.path.join(base_model_dir, f'seed{seed}_{args.vidname}_framenum_0_128_256')
        
        cfg = load_cfg(seed_model_ckpt_dir, dataset_name, vidname)

        save_dir = cfg.logging.checkpoint.logdir
        ckpt_path = helper.find_ckpt(save_dir)
        checkpoint = torch.load(ckpt_path)
                
        # Load checkpoint into this wrapper model cause that is what is stored in disk :)
        model = lightning_model(cfg, ffn(cfg))
        model.load_state_dict(checkpoint['state_dict'])
    
        models[seed] = model.model
            
    single_image_dataloader = get_loader(cfg, dataset_name, val=True)
    with torch.no_grad():
        batch = next(iter(single_image_dataloader))

    # Same image is used by all seeds
    data = batch['data']
    N,C,H,W = data.shape
    features_shape = batch['features_shape'].squeeze().tolist()

    proc = data_process.DataProcessor(cfg.data, device='cpu')
    x = batch['data']
    coords = proc.get_coordinates(data_shape=features_shape,patch_shape=cfg.data.patch_shape,\
                                    split=cfg.data.coord_split,normalize_range=cfg.data.coord_normalize_range)
    coords = coords.to(x)

    inference_results = {}
    for seed in seed_list:
        with torch.no_grad():
            out = models[seed](coords, img=data)
            
            intermediate_results = out["intermediate_results"]
        
        inference_results[seed] = {
            "data": batch["data"],
            "img_hw": (H,W),
            "intermediate_results": intermediate_results
        }
        
    # Construct contributions dictionary
    contribs = {}
    for seed in seed_list:
        
        intermediate_results = inference_results[seed]["intermediate_results"]
        
        compute_contrib_obj = ComputeMLPContributions(
            # pass the correct current seed's model
            models[seed], intermediate_results, (H,W)
        )
        
        layer_1_output_contrib, layer_2_output_contrib, layer_3_output_contrib, _, _, _ = compute_contrib_obj.compute_all_layer_mappings()
        
        
        # kernel to pixel contributions
        contribs[seed] = {
            "layer_1_output_contrib": layer_1_output_contrib,
            "layer_2_output_contrib": layer_2_output_contrib,
            "layer_3_output_contrib": layer_3_output_contrib
        }
    
    # Save dictionary
    contrib_filepath = out_contribs_dir
    os.makedirs(contrib_filepath, exist_ok=True)
    with open(f'{contrib_filepath}/seeded_{vidname}_contribs.pickle', 'wb') as handle:
        pickle.dump(contribs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Dumped contribs to {contrib_filepath}/seeded_{vidname}_contribs.pickle")
    
    # Load and resize (and crop) GT image using torchvision
    gt_img = read_image(gt_img_path)
    gt_img = gt_img.float() / 255.0
    if dataset_name == "vipseg":
        gt_img = transforms.CenterCrop((640, 1280))(gt_img)
    gt_img = transforms.Resize((128, 256))(gt_img)
    
    save_image(gt_img, resized_img_path)
    
def contribs_for_multi_seeds_epochs(base_model_dir, out_contribs_dir, vidname, epochs_list, seed_list, dataset_name):
    
    models = {}

    for seed in seed_list:
        models[seed] = {}
        for num_epochs in epochs_list:
            # Point to checkpoint for current seed
            seed_model_ckpt_dir = os.path.join(base_model_dir, f'{args.vidname}/seed{str(seed)}_E{num_epochs}_framenum_0_128_256')
            
            # exp_config_path = os.path.join(seed_model_ckpt_dir, 'exp_config.yaml')
            # cfg = OmegaConf.load(exp_config_path)
            
            cfg = load_cfg(seed_model_ckpt_dir, dataset_name, vidname)

            save_dir = cfg.logging.checkpoint.logdir
            ckpt_path = helper.find_ckpt(save_dir)
            checkpoint = torch.load(ckpt_path)
            
            # Load checkpoint into this wrapper model cause that is what is stored in disk :)
            model = lightning_model(cfg, ffn(cfg))
            model.load_state_dict(checkpoint['state_dict'])
            
            ffn_model = model.model
            
            models[seed][num_epochs] = ffn_model
            
    single_image_dataloader = get_loader(cfg, dataset_name, val=True)
    with torch.no_grad():
        batch = next(iter(single_image_dataloader))

    # Same image is used by all seeds
    data = batch['data']
    N,C,H,W = data.shape
    features = batch['features'].squeeze()
    features_shape = batch['features_shape'].squeeze().tolist()
    reshape = True

    proc = data_process.DataProcessor(cfg.data, device='cpu')
    x = batch['data']
    coords = proc.get_coordinates(data_shape=features_shape,patch_shape=cfg.data.patch_shape,\
                                    split=cfg.data.coord_split,normalize_range=cfg.data.coord_normalize_range)
    coords = coords.to(x)

    # Create a dictionary to store the intermediate decoder_results from each seeded model, over time.
    inference_results = {}
    for seed in seed_list:
        inference_results[seed] = {}
        for num_epochs in epochs_list:
            with torch.no_grad():
                out = models[seed][num_epochs](coords, img=data)
                
                intermediate_results = out["intermediate_results"]
            
            inference_results[seed][num_epochs] = {
                "data": batch["data"],
                "img_hw": (H,W),
                "intermediate_results": intermediate_results
            }  
    
    # Construct contributions dictionary
    contribs = {}
    for seed in seed_list:
        contribs[seed] = {}
        for num_epochs in epochs_list:
            contribs[seed][num_epochs] = {}
            
            intermediate_results = inference_results[seed][num_epochs]["intermediate_results"]
            
            compute_contrib_obj = ComputeMLPContributions(
                # pass the correct current seed's model
                models[seed][num_epochs], intermediate_results, (H,W)
            )
            
            layer_1_output_contrib, layer_2_output_contrib, layer_3_output_contrib, _, _, _ = compute_contrib_obj.compute_all_layer_mappings()
            
            
            # kernel to pixel contributions
            contribs[seed][num_epochs] = {
                "layer_1_output_contrib": layer_1_output_contrib,
                "layer_2_output_contrib": layer_2_output_contrib,
                "layer_3_output_contrib": layer_3_output_contrib
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
    
    parser.add_argument('--img-path', default=None, required=False, help="path to bunny or GT image")
    
    parser.add_argument('--single-config', action='store_true', help="whether to run for single config")
    parser.add_argument('--resized-img-path', default=None, required=False, help="path to resized GT image")
    
    parser.add_argument('--multi-seeds-for-single-frame', action='store_true', help="whether to run for multiple seeds but only first frame")
    
    
    args = parser.parse_args()
    print(args.seeds, args.epochs)
    
    if args.multi_seed_epoch:
        contribs_for_multi_seeds_epochs(args.base_model_dir, args.out_contribs_dir, args.vidname, args.epochs, args.seeds, args.dataset_name)
        
    if args.single_config:
        contribs_for_single_config(args.base_model_dir, args.out_contribs_dir, args.dataset_name, args.vidname, args.img_path, args.resized_img_path)

    if args.multi_seeds_for_single_frame:
        multi_seeds_for_single_frame(args.base_model_dir, args.out_contribs_dir, args.dataset_name, args.vidname, args.img_path, args.resized_img_path)