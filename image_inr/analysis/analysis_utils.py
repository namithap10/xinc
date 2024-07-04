import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from utils import helper

sys.path.append('../')

from image_vps_datasets import single_image_cityscape_vps_dataset, single_image_vipseg_dataset
from model_all_analysis import ffn, lightning_model


def load_cfg(model_ckpt_dir, dataset_name, vidname):
    
    if dataset_name == "cityscapes":
        exp_config_path = os.path.join(model_ckpt_dir, 'exp_config.yaml')
        cfg = OmegaConf.load(exp_config_path)
        
        cfg.data.cityscapes_vps_root = "../data/cityscapes_vps"
        cfg.data.split = "val"
        cfg.data.panoptic_video_mask_dir = os.path.join(cfg.data.cityscapes_vps_root, cfg.data.split, "panoptic_video")
        cfg.data.panoptic_inst_mask_dir = os.path.join(cfg.data.cityscapes_vps_root, cfg.data.split, "panoptic_inst")
        
        cfg.data.vidname = vidname
        # Work with the first annotated frame in the given video
        cfg.data.frame_num_in_video = 0
        cfg.data.data_path = os.path.join(cfg.data.cityscapes_vps_root, cfg.data.split, "img_all")
        cfg.data.anno_path = '../data/cityscapes_vps/panoptic_gt_val_city_vps.json'
        
        with open(cfg.data.anno_path, 'r') as f:
            panoptic_gt_val_city_vps = json.load(f)
        panoptic_categories = panoptic_gt_val_city_vps['categories']
        categories = panoptic_categories
        categories.append(
            {'id': -1, 'name': 'other', 'supercategory': '', 'color':None}
        )
        categories_dict = {el['id']: el for el in categories}

    elif dataset_name == "vipseg":
        exp_config_path = os.path.join(model_ckpt_dir, 'exp_config.yaml')
        cfg = OmegaConf.load(exp_config_path)
        
        cfg.data.VIPSeg_720P_root = '../data/VIPSeg-Dataset/VIPSeg/VIPSeg_720P'
        cfg.data.panomasks_dir = os.path.join(cfg.data.VIPSeg_720P_root, "panomasks")
        cfg.data.panomasksRGB_dir = os.path.join(cfg.data.VIPSeg_720P_root, "panomasksRGB")
        
        cfg.data.vidname = vidname
        # Work with the first annotated frame in the given video
        cfg.data.frame_num_in_video = 0
        cfg.data.data_path = data_path = os.path.join(cfg.data.VIPSeg_720P_root, "images")
        cfg.data.anno_path = '../data/VIPSeg-Dataset/VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg.json'
        
        # Crop for VIPSeg to match NeRV
        cfg.data.crop=[640,1280]
        
        with open(cfg.data.anno_path, 'r') as f:
            panoptic_gt_VIPSeg = json.load(f)      
        panoptic_categories = panoptic_gt_VIPSeg['categories']
        categories = panoptic_categories
        categories.append(
            {'id': -1, 'name': 'other', 'supercategory': '', 'color':None}
        )
        categories_dict = {el['id']: el for el in categories}
        
    return cfg, categories_dict

def load_model(cfg):
    save_dir = cfg.logging.checkpoint.logdir
    ckpt_path = helper.find_ckpt(save_dir)
    print(f'Loading checkpoint from {ckpt_path}')

    checkpoint = torch.load(ckpt_path)

    model = lightning_model(cfg, ffn(cfg))
    model.load_state_dict(checkpoint['state_dict'])
    ffn_model = model.model
    
    return ffn_model.cuda()

def get_loader(cfg,dataset_name,val=False):
    if dataset_name == "cityscapes":
        img_dataset = single_image_cityscape_vps_dataset(cfg)
    else:
        img_dataset = single_image_vipseg_dataset(cfg)
    # create torch dataset for one image.
    loader = DataLoader(img_dataset, batch_size=1, shuffle = False ,num_workers=0)
    return loader

def convert_annotations_to_numpy(tensor_annotations):
    annotations = []
    
    for tensor_anno in tensor_annotations:
        annotation = {}

        annotation['id'] = tensor_anno['id'].item()
        annotation['inst_id'] = tensor_anno['inst_id'].item()
        annotation['image_id'] = tensor_anno['image_id'][0] #.item()
        annotation['category_id'] = tensor_anno['category_id'].item()
        annotation['area'] = tensor_anno['area'].item()
        annotation['iscrowd'] = tensor_anno['iscrowd'].item()
        annotation['isthing'] = tensor_anno['isthing'].item()
        bbox = [bbox_tensor.item() for bbox_tensor in tensor_anno['bbox']]
        annotation['bbox'] = bbox
        annotation['binary_mask'] = tensor_anno['binary_mask'].numpy()
        
        annotations.append(annotation)

    return annotations

def add_other_annotation(annotations):

    # Create a mask that indicates whether a location contains at least one instance
    object_region_mask = None
    for ann in annotations:
        binary_mask = ann['binary_mask'].squeeze()
        if object_region_mask is None:
            object_region_mask = binary_mask.copy()
        else:
            object_region_mask += binary_mask

    # Binarize
    object_region_mask = object_region_mask != 0
    
    # Create an annotation denoting "other" for regions that have no objects
    annotations.append({
        "id": -1,
        "inst_id": -1,
        "bbox": compute_bbox(object_region_mask),
        "area": object_region_mask.sum(),
        "binary_mask": object_region_mask,
        'iscrowd': 0,
        'isthing': 0,
        'category_id': -1,
        'image_id': annotations[0]['image_id']
    })
    return annotations

def compute_bbox(binary_mask):
    (rows, cols) = np.where(binary_mask > 0)
    x_min, x_max, y_min, y_max = min(cols), max(cols), min(rows), max(rows)
    # Create the bbox in COCO format [x, y, width, height]
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    bbox = [x_min, y_min, width, height]
    return bbox