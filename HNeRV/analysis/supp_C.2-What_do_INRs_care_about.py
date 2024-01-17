# NeRV imports
import json
import os
import pickle
import random
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from get_mappings import ComputeContributions
from model_all_analysis import HNeRV
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
from sklearn.cluster import KMeans
from torch.utils.data import Subset
from vps_datasets import (CityscapesVPSVideoDataSet, VIPSegVideoDataSet)

warnings.filterwarnings("ignore")

class Args:
    pass

def load_model_args():
    args = Args()
    args.embed = 'pe_1.25_80'
    args.ks = '0_3_3'
    args.num_blks = '1_1'
    args.enc_dim = '64_16'
    args.enc_strds, args.dec_strds = [], [4, 2, 2]
    args.fc_dim = 37 
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
    
    args.dump_images=False
    
    return args

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
    print("=> loaded checkpoint '{}' (epoch {})".format(args.weight, checkpoint['epoch'])) 

    for param in model.parameters():
        param.requires_grad = False
        
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
        
    return model.cuda()

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

# Annotations stuff
def convert_tensor_annotations_to_numpy(tensor_annotations):
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

        # Convert 'bbox' back to regular format
        bbox = [bbox_tensor.item() for bbox_tensor in tensor_anno['bbox']]
        annotation['bbox'] = bbox

        annotation['binary_mask'] = tensor_anno['binary_mask'].numpy()
        
        annotations.append(annotation)

    return annotations

def add_other_annotation(annotations):

    # Create a mask that will indicate whether a location contains at least one instance
    object_region_mask = None
    for ann in annotations:
        binary_mask = ann['binary_mask'].squeeze()
        if object_region_mask is None:
            # If the object_region_mask is None, initialize it to current binary_mask otherwise aggregate it
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

def load_dataset_specific_args(args, dataset_name, vidname):
    args.batchSize = args.b
    args.distributed = False
    args.workers = 4
    args.data_split='1_1_1'
    args.shuffle_data = False

    if dataset_name == "cityscapes":

        # Add cityscapes VPS paths
        args.cityscapes_vps_root = "../data/cityscapes_vps"
        args.split = "val"
        args.panoptic_video_mask_dir = os.path.join(args.cityscapes_vps_root, args.split, "panoptic_video")
        args.panoptic_inst_mask_dir = os.path.join(args.cityscapes_vps_root, args.split, "panoptic_inst")

        args.data_path = os.path.join(args.cityscapes_vps_root, args.split, "img_all")
        args.anno_path = '../data/cityscapes_vps/panoptic_gt_val_city_vps.json'

        args.vidname = vidname

        with open(args.anno_path, 'r') as f:
            panoptic_gt_val_city_vps = json.load(f)
                    
        panoptic_categories = panoptic_gt_val_city_vps['categories']
        # panoptic_images = panoptic_gt_val_city_vps['images']
        # panoptic_annotations = panoptic_gt_val_city_vps['annotations']    

        categories = panoptic_categories
        categories.append(
            {'id': -1, 'name': 'other', 'supercategory': '', 'color':None}
        )
        categories_dict = {el['id']: el for el in categories}
        
    elif dataset_name == "vipseg":
        args.VIPSeg_720P_root = '../data/VIPSeg-Dataset/VIPSeg/VIPSeg_720P'
        args.panomasks_dir = os.path.join(args.VIPSeg_720P_root, "panomasks")
        args.panomasksRGB_dir = os.path.join(args.VIPSeg_720P_root, "panomasksRGB")
        args.data_path = os.path.join(args.VIPSeg_720P_root, "images")
        args.anno_path = '../data/VIPSeg-Dataset/VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg.json'

        args.vidname = vidname

        args.crop_list = '640_1280'

        with open(args.anno_path, 'r') as f:
            panoptic_gt_VIPSeg = json.load(f)
                    
        panoptic_categories = panoptic_gt_VIPSeg['categories']
        # panoptic_videos = panoptic_gt_VIPSeg['videos']
        # panoptic_annotations = panoptic_gt_VIPSeg['annotations']    
        
        categories = panoptic_categories
        categories.append(
            {'id': -1, 'name': 'other', 'supercategory': '', 'color':None}
        )
        categories_dict = {el['id']: el for el in categories}
        
    return args, categories_dict

def get_instance_info_for_video(inference_results, object_categories, categories):
    instance_to_ann_id_maps = {}
    
    # Create a map from unique inst_id to a suffix that denotes an instance number in current video. Also stores object category.
    inst_id_to_cat_and_inst_suffix = {}
    
    object_to_instances_map = {}
    obj_to_obj_name_idx = {}
    
    instance_names = []
    object_to_instances_map = defaultdict(list)
    
    for idx, object_cat in enumerate(object_categories):
        obj_to_obj_name_idx[object_cat] = idx
    
    for img_idx in inference_results.keys():
        instance_to_ann_id_maps[img_idx] = {}
    
        # Get annos for current frame
        frame_annos = inference_results[img_idx]["annotations"]
        for ann in frame_annos:
            category_name = [cat["name"] for cat in categories if cat["id"] == ann["category_id"]][0]
            
            # Get the current number of instances of this category            
            num_instances_of_obj = len(object_to_instances_map[category_name])
            
            if ann["inst_id"] not in list(inst_id_to_cat_and_inst_suffix.keys()):
                # Create a dictionary for the instance
                inst_id_to_cat_and_inst_suffix[ann["inst_id"]] = {
                    "category": category_name,
                    "inst_suffix": num_instances_of_obj, #0
                    "instance_name": category_name + '_' + str(num_instances_of_obj)
                }
    
            # Retrieve the stored instance name
            instance_name = inst_id_to_cat_and_inst_suffix[ann["inst_id"]]["instance_name"]
    
            instance_to_ann_id_maps[img_idx][instance_name] = ann['id']
    
            if instance_name not in instance_names:
                object_to_instances_map[category_name].append(instance_name)
                instance_names.append(instance_name)

    def custom_sort_key(item):
        parts = item.split('_')
        return ("_".join(parts[:-1]), int(parts[-1]))
        
    # Sort the instance names
    instance_names = [item for item in sorted(instance_names, key=custom_sort_key)]
    
    # Find "other_0" instance in this list and move it to the back
    instance_names.append(instance_names.pop(instance_names.index("other_0")))
    
    return inst_id_to_cat_and_inst_suffix, instance_to_ann_id_maps, object_to_instances_map, obj_to_obj_name_idx, instance_names

def get_instance_contribs(
    head_layer_output_contrib, nerv_blk_3_output_contrib, annotations, instance_to_ann_id_map, instance_names, inst_id_to_cat_and_inst_suffix   
):
    total_img_area = head_layer_output_contrib.size(-2) * head_layer_output_contrib.size(-1)
    
    # Maps for kernel to object contributions
    num_head_kernels = head_layer_output_contrib.shape[0] * head_layer_output_contrib.shape[1]
    num_blk_3_kernels = nerv_blk_3_output_contrib.shape[0] * nerv_blk_3_output_contrib.shape[1]

    # Get the list of instances which are in current frame
    # instance IDs need to be mapped to their instance names
    instances_in_curr_frame = list(instance_to_ann_id_map.keys())
    
    # Preserve the order wrt original instance_names
    instances_in_curr_frame = [instance_names[i] for i in range(len(instance_names)) if instance_names[i] in instances_in_curr_frame]
    # Hopefully "other_0" is already at the end of this list now
    

    # change num_instances to len(instance_names) if looking at video 
    # num_instances = len(instance_names)
    num_instances = len(instances_in_curr_frame)
    
    instance_areas = torch.zeros(num_instances)
    
    head_kernel_to_instance_contribs = torch.zeros((num_head_kernels, num_instances))
    blk_3_kernel_to_instance_contribs = torch.zeros((num_blk_3_kernels, num_instances))

    head_instance_contrib_ratio_to_total = torch.zeros((num_head_kernels, len(instances_in_curr_frame)))
    blk_3_instance_contrib_ratio_to_total = torch.zeros((num_blk_3_kernels, len(instances_in_curr_frame)))
    # Store the total kernel-wise contributions to output image
    total_head_layer_output_contrib = torch.sum(torch.abs(head_layer_output_contrib), dim=(2,3))
    total_blk_3_output_contrib = torch.sum(torch.abs(nerv_blk_3_output_contrib), dim=(2,3))

    for instance in instances_in_curr_frame:
        ann_id = instance_to_ann_id_map[instance]
        ann = [ann for ann in annotations if ann['id'] == ann_id][0]
        
        area = ann['area']
        binary_mask = ann['binary_mask'].squeeze()
    
        # Use binary mask of shape hxw to index into the n1xn2xhxw contribution tensor
        # to get the contribs for the current instance
        instance_head_contribs = torch.abs(head_layer_output_contrib[:, :, binary_mask]) # n1 x n2 x n_inst_pixels 
        instance_blk_3_contribs = torch.abs(nerv_blk_3_output_contrib[:, :, binary_mask]) # n1 x n2 x n_inst_pixels 
    
        # Get aggregated total contribution for each kernel to the instance
        total_head_contrib = torch.sum(instance_head_contribs, dim=-1)
        total_blk_3_contrib = torch.sum(instance_blk_3_contribs, dim=-1)
        avg_head_contrib = total_head_contrib / area
        avg_blk_3_contrib = total_blk_3_contrib / area

        inst_idx = instances_in_curr_frame.index(instance)
        
        head_kernel_to_instance_contribs[:, inst_idx] = avg_head_contrib.flatten()
        # Store the average contribution from each block 3 kernel to current instance
        blk_3_kernel_to_instance_contribs[:, inst_idx] = avg_blk_3_contrib.flatten()

        # Store percentage of each instance's contribution to total head contribution to image
        head_expected_instance_contrib = total_head_layer_output_contrib.flatten(0,1) * (area / total_img_area)
        head_true_instance_contrib = total_head_contrib.flatten(0,1) 
        head_instance_contrib_ratio_to_total[:, inst_idx] = torch.abs(head_true_instance_contrib - head_expected_instance_contrib) / head_expected_instance_contrib
        
        # Store percentage of each instance's contribution to total head contribution to image
        blk_3_expected_instance_contrib = total_blk_3_output_contrib.flatten(0,1)  * (area / total_img_area)
        blk_3_true_instance_contrib = total_blk_3_contrib.flatten(0,1)
        blk_3_instance_contrib_ratio_to_total[:, inst_idx] = torch.abs(blk_3_true_instance_contrib - blk_3_expected_instance_contrib) / blk_3_expected_instance_contrib

        # Store instance_areas for geometric mean
        instance_areas[inst_idx] = area

    return head_kernel_to_instance_contribs, blk_3_kernel_to_instance_contribs, \
        head_instance_contrib_ratio_to_total, blk_3_instance_contrib_ratio_to_total, instance_areas

def get_gridcell_contribs(
    head_layer_output_contrib, nerv_blk_3_output_contrib, reg_stride_h, reg_stride_w
):
    total_img_area = head_layer_output_contrib.size(-2) * head_layer_output_contrib.size(-1)

    head_layer_output_contrib = torch.flatten(head_layer_output_contrib, 0, 1) # num_kernels x h x w
    nerv_blk_3_output_contrib = torch.flatten(nerv_blk_3_output_contrib, 0, 1)
    
    # Take absolute of contributions # num_kernels x cell_stride x cell_stride x h/cell_stride x w/cell_stride. e.g. k x 4 x 4 x h/4 x w/4
    unfolded_head_kernel_to_gridcell_contribs = torch.abs(head_layer_output_contrib).unfold(1, reg_stride_h, reg_stride_h).unfold(2, reg_stride_w, reg_stride_w).permute(0, 3, 4, 1, 2)
    unfolded_blk_3_kernel_to_gridcell_contribs = torch.abs(nerv_blk_3_output_contrib).unfold(1, reg_stride_h, reg_stride_h).unfold(2, reg_stride_w, reg_stride_w).permute(0, 3, 4, 1, 2)

    # Store the total kernel-wise contributions to output image
    total_head_layer_output_contrib = torch.sum(torch.abs(head_layer_output_contrib), dim=(1,2))
    total_blk_3_output_contrib = torch.sum(torch.abs(nerv_blk_3_output_contrib), dim=(1,2))

    gridcell_area = unfolded_head_kernel_to_gridcell_contribs.size(3) * unfolded_head_kernel_to_gridcell_contribs.size(4)
    
    # take absolute of contributions **after** we store our raw per-region contribs
    head_kernel_to_gridcell_contribs = torch.abs(unfolded_head_kernel_to_gridcell_contribs)
    blk_3_kernel_to_gridcell_contribs = torch.abs(unfolded_blk_3_kernel_to_gridcell_contribs)
    
    # Flatten head_kernel_to_gridcell_contribs by region before taking variance over pixels in region
    flattened_head_gridcell_contribs = head_kernel_to_gridcell_contribs.flatten(3, 4) # num_kernels x cell_stride x cell_stride x h/cell_stride*w/cell_stride
    flattened_head_gridcell_contribs = flattened_head_gridcell_contribs.flatten(1, 2) # num_kernels x num_gridcells x h/cell_stride*w/cell_stride
    flattened_blk_3_gridcell_contribs = blk_3_kernel_to_gridcell_contribs.flatten(3, 4)
    flattened_blk_3_gridcell_contribs = flattened_blk_3_gridcell_contribs.flatten(1, 2)

    # Find delta percentages - ( true contrib - expected contrib ) / expected contrib
    head_expected_region_contrib = total_head_layer_output_contrib[:,None] * (gridcell_area / total_img_area)
    head_true_region_contrib = torch.sum(flattened_head_gridcell_contribs, dim=-1)
    head_gridcell_contrib_ratio_to_total = (head_true_region_contrib - head_expected_region_contrib) / head_expected_region_contrib

    
    blk_3_expected_instance_contrib = total_blk_3_output_contrib[:,None] * (gridcell_area / total_img_area)
    blk_3_true_instance_contrib = torch.sum(flattened_blk_3_gridcell_contribs, dim=-1)
    blk_3_gridcell_contrib_ratio_to_total = (blk_3_true_instance_contrib - blk_3_expected_instance_contrib) / blk_3_expected_instance_contrib
    
    # num_kernels x cell_stride x cell_stride
    head_kernel_to_gridcell_contribs = head_kernel_to_gridcell_contribs.sum(dim=(3, 4)) / gridcell_area
    blk_3_kernel_to_gridcell_contribs = blk_3_kernel_to_gridcell_contribs.sum(dim=(3, 4)) / gridcell_area
    
    # Reshape the (cell_stride x cell_stride) dim to num_gridcells
    head_layer_feature_vectors = head_kernel_to_gridcell_contribs.view(head_kernel_to_gridcell_contribs.size(0), -1) # num_head_kernels x num_gridcells
    nerv_blk_3_feature_vectors = blk_3_kernel_to_gridcell_contribs.view(blk_3_kernel_to_gridcell_contribs.size(0), -1) # num_blk_3_kernels x num_gridcells

    return head_layer_feature_vectors, nerv_blk_3_feature_vectors, head_gridcell_contrib_ratio_to_total, blk_3_gridcell_contrib_ratio_to_total

def get_rgb_cluster_contribs(
    head_layer_output_contrib, nerv_blk_3_output_contrib, rgb_cluster_map
):
    total_img_area = head_layer_output_contrib.size(-2) * head_layer_output_contrib.size(-1)
    
    n_rgb_clusters = len(np.unique(rgb_cluster_map))
    num_head_kernels = head_layer_output_contrib.shape[0] * head_layer_output_contrib.shape[1]
    num_blk_3_kernels = nerv_blk_3_output_contrib.shape[0] * nerv_blk_3_output_contrib.shape[1]
    
    rgb_cluster_areas = torch.zeros(n_rgb_clusters)
    
    head_kernel_to_rgb_cluster_contribs = torch.zeros((num_head_kernels, n_rgb_clusters))
    blk_3_kernel_to_rgb_cluster_contribs = torch.zeros((num_blk_3_kernels, n_rgb_clusters))
    
    head_rgb_cluster_contrib_ratio_to_total = torch.zeros((num_head_kernels, n_rgb_clusters))
    blk_3_rgb_cluster_contrib_ratio_to_total = torch.zeros((num_blk_3_kernels, n_rgb_clusters))
    # Store the total kernel-wise contributions to output image
    total_head_layer_output_contrib = torch.sum(torch.abs(head_layer_output_contrib), dim=(2,3))
    total_blk_3_output_contrib = torch.sum(torch.abs(nerv_blk_3_output_contrib), dim=(2,3))
    
    total_area = 0
    for cluster_id in np.unique(rgb_cluster_map):
        
        # Construct a binary mask of shape hxw for the current rgb cluster
        binary_mask = (rgb_cluster_map == cluster_id)
        binary_mask = binary_mask.squeeze().astype(bool)
        area = binary_mask.sum()
    
        # Use binary mask of shape hxw to index into the n1xn2xhxw contribution tensor
        # to get the contribs for the current superpixel
        rgb_cluster_head_contribs = torch.abs(head_layer_output_contrib[:, :, binary_mask])
        rgb_cluster_blk_3_contribs = torch.abs(nerv_blk_3_output_contrib[:, :, binary_mask])
    
        # Get aggregated total contribution for each kernel to the instance
        total_head_contrib = torch.sum(rgb_cluster_head_contribs, dim=-1)
        avg_head_contrib = total_head_contrib / area
        
        total_blk_3_contrib = torch.sum(rgb_cluster_blk_3_contribs, dim=-1)
        avg_blk_3_contrib = total_blk_3_contrib / area
        

        # Store percentage of each superpixel's contribution to total contribution to image
        head_expected_rgb_cluster_contrib = total_head_layer_output_contrib.flatten(0,1)  * (area / total_img_area)
        head_true_rgb_cluster_contrib = total_head_contrib.flatten(0,1) 
        head_rgb_cluster_contrib_ratio_to_total[:, cluster_id] = torch.abs(head_true_rgb_cluster_contrib - head_expected_rgb_cluster_contrib) / head_expected_rgb_cluster_contrib
        
        blk_3_expected_rgb_cluster_contrib = total_blk_3_output_contrib.flatten(0,1)  * (area / total_img_area)
        blk_3_true_rgb_cluster_contrib = total_blk_3_contrib.flatten(0,1)
        blk_3_rgb_cluster_contrib_ratio_to_total[:, cluster_id] = torch.abs(blk_3_true_rgb_cluster_contrib - blk_3_expected_rgb_cluster_contrib) / blk_3_expected_rgb_cluster_contrib

        rgb_cluster_areas[cluster_id] = area
        
        head_kernel_to_rgb_cluster_contribs[:, cluster_id] = avg_head_contrib.flatten()
        # Store the average contribution from each block 3 kernel to current instance
        blk_3_kernel_to_rgb_cluster_contribs[:, cluster_id] = avg_blk_3_contrib.flatten()
    
    return head_kernel_to_rgb_cluster_contribs, blk_3_kernel_to_rgb_cluster_contribs, \
        head_rgb_cluster_contrib_ratio_to_total, blk_3_rgb_cluster_contrib_ratio_to_total, rgb_cluster_areas

def compute_kmeans_clusters_in_rgb(image, num_clusters):
    # Reshape to 2D array of num_pixels x 3 (for rgb)
    image_reshaped_rgb = image.reshape(-1, 3)
    
    # Perform kmeans clustering
    kmeans = KMeans(n_clusters=num_clusters, n_init=1, random_state=0).fit(image_reshaped_rgb)
    rgb_cluster_map = kmeans.labels_.reshape(image.shape[0], image.shape[1])
    
    return rgb_cluster_map

from einops import rearrange


def get_gabor_kernels():
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25, 0.5):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                            sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
        return kernels

def compute_gabor(image):
    kernels = get_gabor_kernels()
    all_filtered = []
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')[...,None]
        all_filtered.append(filtered)
    all_filtered = np.concatenate(all_filtered, axis=-1)
    return all_filtered

def get_spatial_clustering(img, clustering_type, num_clusters=None):
    img_h, img_w, _ = img.shape
    img_pixels = rearrange(img, 'h w c -> (h w) c')
    if clustering_type == 'kmeans':
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(img_pixels)
        labels = kmeans.labels_
    else:
        raise NotImplementedError(f'{clustering_type} not implemented')
    labels = rearrange(labels, '(h w) -> h w', h=img_h, w=img_w)
    return labels


def get_gabor_label_map(img, num_clusters):
    img_to_cluster = img.copy()
    img_to_cluster = np.array(Image.fromarray(img_to_cluster).convert('L'))
    img_to_cluster = compute_gabor(img_to_cluster)
    label_map = get_spatial_clustering(img_to_cluster, clustering_type='kmeans', num_clusters=num_clusters)
    return label_map

def get_gabor_cluster_contribs(
    head_layer_output_contrib, nerv_blk_3_output_contrib, gabor_cluster_map
):
    total_img_area = head_layer_output_contrib.size(-2) * head_layer_output_contrib.size(-1)
    
    n_gabor_clusters = len(np.unique(gabor_cluster_map))
    num_head_kernels = head_layer_output_contrib.shape[0] * head_layer_output_contrib.shape[1]
    num_blk_3_kernels = nerv_blk_3_output_contrib.shape[0] * nerv_blk_3_output_contrib.shape[1]
    
    gabor_cluster_areas = torch.zeros(n_gabor_clusters)
    
    head_kernel_to_gabor_cluster_contribs = torch.zeros((num_head_kernels, n_gabor_clusters))
    blk_3_kernel_to_gabor_cluster_contribs = torch.zeros((num_blk_3_kernels, n_gabor_clusters))
    
    head_gabor_cluster_contrib_ratio_to_total = torch.zeros((num_head_kernels, n_gabor_clusters))
    blk_3_gabor_cluster_contrib_ratio_to_total = torch.zeros((num_blk_3_kernels, n_gabor_clusters))
    # Store the total kernel-wise contributions to output image
    total_head_layer_output_contrib = torch.sum(torch.abs(head_layer_output_contrib), dim=(2,3))
    total_blk_3_output_contrib = torch.sum(torch.abs(nerv_blk_3_output_contrib), dim=(2,3))
    
    total_area = 0
    for cluster_id in np.unique(gabor_cluster_map):
        
        # Construct a binary mask of shape hxw for the current rgb cluster
        binary_mask = (gabor_cluster_map == cluster_id)
        binary_mask = binary_mask.squeeze().astype(bool)
        area = binary_mask.sum()
    
        # Use binary mask of shape hxw to index into the n1xn2xhxw contribution tensor
        # to get the contribs for the current superpixel
        # Take ABSOLUTE values
        gabor_cluster_head_contribs = torch.abs(head_layer_output_contrib[:, :, binary_mask])
        gabor_cluster_blk_3_contribs = torch.abs(nerv_blk_3_output_contrib[:, :, binary_mask])
    
        # Get aggregated total contribution for each kernel to the instance
        total_head_contrib = torch.sum(gabor_cluster_head_contribs, dim=-1)
        avg_head_contrib = total_head_contrib / area
        
        total_blk_3_contrib = torch.sum(gabor_cluster_blk_3_contribs, dim=-1)
        avg_blk_3_contrib = total_blk_3_contrib / area
        

        # Store percentage of each superpixel's contribution to total contribution to image
        head_expected_gabor_cluster_contrib = total_head_layer_output_contrib.flatten(0,1)  * (area / total_img_area)
        head_true_gabor_cluster_contrib = total_head_contrib.flatten(0,1) 
        head_gabor_cluster_contrib_ratio_to_total[:, cluster_id] = torch.abs(head_true_gabor_cluster_contrib - head_expected_gabor_cluster_contrib) / head_expected_gabor_cluster_contrib
        
        blk_3_expected_gabor_cluster_contrib = total_blk_3_output_contrib.flatten(0,1)  * (area / total_img_area)
        blk_3_true_gabor_cluster_contrib = total_blk_3_contrib.flatten(0,1)
        blk_3_gabor_cluster_contrib_ratio_to_total[:, cluster_id] = torch.abs(blk_3_true_gabor_cluster_contrib - blk_3_expected_gabor_cluster_contrib) / blk_3_expected_gabor_cluster_contrib

        gabor_cluster_areas[cluster_id] = area
        
        head_kernel_to_gabor_cluster_contribs[:, cluster_id] = avg_head_contrib.flatten()
        # Store the average contribution from each block 3 kernel to current instance
        blk_3_kernel_to_gabor_cluster_contribs[:, cluster_id] = avg_blk_3_contrib.flatten()
    
    return head_kernel_to_gabor_cluster_contribs, blk_3_kernel_to_gabor_cluster_contribs, \
        head_gabor_cluster_contrib_ratio_to_total, blk_3_gabor_cluster_contrib_ratio_to_total, gabor_cluster_areas
        

def compute_inference_results(dataset_name, train_dataloader, model, categories_dict, args, num_rgb_clusters, num_gabor_clusters):
    # Find object categories across instances
    
    if dataset_name == "vipseg":
        # Sample few frames
        num_indices = len(train_dataloader) * args.b
        num_samples = 6
        sampled_img_indices = [i * (num_indices - 1) // (num_samples - 1) for i in range(num_samples)]
    
    categories_in_video = {}
    inference_results = {}
    
    with torch.no_grad():
        for batch in train_dataloader:
            img_data, norm_idx, img_idx = batch['img'].to('cuda'), batch['norm_idx'].to('cuda'), batch['idx'].to('cuda')
            annotations = batch['annotations']
            
            if dataset_name == "vipseg" and (img_idx not in sampled_img_indices):
                continue
            
            if len(annotations) > 0:
                annotations = convert_tensor_annotations_to_numpy(annotations)
                # Filter annotations to remove non-persistent instances - i.e. those that are not in all frames
                annotations = add_other_annotation(annotations)
                
                images = batch['img'].cuda()
                downsample_results, stage_results, img_embed, decoder_results, img_out = model(norm_idx)


                # Compute superpixels on the image
                image_numpy = images[0].permute(1,2,0).cpu().numpy()
                
                rgb_cluster_map = compute_kmeans_clusters_in_rgb(image_numpy, num_rgb_clusters)
                
                # Compute Gabor clusters map
                image_pil_format = (images[0].clamp(0,1) * 255).permute(1,2,0).cpu().numpy().astype(np.uint8)
                gabor_cluster_map = get_gabor_label_map(image_pil_format, num_gabor_clusters)

                # Save all input-output information related to annotated images
                inference_results[img_idx.item()] = {
                    "decoder_results": decoder_results,
                    "img_out": img_out,
                    "annotations": annotations,
                    "rgb_cluster_map": rgb_cluster_map,
                    "gabor_cluster_map": gabor_cluster_map
                }
                
                for ann in annotations:
                    if ann["category_id"] not in categories_in_video:
                        categories_in_video[ann["category_id"]] = categories_dict[ann["category_id"]]

    # Add "other" and turn object_categories into a list
    categories_in_video[-1] = categories_dict[-1]
    object_categories = [v['name'] for k, v in categories_in_video.items()]
    categories_in_video = [v for k, v in categories_in_video.items()]
    
    return inference_results, categories_in_video, object_categories

def compute_all_variables_for_video(model, args, inference_results, instance_to_ann_id_maps, cell_stride_h, cell_stride_w, 
                                    instance_names, inst_id_to_cat_and_inst_suffix, num_rgb_clusters,
                                    num_gabor_clusters, compute_across_frames=True):

    all_variables_for_video = {}
    num_regions = cell_stride_h * cell_stride_w

    for img_idx, value in inference_results.items():
        img_out = value["img_out"]
        decoder_results = value["decoder_results"]
        annotations = value["annotations"]
        rgb_cluster_map = value["rgb_cluster_map"]
        gabor_cluster_map = value["gabor_cluster_map"]

        instance_to_ann_id_map = instance_to_ann_id_maps[img_idx]
        
        compute_contrib_obj = ComputeContributions(
            model, args, decoder_results, img_out.detach().clone()[0]
        )
        
        head_layer_output_contrib = compute_contrib_obj.compute_head_mappings()
        nerv_blk_3_output_contrib, _ = compute_contrib_obj.compute_last_nerv_block_mappings()

        # Get instance contributions and normalize them. Also get variances within instance.
        head_kernel_to_instance_contribs, blk_3_kernel_to_instance_contribs, \
            head_instance_contrib_ratio_to_total, blk_3_instance_contrib_ratio_to_total, instance_areas \
            = get_instance_contribs(head_layer_output_contrib, nerv_blk_3_output_contrib, annotations, instance_to_ann_id_map, instance_names, inst_id_to_cat_and_inst_suffix)

        # Get gridcell contributions and normalize them. Also get variances within regions.
        head_kernel_to_gridcell_contribs, blk_3_kernel_to_gridcell_contribs, \
            head_gridcell_contrib_ratio_to_total, blk_3_gridcell_contrib_ratio_to_total \
            = get_gridcell_contribs(head_layer_output_contrib, nerv_blk_3_output_contrib, cell_stride_h, cell_stride_w)

        # Get RGB kmeans clustered contributions and normalize them. Also get variances within RGB clusters.
        head_kernel_to_rgb_cluster_contribs, blk_3_kernel_to_rgb_cluster_contribs, \
            head_rgb_cluster_contrib_ratio_to_total, blk_3_rgb_cluster_contrib_ratio_to_total, rgb_cluster_areas \
                = get_rgb_cluster_contribs(head_layer_output_contrib, nerv_blk_3_output_contrib, rgb_cluster_map)

        # Get Gabor filter clustered contribs
        head_kernel_to_gabor_cluster_contribs, blk_3_kernel_to_gabor_cluster_contribs, \
            head_gabor_cluster_contrib_ratio_to_total, blk_3_gabor_cluster_contrib_ratio_to_total, gabor_cluster_areas \
                = get_gabor_cluster_contribs(head_layer_output_contrib, nerv_blk_3_output_contrib, gabor_cluster_map)
        
        
        all_variables_for_video[img_idx] = {
            "head_layer_output_contrib": torch.abs(head_layer_output_contrib),
            "nerv_blk_3_output_contrib": torch.abs(nerv_blk_3_output_contrib),
            # head contribs
            "head_kernel_to_instance_contribs": head_kernel_to_instance_contribs,
            "head_kernel_to_gridcell_contribs": head_kernel_to_gridcell_contribs,
            "head_kernel_to_rgb_cluster_contribs": head_kernel_to_rgb_cluster_contribs,
            "head_kernel_to_gabor_cluster_contribs": head_kernel_to_gabor_cluster_contribs,
            
            # blk_3 contribs
            "blk_3_kernel_to_instance_contribs": blk_3_kernel_to_instance_contribs,
            "blk_3_kernel_to_gridcell_contribs": blk_3_kernel_to_gridcell_contribs,
            "blk_3_kernel_to_rgb_cluster_contribs": blk_3_kernel_to_rgb_cluster_contribs,
            "blk_3_kernel_to_gabor_cluster_contribs": blk_3_kernel_to_gabor_cluster_contribs,

            # areas
            "instance_areas": instance_areas,
            "rgb_cluster_areas": rgb_cluster_areas,
            "gabor_cluster_areas": gabor_cluster_areas,

            "num_instances_in_frame": len(instance_areas),
            "cell_stride_h": cell_stride_h,
            "cell_stride_w": cell_stride_w,
            "num_rgb_clusters": num_rgb_clusters,
            "num_gabor_clusters": num_gabor_clusters,

            # per-patch contribution ratios
            "head_instance_contrib_ratio_to_total": head_instance_contrib_ratio_to_total,
            "blk_3_instance_contrib_ratio_to_total": blk_3_instance_contrib_ratio_to_total,
            "head_gridcell_contrib_ratio_to_total": head_gridcell_contrib_ratio_to_total,
            "blk_3_gridcell_contrib_ratio_to_total": blk_3_gridcell_contrib_ratio_to_total,
            "head_rgb_cluster_contrib_ratio_to_total": head_rgb_cluster_contrib_ratio_to_total,
            "blk_3_rgb_cluster_contrib_ratio_to_total": blk_3_rgb_cluster_contrib_ratio_to_total,
            "head_gabor_cluster_contrib_ratio_to_total": head_gabor_cluster_contrib_ratio_to_total,
            "blk_3_gabor_cluster_contrib_ratio_to_total": blk_3_gabor_cluster_contrib_ratio_to_total
        }
    
    return all_variables_for_video

def compute_variance_of_deltas(all_variables_for_video):

    # Take variance of deltas (between expected patch contrib, actual patch contrib)
    img_idx = 0
    all_vars_for_first_frame = all_variables_for_video[img_idx]
    
    num_instances_in_frame = all_vars_for_first_frame["num_instances_in_frame"]
    
    fig, axs = plt.subplots(1, 2, tight_layout=True) #share_y=True
    
    # One variance per neuron - variance is over all H*W pixels in the contribution map of that neuron
    # If any variances are nan, then the neuron is not contribution to any pixel in the image - it is dead. prune such neurons - dirty way is to drop nans later :)
    
    head_instance_variances = torch.var(all_vars_for_first_frame["head_instance_contrib_ratio_to_total"], dim=-1)
    head_gridcell_variances = torch.var(all_vars_for_first_frame["head_gridcell_contrib_ratio_to_total"], dim=-1)
    head_rgb_cluster_variances = torch.var(all_vars_for_first_frame["head_rgb_cluster_contrib_ratio_to_total"], dim=-1)
    head_gabor_cluster_variances = torch.var(all_vars_for_first_frame["head_gabor_cluster_contrib_ratio_to_total"], dim=-1)
    
    blk_3_instance_variances = torch.var(all_vars_for_first_frame["blk_3_instance_contrib_ratio_to_total"], dim=-1)
    blk_3_gridcell_variances = torch.var(all_vars_for_first_frame["blk_3_gridcell_contrib_ratio_to_total"], dim=-1)
    blk_3_rgb_cluster_variances = torch.var(all_vars_for_first_frame["blk_3_rgb_cluster_contrib_ratio_to_total"], dim=-1)
    blk_3_gabor_cluster_variances = torch.var(all_vars_for_first_frame["blk_3_gabor_cluster_contrib_ratio_to_total"], dim=-1)
    
    # Sort the variances in ascending order
    sorted_variance_head_instance_contrib_ratio, head_instance_sorted_indices = torch.sort(head_instance_variances)
    sorted_variance_head_gridcell_contrib_ratio, head_gridcell_sorted_indices = torch.sort(head_gridcell_variances)
    sorted_variance_head_rgb_cluster_contrib_ratio, head_rgb_sorted_indices = torch.sort(head_rgb_cluster_variances)
    sorted_variance_head_gabor_cluster_contrib_ratio, head_gabor_sorted_indices = torch.sort(head_gabor_cluster_variances)
    
    sorted_variance_blk_3_instance_contrib_ratio, blk_3_instance_sorted_indices = torch.sort(blk_3_instance_variances)
    sorted_variance_blk_3_gridcell_contrib_ratio, blk_3_gridcell_sorted_indices = torch.sort(blk_3_gridcell_variances)
    sorted_variance_blk_3_rgb_cluster_contrib_ratio, blk_3_rgb_sorted_indices = torch.sort(blk_3_rgb_cluster_variances)
    sorted_variance_blk_3_gabor_cluster_contrib_ratio, blk_3_gabor_sorted_indices = torch.sort(blk_3_gabor_cluster_variances)
    
    # Drop NaNs from above sortings
    
    axs[0].plot(sorted_variance_head_instance_contrib_ratio, label="Instances variance", c='r')
    axs[0].plot(sorted_variance_head_gridcell_contrib_ratio, label="Grid cells variance", c='g')
    axs[0].plot(sorted_variance_head_rgb_cluster_contrib_ratio, label="RGB Cluster variance", c='b')
    axs[0].plot(sorted_variance_head_gabor_cluster_contrib_ratio, label="Gabor Cluster variance", c='m')
    
    axs[0].set_title(f"Head Layer")
    
    
    axs[1].plot(sorted_variance_blk_3_instance_contrib_ratio, label="Instances variance", c='r')
    axs[1].plot(sorted_variance_blk_3_gridcell_contrib_ratio, label="Grid cells variance", c='g')
    axs[1].plot(sorted_variance_blk_3_rgb_cluster_contrib_ratio, label="RGB Cluster variance", c='b')
    axs[1].plot(sorted_variance_blk_3_gabor_cluster_contrib_ratio, label="Gabor Cluster variance", c='m')
    
    axs[1].set_title(f"NeRV Block 3")

    
    sorted_deltas_dict = {
        "head": {
            "instances_deltas": sorted_variance_head_instance_contrib_ratio,
            "gridcells_deltas": sorted_variance_head_gridcell_contrib_ratio,
            "rgb_clusters_deltas": sorted_variance_head_rgb_cluster_contrib_ratio,
            "gabor_clusters_deltas": sorted_variance_head_gabor_cluster_contrib_ratio
        }, "blk_3": {
            "instances_deltas": sorted_variance_blk_3_instance_contrib_ratio,
            "gridcells_deltas": sorted_variance_blk_3_gridcell_contrib_ratio,
            "rgb_clusters_deltas": sorted_variance_blk_3_rgb_cluster_contrib_ratio,
            "gabor_clusters_deltas": sorted_variance_blk_3_gabor_cluster_contrib_ratio
        }, "sorted_indices": {
            "head":{
                "instances": head_instance_sorted_indices,
                "gridcells": head_gridcell_sorted_indices,
                "rgb_clusters": head_rgb_sorted_indices,
                "gabor_clusters": head_gabor_sorted_indices
            }, "blk_3": {
                "instances": blk_3_instance_sorted_indices,
                "gridcells": blk_3_gridcell_sorted_indices,
                "rgb_clusters": blk_3_rgb_sorted_indices,
                "gabor_clusters": blk_3_gabor_sorted_indices
            }
        }
    }
    
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.05))
    
    return sorted_deltas_dict


def main():
    dataset_names = ['cityscapes', 'vipseg']
    vidnames = {
        'cityscapes': ["0495", "0065",],
        'vipseg': ["1327_e96s6pJ5x3Y", "127_-hIVCYO4C90", "2350_sCLtK1a2GGc", ]
    }
    vid_data_folder_name = {
        "cityscapes": "Cityscapes_VPS_models",
        "vipseg": "VIPSeg_models"
    }


    args_dict = {}
    dataloader_dict = {}
    weights_dict = {}
    models_dict = {}
    categories_dicts = {}

    for dataset_name in dataset_names:
        weights_dict[dataset_name] = {}
        args_dict[dataset_name] = {}
        dataloader_dict[dataset_name] = {}
        models_dict[dataset_name] = {}
        categories_dicts[dataset_name] = {}
        
        for vidname in vidnames[dataset_name]:
            vid_data_folder = vid_data_folder_name[dataset_name]
            weights_dict[dataset_name][vidname] = f'../HNeRV/output/{vid_data_folder}/{vidname}_128_256_modelsize1.0/{vidname}/1_1_1_pe_1.25_80_Dim64_16_FC8_16_KS0_3_3_RED1.2_low6_blk1_1_e1000_b2_quant_M8_E6_lr0.001_cosine_0.1_1_0.1_L2_Size1.0_ENC_convnext__DEC_pshuffel_4,2,2_gelu1_1'
            
            args = load_model_args()
            
            args.weight = os.path.join(weights_dict[dataset_name][vidname], f'model_best.pth')
            args.crop_list = '-1' if dataset_name == "cityscapes" else '640_1280'

            model = HNeRV(args)  
            model = load_model_checkpoint(model, args)
            
            models_dict[dataset_name][vidname] = model
            
            # Add dataset specific args
            args, categories_dicts[dataset_name][vidname] = load_dataset_specific_args(args, dataset_name, vidname)
            
            args_dict[dataset_name][vidname] = args
        
    # Create dataloader

    for dataset_name in dataset_names:
        for vidname in vidnames[dataset_name]:
            
            args = args_dict[dataset_name][vidname]
            
            if dataset_name == "cityscapes":
                full_dataset = CityscapesVPSVideoDataSet(args)
            else:
                full_dataset = VIPSegVideoDataSet(args)
                
            sampler = torch.utils.data.distributed.DistributedSampler(full_dataset) if args.distributed else None

            args.final_size = full_dataset.final_size
            args.full_data_length = len(full_dataset)
            split_num_list = [int(x) for x in args.data_split.split('_')]
            train_ind_list, args.val_ind_list = data_split(list(range(args.full_data_length)), split_num_list, args.shuffle_data, 0)

            #  Make sure the testing dataset is fixed for every run
            train_dataset =  Subset(full_dataset, train_ind_list)
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
            # make sure that every field in annotations dict is not converted to tensor while constructing the dataloader

            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False,#train_sampler is None),
                num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, worker_init_fn=worker_init_fn)#
        
            dataloader_dict[dataset_name][vidname] = train_dataloader
    
    per_vid_patches_deltas_var_dict = {}

    # Cluster settings (for now set all equalish to num_instances)
    num_rgb_and_gabor_clusters_dict = {
        "0495": 26,
        "0065": 27,
        "127_-hIVCYO4C90": 23,
        "2350_sCLtK1a2GGc": 26,
        "1327_e96s6pJ5x3Y": 16
    }
    cell_stride_h_dict = {
        "0495": 4,
        "0065": 4,
        "127_-hIVCYO4C90": 4,
        "2350_sCLtK1a2GGc": 4,
        "1327_e96s6pJ5x3Y": 4
    }
    cell_stride_w_dict = {
        "0495": 6,
        "0065": 6,
        "127_-hIVCYO4C90": 6,
        "2350_sCLtK1a2GGc": 6,
        "1327_e96s6pJ5x3Y": 4
    }



    for dataset_name in dataset_names:
        for vidname in vidnames[dataset_name]:
            train_dataloader = dataloader_dict[dataset_name][vidname]
            model = models_dict[dataset_name][vidname]
            args = args_dict[dataset_name][vidname]
            categories_dict = categories_dicts[dataset_name][vidname]
            
            categories = list(categories_dict.values())

            num_rgb_clusters = num_rgb_and_gabor_clusters_dict[vidname]
            num_gabor_clusters = num_rgb_and_gabor_clusters_dict[vidname]
            cell_stride_h, cell_stride_w = cell_stride_h_dict[vidname], cell_stride_w_dict[vidname]
            

            inference_results, categories_in_video, object_categories = compute_inference_results(
                dataset_name, train_dataloader, model, categories_dict, args, num_rgb_clusters, num_gabor_clusters
            )

            inst_id_to_cat_and_inst_suffix, instance_to_ann_id_maps, object_to_instances_map, \
                obj_to_obj_name_idx, instance_names = get_instance_info_for_video(inference_results, object_categories, categories)

            all_variables_for_video = compute_all_variables_for_video(
                model, args, inference_results, instance_to_ann_id_maps, cell_stride_h, cell_stride_w, 
                instance_names, inst_id_to_cat_and_inst_suffix, num_rgb_clusters,
                num_gabor_clusters
            )

            sorted_deltas_dict = compute_variance_of_deltas(all_variables_for_video)

            # For optical flow stuff, it might be easiest to 
            per_vid_patches_deltas_var_dict[vidname] = {
                "sorted_deltas_dict" : sorted_deltas_dict,
                "cluster_info": {
                    "num_instances": ["num_instances_in_frame"],
                    "num_rgb_clusters": num_rgb_clusters,
                    "num_gabor_clusters": num_gabor_clusters,
                    "cell_stride_h": cell_stride_h,
                    "cell_stride_w": cell_stride_w
                }
            }
            
    save_dir = '../plotting_source_data/supplementary/NeRV/C-INRs_perhaps_care_about_objects'
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, f"per_vid_patch_deltas_var_dict.pkl"), 'wb') as f:
        pickle.dump(per_vid_patches_deltas_var_dict, f)
        
if __name__ == "__main__":
    main()