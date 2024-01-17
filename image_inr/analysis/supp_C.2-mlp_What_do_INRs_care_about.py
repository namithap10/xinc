import json
import os
import pickle
import warnings
from collections import defaultdict

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from get_mlp_mappings import ComputeMLPContributions
from image_vps_datasets import (single_image_cityscape_vps_dataset,
                                single_image_vipseg_dataset)
from matplotlib.colors import ListedColormap
from model_all_analysis import ffn, lightning_model
from omegaconf import OmegaConf
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from utils import data_process, helper

warnings.filterwarnings("ignore")

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

def plot_image_with_instances(image, annotations, categories_dict, title=None):
    plt.rcParams["figure.figsize"] = 15, 10
    fig, ax = plt.subplots()

    # Plot the image
    ax.imshow(image)

    for anno in annotations:
        # Skip plotting "other" regions (regions without objects)
        if anno["category_id"] == -1:
            continue
        # Draw bbox
        x, y, w, h = anno["bbox"]

        cat_color = np.array(categories_dict[int(anno["category_id"])]['color']) / 255
        rectangle = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=cat_color, facecolor='none')
        ax.add_patch(rectangle)

        if 'binary_mask' in anno.keys():
            binary_mask = anno["binary_mask"].squeeze(0)
        else:
            raise ValueError("No binary mask found in annotation")
        # Create a mask where the binary mask is not zero
        mask = binary_mask != 0

        # Create rgba mask
        cmap = ListedColormap(cat_color)
        colored_mask = cmap(binary_mask.astype(float) / 1.0)

        # Create a mask where the binary mask is not zero
        mask = binary_mask != 0

        # Set the alpha channel to 0 for regions where the binary mask is zero
        colored_mask[:, :, 3] = mask.astype(float)
        
        # Display the colored mask over the image
        ax.imshow(colored_mask, alpha=0.5)

    if title is not None:
        plt.title(title)
    plt.show()
    
def compute_bbox(binary_mask):
    (rows, cols) = np.where(binary_mask > 0)
    x_min, x_max, y_min, y_max = min(cols), max(cols), min(rows), max(rows)
    # Create the bbox in COCO format [x, y, width, height]
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    bbox = [x_min, y_min, width, height]
    return bbox

def load_cfg(model_ckpt_dir, dataset_name, vidname):
    
    if dataset_name == "cityscapes":
        # Add cityscapes VPS paths
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
        
        with open(cfg.data.anno_path, 'r') as f:
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

def get_instance_info(inference_results, object_categories, categories):
    
    # Create a map from unique inst_id to a suffix that denotes an instance number in current video. Also stores object category.
    inst_id_to_cat_and_inst_suffix = {}
    
    object_to_instances_map = {}
    obj_to_obj_name_idx = {}
    
    instance_names = []
    object_to_instances_map = defaultdict(list)
    
    for idx, object_cat in enumerate(object_categories):
        obj_to_obj_name_idx[object_cat] = idx
    
    instance_to_ann_id_map = {}

    # Get annos for current frame
    frame_annos = inference_results["annotations"]
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

        instance_to_ann_id_map[instance_name] = ann['id']

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
    
    return inst_id_to_cat_and_inst_suffix, instance_to_ann_id_map, instance_names, object_to_instances_map, obj_to_obj_name_idx, instance_names

# For each instance - get average contrib, total contrib and total area (other useful info too)
def get_instance_contribs(
    layer_1_output_contrib, layer_2_output_contrib, layer_3_output_contrib, annotations, instance_to_ann_id_map, instance_names 
):
    total_img_area = layer_1_output_contrib.size(-2) * layer_1_output_contrib.size(-1)
    
    # Maps for kernel to object contributions
    num_layer_1_neurons = layer_1_output_contrib.shape[0]
    num_layer_2_neurons = layer_2_output_contrib.shape[0]
    num_layer_3_neurons = layer_3_output_contrib.shape[0]

    num_instances = len(instance_names)
    layer_1_to_instance_contribs = torch.zeros((num_layer_1_neurons, num_instances))
    layer_2_to_instance_contribs = torch.zeros((num_layer_2_neurons, num_instances))
    layer_3_to_instance_contribs = torch.zeros((num_layer_3_neurons, num_instances))

    instance_areas = torch.zeros(num_instances)
    
    # Use deltas idea to find percentage deviation of instance's actual contribution from expected
    layer_1_instance_contrib_ratio_to_total = torch.zeros((num_layer_1_neurons, num_instances))
    layer_2_instance_contrib_ratio_to_total = torch.zeros((num_layer_2_neurons, num_instances))
    layer_3_instance_contrib_ratio_to_total = torch.zeros((num_layer_3_neurons, num_instances))
    
    # Store the total neuron-wise contributions to output image
    total_layer_1_output_contrib = torch.sum(torch.abs(layer_1_output_contrib), dim=(1,2))
    total_layer_2_output_contrib = torch.sum(torch.abs(layer_2_output_contrib), dim=(1,2))
    total_layer_3_output_contrib = torch.sum(torch.abs(layer_3_output_contrib), dim=(1,2))

    for instance in instance_to_ann_id_map:
        ann_id = instance_to_ann_id_map[instance]
        ann = [ann for ann in annotations if ann['id'] == ann_id][0]
        
        area = ann['area']
        binary_mask = ann['binary_mask'].squeeze()
        
        # Use binary mask of shape hxw to index into the n_featsxhxw contribution tensor
        # to get the contribs for the current instance
        curr_instance_layer_1_contribs = torch.abs(layer_1_output_contrib[:, binary_mask])
        curr_instance_layer_2_contribs = torch.abs(layer_2_output_contrib[:, binary_mask])
        curr_instance_layer_3_contribs = torch.abs(layer_3_output_contrib[:, binary_mask])
        
        # Get aggregated total contribution for each kernel to the instance
        total_layer_1_inst_contrib = torch.sum(curr_instance_layer_1_contribs, dim=-1)
        total_layer_2_inst_contrib = torch.sum(curr_instance_layer_2_contribs, dim=-1)
        total_layer_3_inst_contrib = torch.sum(curr_instance_layer_3_contribs, dim=-1)
        avg_layer_1_contrib = total_layer_1_inst_contrib / area
        avg_layer_2_contrib = total_layer_2_inst_contrib / area
        avg_layer_3_contrib = total_layer_3_inst_contrib / area
            
        # Store the average contribution from each layer neurons to current instance
        inst_idx = instance_names.index(instance)
        layer_1_to_instance_contribs[:, inst_idx] = avg_layer_1_contrib.flatten()
        layer_2_to_instance_contribs[:, inst_idx] = avg_layer_2_contrib.flatten()
        layer_3_to_instance_contribs[:, inst_idx] = avg_layer_3_contrib.flatten()
        
        # Find delta percentages - ( true contrib - expected contrib ) / expected contrib
        layer_1_expected_instance_contrib = total_layer_1_output_contrib * (area / total_img_area)
        layer_1_instance_contrib_ratio_to_total[:, inst_idx] = torch.abs(total_layer_1_inst_contrib - layer_1_expected_instance_contrib) / layer_1_expected_instance_contrib

        layer_2_expected_instance_contrib = total_layer_2_output_contrib * (area / total_img_area)
        layer_2_instance_contrib_ratio_to_total[:, inst_idx] = torch.abs(total_layer_2_inst_contrib - layer_2_expected_instance_contrib) / layer_2_expected_instance_contrib

        layer_3_expected_instance_contrib = total_layer_3_output_contrib * (area / total_img_area)
        layer_3_instance_contrib_ratio_to_total[:, inst_idx] = torch.abs(total_layer_3_inst_contrib - layer_3_expected_instance_contrib) / layer_3_expected_instance_contrib

    return layer_1_to_instance_contribs, layer_2_to_instance_contribs, layer_3_to_instance_contribs, \
        layer_1_instance_contrib_ratio_to_total, layer_2_instance_contrib_ratio_to_total, layer_3_instance_contrib_ratio_to_total, instance_areas
        
def get_gridcell_contribs(
    layer_1_output_contrib, layer_2_output_contrib, layer_3_output_contrib, reg_stride_h, reg_stride_w
):
    total_img_area = layer_1_output_contrib.size(-2) * layer_1_output_contrib.size(-1)
    
    # Take absolute of contributions # num_neurons x cell_stride x cell_stride x h/cell_stride x w/cell_stride. e.g. k x 4 x 4 x h/4 x w/4
    unfolded_layer_1_to_gridcell_contribs = torch.abs(layer_1_output_contrib).unfold(1, reg_stride_h, reg_stride_h).unfold(2, reg_stride_w, reg_stride_w).permute(0, 3, 4, 1, 2)
    unfolded_layer_2_to_gridcell_contribs = torch.abs(layer_2_output_contrib).unfold(1, reg_stride_h, reg_stride_h).unfold(2, reg_stride_w, reg_stride_w).permute(0, 3, 4, 1, 2)
    unfolded_layer_3_to_gridcell_contribs = torch.abs(layer_3_output_contrib).unfold(1, reg_stride_h, reg_stride_h).unfold(2, reg_stride_w, reg_stride_w).permute(0, 3, 4, 1, 2)

    # Store the total neuron-wise contributions to output image
    total_layer_1_output_contrib = torch.sum(torch.abs(layer_1_output_contrib), dim=(1,2))
    total_layer_2_output_contrib = torch.sum(torch.abs(layer_2_output_contrib), dim=(1,2))
    total_layer_3_output_contrib = torch.sum(torch.abs(layer_3_output_contrib), dim=(1,2))

    gridcell_area = unfolded_layer_1_to_gridcell_contribs.size(3) * unfolded_layer_1_to_gridcell_contribs.size(4)
    
    # take absolute of contributions **after** we store our raw per-region contribs
    layer_1_to_gridcell_contribs = torch.abs(unfolded_layer_1_to_gridcell_contribs)
    layer_2_to_gridcell_contribs = torch.abs(unfolded_layer_2_to_gridcell_contribs)
    layer_3_to_gridcell_contribs = torch.abs(unfolded_layer_3_to_gridcell_contribs)
        
    # Flatten contribs by region before taking variance over pixels in region
    flattened_layer_1_gridcell_contribs = layer_1_to_gridcell_contribs.flatten(3, 4).flatten(1, 2) # num_neurons x num_gridcells x h/cell_stride*w/cell_stride
    flattened_layer_2_gridcell_contribs = layer_2_to_gridcell_contribs.flatten(3, 4).flatten(1, 2)
    flattened_layer_3_gridcell_contribs = layer_3_to_gridcell_contribs.flatten(3, 4).flatten(1, 2)


    # Find delta percentages - ( true contrib - expected contrib ) / expected contrib
    layer_1_expected_region_contrib = total_layer_1_output_contrib[:,None] * (gridcell_area / total_img_area)
    layer_1_gridcell_contrib_ratio_to_total = (torch.sum(flattened_layer_1_gridcell_contribs, dim=-1) - layer_1_expected_region_contrib) / layer_1_expected_region_contrib
    
    layer_2_expected_region_contrib = total_layer_2_output_contrib[:,None] * (gridcell_area / total_img_area)
    layer_2_gridcell_contrib_ratio_to_total = (torch.sum(flattened_layer_2_gridcell_contribs, dim=-1) - layer_2_expected_region_contrib) / layer_2_expected_region_contrib
    
    layer_3_expected_region_contrib = total_layer_3_output_contrib[:,None] * (gridcell_area / total_img_area)
    layer_3_gridcell_contrib_ratio_to_total = (torch.sum(flattened_layer_3_gridcell_contribs, dim=-1) - layer_3_expected_region_contrib) / layer_3_expected_region_contrib

    # Aggregate the maps by summing up contributions within each cell_stride x cell_stride region of size h/cell_stride and w/cell_stride    
    # num_neurons x cell_stride x cell_stride
    layer_1_to_gridcell_contribs = layer_1_to_gridcell_contribs.sum(dim=(3, 4)) / gridcell_area
    layer_2_to_gridcell_contribs = layer_2_to_gridcell_contribs.sum(dim=(3, 4)) / gridcell_area
    layer_3_to_gridcell_contribs = layer_3_to_gridcell_contribs.sum(dim=(3, 4)) / gridcell_area

    # Reshape the (cell_stride x cell_stride) dim to num_gridcells
    layer_1_feature_vectors = layer_1_to_gridcell_contribs.view(layer_1_to_gridcell_contribs.size(0), -1) # num_neurons x num_gridcells
    layer_2_feature_vectors = layer_2_to_gridcell_contribs.view(layer_2_to_gridcell_contribs.size(0), -1) # num_neurons x num_gridcells
    layer_3_feature_vectors = layer_3_to_gridcell_contribs.view(layer_3_to_gridcell_contribs.size(0), -1) # num_neurons x num_gridcells

    return layer_1_feature_vectors, layer_2_feature_vectors, layer_3_feature_vectors, \
        layer_1_gridcell_contrib_ratio_to_total, layer_2_gridcell_contrib_ratio_to_total, layer_3_gridcell_contrib_ratio_to_total

def compute_kmeans_clusters_in_rgb(image, num_clusters):
    # Reshape to 2D array of num_pixels x 3 (for rgb)
    image_reshaped_rgb = image.reshape(-1, 3)
    
    # Perform kmeans clustering
    kmeans = KMeans(n_clusters=num_clusters, n_init=1, random_state=0).fit(image_reshaped_rgb)
    rgb_cluster_map = kmeans.labels_.reshape(image.shape[0], image.shape[1])
    
    return rgb_cluster_map

# For each rgb cluster - get average contrib, total contrib and total area (other useful info too)
def get_rgb_cluster_contribs(
    layer_1_output_contrib, layer_2_output_contrib, layer_3_output_contrib, rgb_cluster_map 
):
    total_img_area = layer_1_output_contrib.size(-2) * layer_1_output_contrib.size(-1)
    
    # Maps for kernel to object contributions
    num_layer_1_neurons = layer_1_output_contrib.shape[0]
    num_layer_2_neurons = layer_2_output_contrib.shape[0]
    num_layer_3_neurons = layer_3_output_contrib.shape[0]

    n_rgb_clusters = len(np.unique(rgb_cluster_map))
    layer_1_to_rgb_cluster_contribs = torch.zeros((num_layer_1_neurons, n_rgb_clusters))
    layer_2_to_rgb_cluster_contribs = torch.zeros((num_layer_2_neurons, n_rgb_clusters))
    layer_3_to_rgb_cluster_contribs = torch.zeros((num_layer_3_neurons, n_rgb_clusters))

    rgb_cluster_areas = torch.zeros(n_rgb_clusters)
    
    # Use deltas idea to find percentage deviation of rgb cluster's actual contribution from expected
    layer_1_rgb_cluster_contrib_ratio_to_total = torch.zeros((num_layer_1_neurons, n_rgb_clusters))
    layer_2_rgb_cluster_contrib_ratio_to_total = torch.zeros((num_layer_2_neurons, n_rgb_clusters))
    layer_3_rgb_cluster_contrib_ratio_to_total = torch.zeros((num_layer_3_neurons, n_rgb_clusters))
    
    # Store the total neuron-wise contributions to output image
    total_layer_1_output_contrib = torch.sum(torch.abs(layer_1_output_contrib), dim=(1,2))
    total_layer_2_output_contrib = torch.sum(torch.abs(layer_2_output_contrib), dim=(1,2))
    total_layer_3_output_contrib = torch.sum(torch.abs(layer_3_output_contrib), dim=(1,2))

    for cluster_id in np.unique(rgb_cluster_map):
        
        # Construct a binary mask of shape hxw for the current rgb cluster
        binary_mask = (rgb_cluster_map == cluster_id)
        binary_mask = binary_mask.squeeze().astype(bool)
        area = binary_mask.sum()
        
        # Use binary mask of shape hxw to index into the n_featsxhxw contribution tensor
        # to get the contribs for the current rgb cluster
        curr_rgb_cluster_layer_1_contribs = torch.abs(layer_1_output_contrib[:, binary_mask])
        curr_rgb_cluster_layer_2_contribs = torch.abs(layer_2_output_contrib[:, binary_mask])
        curr_rgb_cluster_layer_3_contribs = torch.abs(layer_3_output_contrib[:, binary_mask])
        
        # Get aggregated total contribution for each kernel to the superpixel
        total_layer_1_spix_contrib = torch.sum(curr_rgb_cluster_layer_1_contribs, dim=-1)
        total_layer_2_spix_contrib = torch.sum(curr_rgb_cluster_layer_2_contribs, dim=-1)
        total_layer_3_spix_contrib = torch.sum(curr_rgb_cluster_layer_3_contribs, dim=-1)
        avg_layer_1_contrib = total_layer_1_spix_contrib / area
        avg_layer_2_contrib = total_layer_2_spix_contrib / area
        avg_layer_3_contrib = total_layer_3_spix_contrib / area
            
        # Store the average contribution from each layer neurons to current rgb cluster
        layer_1_to_rgb_cluster_contribs[:, cluster_id] = avg_layer_1_contrib.flatten()
        layer_2_to_rgb_cluster_contribs[:, cluster_id] = avg_layer_2_contrib.flatten()
        layer_3_to_rgb_cluster_contribs[:, cluster_id] = avg_layer_3_contrib.flatten()
        
        # Find delta percentages -> ( true contrib - expected contrib ) / expected contrib
        layer_1_expected_rgb_cluster_contrib = total_layer_1_output_contrib * (area / total_img_area)
        layer_1_rgb_cluster_contrib_ratio_to_total[:, cluster_id] = torch.abs(total_layer_1_spix_contrib - layer_1_expected_rgb_cluster_contrib) / layer_1_expected_rgb_cluster_contrib

        layer_2_expected_rgb_cluster_contrib = total_layer_2_output_contrib * (area / total_img_area)
        layer_2_rgb_cluster_contrib_ratio_to_total[:, cluster_id] = torch.abs(total_layer_2_spix_contrib - layer_2_expected_rgb_cluster_contrib) / layer_2_expected_rgb_cluster_contrib

        layer_3_expected_rgb_cluster_contrib = total_layer_3_output_contrib * (area / total_img_area)
        layer_3_rgb_cluster_contrib_ratio_to_total[:, cluster_id] = torch.abs(total_layer_3_spix_contrib - layer_3_expected_rgb_cluster_contrib) / layer_3_expected_rgb_cluster_contrib

    return layer_1_to_rgb_cluster_contribs, layer_2_to_rgb_cluster_contribs, layer_3_to_rgb_cluster_contribs, \
        layer_1_rgb_cluster_contrib_ratio_to_total, layer_2_rgb_cluster_contrib_ratio_to_total, layer_3_rgb_cluster_contrib_ratio_to_total, rgb_cluster_areas

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

# For each gabor cluster - get average contrib, total contrib and total area (other useful info too)
def get_gabor_cluster_contribs(
    layer_1_output_contrib, layer_2_output_contrib, layer_3_output_contrib, gabor_cluster_map 
):
    total_img_area = layer_1_output_contrib.size(-2) * layer_1_output_contrib.size(-1)
    
    # Maps for kernel to object contributions
    num_layer_1_neurons = layer_1_output_contrib.shape[0]
    num_layer_2_neurons = layer_2_output_contrib.shape[0]
    num_layer_3_neurons = layer_3_output_contrib.shape[0]

    n_gabor_clusters = len(np.unique(gabor_cluster_map))
    layer_1_to_gabor_cluster_contribs = torch.zeros((num_layer_1_neurons, n_gabor_clusters))
    layer_2_to_gabor_cluster_contribs = torch.zeros((num_layer_2_neurons, n_gabor_clusters))
    layer_3_to_gabor_cluster_contribs = torch.zeros((num_layer_3_neurons, n_gabor_clusters))

    gabor_cluster_areas = torch.zeros(n_gabor_clusters)
    
    # Use deltas idea to find percentage deviation of gabor cluster's actual contribution from expected
    layer_1_gabor_cluster_contrib_ratio_to_total = torch.zeros((num_layer_1_neurons, n_gabor_clusters))
    layer_2_gabor_cluster_contrib_ratio_to_total = torch.zeros((num_layer_2_neurons, n_gabor_clusters))
    layer_3_gabor_cluster_contrib_ratio_to_total = torch.zeros((num_layer_3_neurons, n_gabor_clusters))
    
    # Store the total neuron-wise contributions to output image
    total_layer_1_output_contrib = torch.sum(torch.abs(layer_1_output_contrib), dim=(1,2))
    total_layer_2_output_contrib = torch.sum(torch.abs(layer_2_output_contrib), dim=(1,2))
    total_layer_3_output_contrib = torch.sum(torch.abs(layer_3_output_contrib), dim=(1,2))

    for cluster_id in np.unique(gabor_cluster_map):
        
        # Construct a binary mask of shape hxw for the current rgb cluster
        binary_mask = (gabor_cluster_map == cluster_id)
        binary_mask = binary_mask.squeeze().astype(bool)
        area = binary_mask.sum()
        
        # Use binary mask of shape hxw to index into the n_featsxhxw contribution tensor
        # to get the contribs for the current gabor cluster
        curr_gabor_cluster_layer_1_contribs = torch.abs(layer_1_output_contrib[:, binary_mask])
        curr_gabor_cluster_layer_2_contribs = torch.abs(layer_2_output_contrib[:, binary_mask])
        curr_gabor_cluster_layer_3_contribs = torch.abs(layer_3_output_contrib[:, binary_mask])
        
        # Get aggregated total contribution for each kernel to the superpixel
        total_layer_1_gabor_clust_contrib = torch.sum(curr_gabor_cluster_layer_1_contribs, dim=-1)
        total_layer_2_gabor_clust_contrib = torch.sum(curr_gabor_cluster_layer_2_contribs, dim=-1)
        total_layer_3_gabor_clust_contrib = torch.sum(curr_gabor_cluster_layer_3_contribs, dim=-1)
        avg_layer_1_contrib = total_layer_1_gabor_clust_contrib / area
        avg_layer_2_contrib = total_layer_2_gabor_clust_contrib / area
        avg_layer_3_contrib = total_layer_3_gabor_clust_contrib / area
            
        # Store the average contribution from each layer neurons to current gabor cluster
        layer_1_to_gabor_cluster_contribs[:, cluster_id] = avg_layer_1_contrib.flatten()
        layer_2_to_gabor_cluster_contribs[:, cluster_id] = avg_layer_2_contrib.flatten()
        layer_3_to_gabor_cluster_contribs[:, cluster_id] = avg_layer_3_contrib.flatten()
        
        # Find delta percentages -> ( true contrib - expected contrib ) / expected contrib
        layer_1_expected_gabor_cluster_contrib = total_layer_1_output_contrib * (area / total_img_area)
        layer_1_gabor_cluster_contrib_ratio_to_total[:, cluster_id] = torch.abs(total_layer_1_gabor_clust_contrib - layer_1_expected_gabor_cluster_contrib) / layer_1_expected_gabor_cluster_contrib

        layer_2_expected_gabor_cluster_contrib = total_layer_2_output_contrib * (area / total_img_area)
        layer_2_gabor_cluster_contrib_ratio_to_total[:, cluster_id] = torch.abs(total_layer_2_gabor_clust_contrib - layer_2_expected_gabor_cluster_contrib) / layer_2_expected_gabor_cluster_contrib

        layer_3_expected_gabor_cluster_contrib = total_layer_3_output_contrib * (area / total_img_area)
        layer_3_gabor_cluster_contrib_ratio_to_total[:, cluster_id] = torch.abs(total_layer_3_gabor_clust_contrib - layer_3_expected_gabor_cluster_contrib) / layer_3_expected_gabor_cluster_contrib

    return layer_1_to_gabor_cluster_contribs, layer_2_to_gabor_cluster_contribs, layer_3_to_gabor_cluster_contribs, \
        layer_1_gabor_cluster_contrib_ratio_to_total, layer_2_gabor_cluster_contrib_ratio_to_total, layer_3_gabor_cluster_contrib_ratio_to_total, gabor_cluster_areas

def compute_inference_results(single_image_dataloader, ffn_model, cfg, categories_dict, num_rgb_clusters, num_gabor_clusters):
    with torch.no_grad():
        batch = next(iter(single_image_dataloader))

    data = batch['data'].cuda()
    N,C,H,W = data.shape
    annotations = convert_tensor_annotations_to_numpy(batch['annotations'])
    annotations = add_other_annotation(annotations)

    features = batch['features'].squeeze().cuda()
    features_shape = batch['features_shape'].squeeze().tolist()
    reshape = True

    proc = data_process.DataProcessor(cfg.data, device='cpu')
    x = batch['data']
    coords = proc.get_coordinates(data_shape=features_shape,patch_shape=cfg.data.patch_shape,\
                                    split=cfg.data.coord_split,normalize_range=cfg.data.coord_normalize_range)
    coords = coords.to(x).cuda()

    # Create a dictionary to store the intermediate decoder_results from each seeded model, over time.
    inference_results = {}
    kwargs = {}
    with torch.no_grad():
        out = ffn_model(coords, img=data)
        pred = out['predicted']
        intermediate_results = out["intermediate_results"]
        
        if reshape:
            # This reshapes the prediction into an image
            pred = proc.process_outputs(
                pred,input_img_shape=batch['data_shape'].squeeze().tolist(),\
                features_shape=features_shape,\
                patch_shape=cfg.data.patch_shape)

    # Compute superpixels on the image
    image_numpy = data[0].permute(1,2,0).cpu().numpy()

    rgb_cluster_map = compute_kmeans_clusters_in_rgb(image_numpy, num_rgb_clusters)

    # Compute Gabor clusters map
    image_pil_format = (data[0].clamp(0,1) * 255).permute(1,2,0).cpu().numpy().astype(np.uint8)
    gabor_cluster_map = get_gabor_label_map(image_pil_format, num_gabor_clusters)

    inference_results = {
        "data": batch["data"],
        "pred": pred,
        "annotations": annotations,
        "img_hw": (H,W),
        "intermediate_results": intermediate_results,
        "rgb_cluster_map": rgb_cluster_map,
        "gabor_cluster_map": gabor_cluster_map

    }
    
    categories_in_frame = {}
    for ann in annotations:
        if ann["category_id"] not in categories_in_frame:
            categories_in_frame[ann["category_id"]] = categories_dict[ann["category_id"]]

    categories_in_frame[-1] = categories_dict[-1]
    object_categories = [v['name'] for k, v in categories_in_frame.items()]
    categories_in_frame = [v for k, v in categories_in_frame.items()]
    
    return inference_results, categories_in_frame, object_categories

def compute_all_variables_for_image(inference_results, ffn_model, instance_to_ann_id_map, cell_stride_h, cell_stride_w, 
                                    instance_names):
    intermediate_results = inference_results["intermediate_results"]
    (H,W) = inference_results["img_hw"]
    annotations = inference_results["annotations"]

    all_variables_for_image = {}
    
    num_regions = cell_stride_h * cell_stride_w
    
    # for img_idx, value in inference_results.items():
    pred = inference_results["pred"]
    data = inference_results["data"]
    intermediate_results = inference_results["intermediate_results"]
    
    # superpixel_map = inference_results["superpixel_map"]
    rgb_cluster_map = inference_results["rgb_cluster_map"]
    gabor_cluster_map = inference_results["gabor_cluster_map"]

    # Get model contributions
    compute_contrib_obj = ComputeMLPContributions(
        ffn_model, intermediate_results, (H,W)
    )

    layer_1_output_contrib, layer_2_output_contrib, layer_3_output_contrib, _, _, _ = compute_contrib_obj.compute_all_layer_mappings()

    # Get instance contributions
    layer_1_to_instance_contribs, layer_2_to_instance_contribs, layer_3_to_instance_contribs, \
        layer_1_instance_contrib_ratio_to_total, layer_2_instance_contrib_ratio_to_total, layer_3_instance_contrib_ratio_to_total, instance_areas \
            = get_instance_contribs(layer_1_output_contrib, layer_2_output_contrib, layer_3_output_contrib, annotations, instance_to_ann_id_map, instance_names)

    # Get gridcell contributions
    layer_1_to_gridcell_contribs, layer_2_to_gridcell_contribs, layer_3_to_gridcell_contribs, \
        layer_1_gridcell_contrib_ratio_to_total, layer_2_gridcell_contrib_ratio_to_total, layer_3_gridcell_contrib_ratio_to_total \
            = get_gridcell_contribs(layer_1_output_contrib, layer_2_output_contrib, layer_3_output_contrib, cell_stride_h, cell_stride_w)
        
    # Get RGB kmeans clustered contributions and normalize them. Also get variances within RGB clusters.
    layer_1_to_rgb_cluster_contribs, layer_2_to_rgb_cluster_contribs, layer_3_to_rgb_cluster_contribs, \
        layer_1_rgb_cluster_contrib_ratio_to_total, layer_2_rgb_cluster_contrib_ratio_to_total, layer_3_rgb_cluster_contrib_ratio_to_total, rgb_cluster_areas \
            = get_rgb_cluster_contribs(layer_1_output_contrib, layer_2_output_contrib, layer_3_output_contrib, rgb_cluster_map)
            
    # Get RGB kmeans clustered contributions and normalize them. Also get variances within RGB clusters.
    layer_1_to_gabor_cluster_contribs, layer_2_to_gabor_cluster_contribs, layer_3_to_gabor_cluster_contribs, \
        layer_1_gabor_cluster_contrib_ratio_to_total, layer_2_gabor_cluster_contrib_ratio_to_total, layer_3_gabor_cluster_contrib_ratio_to_total, gabor_cluster_areas \
            = get_gabor_cluster_contribs(layer_1_output_contrib, layer_2_output_contrib, layer_3_output_contrib, gabor_cluster_map)

    # Beware, some of the neurons in MLP are dead (all 0 contribs). These are removed in normalization
    all_variables_for_image = {
        # "instance_names": instance_names,
        "layer_1_output_contrib": torch.abs(layer_1_output_contrib),
        "layer_2_output_contrib": torch.abs(layer_2_output_contrib),
        "layer_3_output_contrib": torch.abs(layer_3_output_contrib),

        # areas
        "instance_areas": instance_areas,
        # "superpixel_areas": superpixel_areas,
        "rgb_cluster_areas": rgb_cluster_areas,
        "gabor_cluster_areas": gabor_cluster_areas,
        
        # per-patch contribution ratios
        "layer_3_instance_contrib_ratio_to_total": layer_3_instance_contrib_ratio_to_total,
        "layer_2_instance_contrib_ratio_to_total": layer_2_instance_contrib_ratio_to_total,
        "layer_1_instance_contrib_ratio_to_total": layer_1_instance_contrib_ratio_to_total,
        
        "layer_3_gridcell_contrib_ratio_to_total": layer_3_gridcell_contrib_ratio_to_total,
        "layer_2_gridcell_contrib_ratio_to_total": layer_2_gridcell_contrib_ratio_to_total,
        "layer_1_gridcell_contrib_ratio_to_total": layer_1_gridcell_contrib_ratio_to_total,
        
        "layer_3_rgb_cluster_contrib_ratio_to_total": layer_3_rgb_cluster_contrib_ratio_to_total,
        "layer_2_rgb_cluster_contrib_ratio_to_total": layer_2_rgb_cluster_contrib_ratio_to_total,
        "layer_1_rgb_cluster_contrib_ratio_to_total": layer_1_rgb_cluster_contrib_ratio_to_total,

        "layer_3_gabor_cluster_contrib_ratio_to_total": layer_3_gabor_cluster_contrib_ratio_to_total,
        "layer_2_gabor_cluster_contrib_ratio_to_total": layer_2_gabor_cluster_contrib_ratio_to_total,
        "layer_1_gabor_cluster_contrib_ratio_to_total": layer_1_gabor_cluster_contrib_ratio_to_total,
        
        "num_instances_in_frame": len(instance_areas),
    }

    return all_variables_for_image

def compute_variance_of_deltas(all_variables_for_image):

    num_instances_in_frame = all_variables_for_image["num_instances_in_frame"]
    
    fig, axs = plt.subplots(1, 3, figsize=(12,8), tight_layout=True)
    
    
    layer_3_instance_variances = torch.var(all_variables_for_image["layer_3_instance_contrib_ratio_to_total"], dim=-1)
    layer_3_gridcell_variances = torch.var(all_variables_for_image["layer_3_gridcell_contrib_ratio_to_total"], dim=-1)
    layer_3_rgb_cluster_variances = torch.var(all_variables_for_image["layer_3_rgb_cluster_contrib_ratio_to_total"], dim=-1)
    layer_3_gabor_cluster_variances = torch.var(all_variables_for_image["layer_3_gabor_cluster_contrib_ratio_to_total"], dim=-1)
    
    layer_2_instance_variances = torch.var(all_variables_for_image["layer_2_instance_contrib_ratio_to_total"], dim=-1)
    layer_2_gridcell_variances = torch.var(all_variables_for_image["layer_2_gridcell_contrib_ratio_to_total"], dim=-1)
    layer_2_rgb_cluster_variances = torch.var(all_variables_for_image["layer_2_rgb_cluster_contrib_ratio_to_total"], dim=-1)
    layer_2_gabor_cluster_variances = torch.var(all_variables_for_image["layer_2_gabor_cluster_contrib_ratio_to_total"], dim=-1)
    
    layer_1_instance_variances = torch.var(all_variables_for_image["layer_1_instance_contrib_ratio_to_total"], dim=-1)
    layer_1_gridcell_variances = torch.var(all_variables_for_image["layer_1_gridcell_contrib_ratio_to_total"], dim=-1) 
    layer_1_rgb_cluster_variances = torch.var(all_variables_for_image["layer_1_rgb_cluster_contrib_ratio_to_total"], dim=-1) 
    layer_1_gabor_cluster_variances = torch.var(all_variables_for_image["layer_1_gabor_cluster_contrib_ratio_to_total"], dim=-1)
    
    sorted_variance_layer_3_instance_contrib_ratio, layer_3_instance_sorted_indices = torch.sort(layer_3_instance_variances)
    sorted_variance_layer_3_gridcell_contrib_ratio, layer_3_gridcell_sorted_indices = torch.sort(layer_3_gridcell_variances)
    sorted_variance_layer_3_rgb_cluster_contrib_ratio, layer_3_rgb_cluster_sorted_indices = torch.sort(layer_3_rgb_cluster_variances)
    sorted_variance_layer_3_gabor_cluster_contrib_ratio, layer_3_gabor_cluster_sorted_indices = torch.sort(layer_3_gabor_cluster_variances)
    
    sorted_variance_layer_2_instance_contrib_ratio, layer_2_instance_sorted_indices = torch.sort(layer_2_instance_variances)
    sorted_variance_layer_2_gridcell_contrib_ratio, layer_2_gridcell_sorted_indices = torch.sort(layer_2_gridcell_variances)
    sorted_variance_layer_2_rgb_cluster_contrib_ratio, layer_2_rgb_cluster_sorted_indices = torch.sort(layer_2_rgb_cluster_variances)
    sorted_variance_layer_2_gabor_cluster_contrib_ratio, layer_2_gabor_cluster_sorted_indices = torch.sort(layer_2_gabor_cluster_variances)
    
    sorted_variance_layer_1_instance_contrib_ratio, layer_1_instance_sorted_indices = torch.sort(layer_1_instance_variances)
    sorted_variance_layer_1_gridcell_contrib_ratio, layer_1_gridcell_sorted_indices = torch.sort(layer_1_gridcell_variances)
    sorted_variance_layer_1_rgb_cluster_contrib_ratio, layer_1_rgb_cluster_sorted_indices = torch.sort(layer_1_rgb_cluster_variances)
    sorted_variance_layer_1_gabor_cluster_contrib_ratio, layer_1_gabor_cluster_sorted_indices = torch.sort(layer_1_gabor_cluster_variances)
    
    labels = ["Instances variance", "Grid cells variance", "RGB Cluster variance", "Gabor Cluster variance"]
    colors = ['r', 'g', 'b', 'm']
    
    # Plot layer 3
    for idx, var in enumerate([sorted_variance_layer_3_instance_contrib_ratio, sorted_variance_layer_3_gridcell_contrib_ratio, sorted_variance_layer_3_rgb_cluster_contrib_ratio, sorted_variance_layer_3_gabor_cluster_contrib_ratio]):
        axs[0].plot(var, label=labels[idx], c=colors[idx])        
    axs[0].set_title(f"Layer 3")
    
    # Plot layer 2
    for idx, var in enumerate([sorted_variance_layer_2_instance_contrib_ratio, sorted_variance_layer_2_gridcell_contrib_ratio, sorted_variance_layer_2_rgb_cluster_contrib_ratio, sorted_variance_layer_2_gabor_cluster_contrib_ratio]):
        axs[1].plot(var, label=labels[idx], c=colors[idx])
    axs[1].set_title(f"Layer 2")
    
    # Plot layer 1
    for idx, var in enumerate([sorted_variance_layer_1_instance_contrib_ratio, sorted_variance_layer_1_gridcell_contrib_ratio, sorted_variance_layer_1_rgb_cluster_contrib_ratio, sorted_variance_layer_1_gabor_cluster_contrib_ratio]):
        axs[2].plot(var, label=labels[idx], c=colors[idx])
    axs[2].set_title(f"Layer 1")
        
    # Every subplot has the same legend, let us pick one 
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.05))
    
    sorted_deltas_dict = {
            "layer_1": {
                "instances_deltas": sorted_variance_layer_1_instance_contrib_ratio,
                "gridcells_deltas": sorted_variance_layer_1_gridcell_contrib_ratio,
                "rgb_clusters_deltas": sorted_variance_layer_1_rgb_cluster_contrib_ratio,
                "gabor_clusters_deltas": sorted_variance_layer_1_gabor_cluster_contrib_ratio
            }, "layer_2": {
                "instances_deltas": sorted_variance_layer_2_instance_contrib_ratio,
                "gridcells_deltas": sorted_variance_layer_2_gridcell_contrib_ratio,
                "rgb_clusters_deltas": sorted_variance_layer_2_rgb_cluster_contrib_ratio,
                "gabor_clusters_deltas": sorted_variance_layer_2_gabor_cluster_contrib_ratio
            }, "layer_3": {
                "instances_deltas": sorted_variance_layer_3_instance_contrib_ratio,
                "gridcells_deltas": sorted_variance_layer_3_gridcell_contrib_ratio,
                "rgb_clusters_deltas": sorted_variance_layer_3_rgb_cluster_contrib_ratio,
                "gabor_clusters_deltas": sorted_variance_layer_3_gabor_cluster_contrib_ratio
            }, "sorted_indices": {
                "layer_1": {
                    "instances": layer_1_instance_sorted_indices,
                    "gridcells": layer_1_gridcell_sorted_indices,
                    "rgb_clusters": layer_1_rgb_cluster_sorted_indices,
                    "gabor_clusters": layer_1_gabor_cluster_sorted_indices
                }, "lsyer_2": {
                    "instances": layer_2_instance_sorted_indices,
                    "gridcells": layer_2_gridcell_sorted_indices,
                    "rgb_clusters": layer_2_rgb_cluster_sorted_indices,
                    "gabor_clusters": layer_2_gabor_cluster_sorted_indices
                }, "layer_3": {
                    "instances": layer_3_instance_sorted_indices,
                    "gridcells": layer_3_gridcell_sorted_indices,
                    "rgb_clusters": layer_3_rgb_cluster_sorted_indices,
                    "gabor_clusters": layer_3_gabor_cluster_sorted_indices
                }
            }
        }

    return sorted_deltas_dict



def main():
    # Multiple videos
    dataset_names = ['cityscapes', 'vipseg']
    # Choose videos with 30ish instances at least
    vidnames = {
        'cityscapes': ["0495", "0065",],
        'vipseg': ["1327_e96s6pJ5x3Y", "127_-hIVCYO4C90", "2350_sCLtK1a2GGc", ]
    }
    

    vid_data_folder_name = {
        "cityscapes": "Cityscapes_VPS_models",
        "vipseg": "VIPSeg_models"
    }


    cfg_dict = {}
    dataloader_dict = {}
    weights_dict = {}
    ffn_models_dict = {}
    categories_dicts = {}
    
    for dataset_name in dataset_names:
        weights_dict[dataset_name] = {}
        cfg_dict[dataset_name] = {}
        ffn_models_dict[dataset_name] = {}
        categories_dicts[dataset_name] = {}

        for vidname in vidnames[dataset_name]:
            vid_data_folder = vid_data_folder_name[dataset_name]
            weights_dict[dataset_name][vidname] = f'output/{vid_data_folder}/{vidname}/{vidname}_framenum_0_128_256'
            
            cfg, categories_dict = load_cfg(weights_dict[dataset_name][vidname], dataset_name, vidname)
            cfg_dict[dataset_name][vidname] = cfg
            categories_dicts[dataset_name][vidname] = categories_dict
            
            
            ffn_models_dict[dataset_name][vidname] = load_model(cfg)
            
    for dataset_name in dataset_names:
        dataloader_dict[dataset_name] = {}
        
        for vidname in vidnames[dataset_name]:
            single_image_dataloader = get_loader(cfg_dict[dataset_name][vidname], dataset_name)
            
            dataloader_dict[dataset_name][vidname] = single_image_dataloader
            
            
    per_vid_patch_deltas_var_dict = {}

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
            single_image_dataloader = dataloader_dict[dataset_name][vidname]
            ffn_model = ffn_models_dict[dataset_name][vidname]
            categories_dict = categories_dicts[dataset_name][vidname]
            cfg = cfg_dict[dataset_name][vidname]

            categories = list(categories_dict.values())
            
            num_rgb_clusters = num_rgb_and_gabor_clusters_dict[vidname]
            num_gabor_clusters = num_rgb_and_gabor_clusters_dict[vidname]
            cell_stride_h, cell_stride_w = cell_stride_h_dict[vidname], cell_stride_w_dict[vidname]
                        
            inference_results, categories_in_frame, object_categories = compute_inference_results(
                single_image_dataloader, ffn_model, cfg, categories_dict, num_rgb_clusters, num_gabor_clusters
            )

            inst_id_to_cat_and_inst_suffix, instance_to_ann_id_map, instance_names, object_to_instances_map, \
                obj_to_obj_name_idx, instance_names = get_instance_info(inference_results, object_categories, categories)

            all_variables_for_image = compute_all_variables_for_image(
                inference_results, ffn_model, instance_to_ann_id_map, cell_stride_h, cell_stride_w,
                instance_names
            )
            
            print(vidname, all_variables_for_image["num_instances_in_frame"])

            sorted_deltas_dict = compute_variance_of_deltas(all_variables_for_image)

            # For optical flow stuff, it might be easiest to 
            per_vid_patch_deltas_var_dict[vidname] = {
                "sorted_deltas_dict" : sorted_deltas_dict,
                "cluster_info": {
                    "num_instances": all_variables_for_image["num_instances_in_frame"],
                    "num_rgb_clusters": num_rgb_clusters,
                    "num_gabor_clusters": num_gabor_clusters,
                    "cell_stride_h": cell_stride_h,
                    "cell_stride_w": cell_stride_w
                }
            }

    save_dir = '../plotting_source_data/supplementary/MLP/C-INRs_perhaps_care_about_objects'
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, f"per_vid_patch_deltas_var_dict.pkl"), 'wb') as f:
        pickle.dump(per_vid_patch_deltas_var_dict, f)
    
if __name__ == "__main__":
    main()