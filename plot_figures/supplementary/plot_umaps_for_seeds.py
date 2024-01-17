import glob
import os
from importlib import reload

import cluster_utils
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib import cm
from sklearn.cluster import KMeans

reload(cluster_utils)
import warnings

from cluster_utils import get_kernel_vecs_from_layer, load_stuff

warnings.filterwarnings("ignore")

def draw_umap(sample_points, ax, data, data_ids, n_neighbors=15, min_dist=0.1, n_components=2, 
              metric='euclidean', title='', num_clusters=5, kmeans=False, axis_off=False):
    data_ids = np.array(data_ids).copy()
    if n_neighbors is None:
        fit = umap.UMAP()
    else:
        fit = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=42
        )
    u = fit.fit_transform(data)
    all_ids = np.arange(len(data_ids))
    cmap = cm.get_cmap('viridis')
    norm = cm.colors.Normalize(vmin=np.min(data_ids), vmax=np.max(data_ids))
    if len(np.unique(data_ids)) == 2 or len(np.unique(data_ids)) == 1:
        norm = cm.colors.Normalize(vmin=0, vmax=20)
        data_ids[data_ids == 0] = 15
        data_ids[data_ids == 1] = 10
    ids_to_take = []
    # import pdb; pdb.set_trace()

    ax.scatter(u[:,0], u[:,1], c=cmap(norm(data_ids)),  s=3, alpha=0.75)

    if kmeans:
        kmean_labels = KMeans(n_clusters=num_clusters, 
                              random_state=0).fit_predict(u)
        cluster_colours = np.random.rand(num_clusters, 3)
        for cluster_id in range(num_clusters):
            cluster_data = u[kmean_labels == cluster_id]
            cluster_id_to_take = np.random.choice(
                all_ids[kmean_labels == cluster_id], 5)
            ids_to_take.extend(list(cluster_id_to_take))
            
            center = np.mean(cluster_data, axis=0)
            radius = np.max(np.linalg.norm(cluster_data - center, axis=1))
            circle = patches.Circle(center, radius=radius,
                                    edgecolor=cluster_colours[cluster_id], 
                                    facecolor='none')
            # ax.add_patch(circle)
    
    # Set x, y ticks and labels off
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return u, kmean_labels, ax

def get_all_data( dataset_name, vid_name, inr_types, base_path, pixel_clusters):
    all_data = {}

    # for dataset_name, vid_name in dataset_vid_name_pairs:
    img_path = None
    label_maps = None
    vid_data = {}
    for inr_type in inr_types:
        vid_path = base_path.format(inr_type, dataset_name, vid_name)
        if img_path is None:
            img_path = glob.glob(os.path.join(vid_path, '*.png'))
            if len(img_path)==0:
                img_path = glob.glob(os.path.join(vid_path, '*.jpg'))
            img_path = img_path[0]
        vid_data[inr_type] = load_stuff(vid_path, pixel_clusters=pixel_clusters, 
                                        img_path=img_path, label_maps=label_maps)
        label_maps = vid_data[inr_type]['label_maps']
    all_data[f'{dataset_name}'] = vid_data
    
    return all_data
    
def get_final_stuff(all_data, dataset_name, inr_types,):
    final_stuff = {}
    
    vid_data = all_data[dataset_name]
    
    final_stuff[dataset_name] = {}
    for inr_type in inr_types:
        if inr_type == 'HNeRV':
            mlp = False
        else:
            mlp = True
        head_dict = vid_data[inr_type]['head_dict']
        vid_label_maps = vid_data[inr_type]['label_maps']
        # Drop dead neurons as necessary
        vecs = get_kernel_vecs_from_layer(head_dict, vid_label_maps, frame_no=0,
                                        mlp=mlp, kernel_size=16, drop_dead_neurons=True)
        final_stuff[dataset_name][inr_type] = vecs
    
    return final_stuff

def get_sample_img_frames(vidname, dataset_name, num_samples=5):
    if dataset_name == "Cityscapes_VPS_models":
        # the frame for training cityscapes is the first frame with provided mask/annotation
        cityscapes_vps_root = "../../data/cityscapes_vps"
        split = "val"
        panoptic_video_mask_dir = os.path.join(cityscapes_vps_root, split, "panoptic_video")
        
        data_path = os.path.join(cityscapes_vps_root, split, "img_all")
        
        rgb_mask_names = [file for file in sorted(os.listdir(panoptic_video_mask_dir)) if file.startswith(vidname)]
        # pick equally spaced frames, getting num_samples in total
        rgb_mask_names = rgb_mask_names[::len(rgb_mask_names)//num_samples]

        # get RGB image corresponding to first mask above
        img_paths = [os.path.join(data_path, "_".join(rgb_mask_name.split("_")[2:5]) + "_leftImg8bit.png") for rgb_mask_name in rgb_mask_names]
        return img_paths
    
    elif dataset_name == "VIPSeg_models":
        # all vipseg frames are annotated, we will take the first frame
        
        VIPSeg_720P_root = '../../data/VIPSeg-Dataset/VIPSeg/VIPSeg_720P'
        
        vid_path = os.path.join(VIPSeg_720P_root, "images", vidname)
        # pick equally spaced frames, getting num_samples in total
        img_paths = [os.path.join(vid_path, frame) for frame in sorted(os.listdir(vid_path))]
        img_paths = img_paths[::len(img_paths)//num_samples]
        
        return img_paths
    
def main():
    inr_types = ['image_inr', 'HNeRV']

    cityscapes_vids = ["0155"]
    cityscapes_vid_pairs = [("Cityscapes_VPS_models", vid) for vid in cityscapes_vids]
    vipseg_vids = ["20_o-wWIdQ1H98", "604_gFFyhdGeQ1A", "1327_e96s6pJ5x3Y", "2350_sCLtK1a2GGc"]
    vipseg_vid_pairs = [("VIPSeg_models", vid) for vid in vipseg_vids]

    dataset_vid_name_pairs = cityscapes_vid_pairs + vipseg_vid_pairs
    
    frame_no = 0
    pixel_clusters = 32
    base_path = '../../{}/output/{}/similarity/contributions/{}'
    
    layer_names = {
        "HNeRV": ['head_layer_output_contrib', 'nerv_blk_3_output_contrib'],
        "image_inr": ['layer_3_output_contrib', 'layer_2_output_contrib', 'layer_1_output_contrib']
    }
    num_neighbors = {
        'head_layer_output_contrib': 5,
        'nerv_blk_3_output_contrib': 25,
        'layer_3_output_contrib': 5,
        'layer_2_output_contrib': 5,
        'layer_1_output_contrib': 5
    }
    layer_titles = {
        'head_layer_output_contrib': 'NeRV Head layer',
        'nerv_blk_3_output_contrib': 'NeRV Block 3',
        'layer_3_output_contrib': 'FFN Layer 3',
        'layer_2_output_contrib': 'FFN Layer 2',
        'layer_1_output_contrib': 'FFN Layer 1'
    }
    
    seeds_to_consider = [1, 10, 20, 30, 40]
    data_id_label_type = 'seed'
    colors = np.random.rand(len(seeds_to_consider), 3)
    colors = np.hstack([colors, np.ones((len(seeds_to_consider), 1))* 0.25])
    
    # Plot UMAPs for each layer for each video (each UMAP with multiple seeds)
    fig, axs = plt.subplots(len(cityscapes_vid_pairs + vipseg_vid_pairs), 6, gridspec_kw={'width_ratios': [1.1, 1, 1, 1, 1, 1]} ,figsize=(16, 9), tight_layout=True)

    for vid_idx, (dataset_name, vid_name) in enumerate(dataset_vid_name_pairs):
        print(dataset_name, vid_name)
        
        num_samples = 6
        img_paths = get_sample_img_frames(vid_name, dataset_name, num_samples=num_samples)
        # Plot first frame of the video
        
        rgb_img = plt.imread(img_paths[0])
        # if dataset is "vipseg", center crop the 720x1280 image to 640x1280
        if dataset_name == "VIPSeg_models":
            rgb_img = rgb_img[40:680, :]
            
        axs[vid_idx, 0].imshow(rgb_img)
        axs[vid_idx, 0].axis('off')
        
        all_data = get_all_data(dataset_name, vid_name, inr_types, base_path, pixel_clusters)
        
        final_stuff = get_final_stuff(all_data, dataset_name, inr_types)
        
        inrs_to_consider = ['image_inr', 'HNeRV']
        dataset_to_use = dataset_name
            
        # Perform the clustering for each layer (of FFN + NeRV) for the current video
        for inr_idx, inr_type in enumerate(inrs_to_consider):
            
            for layer_idx, layer_name_to_take in enumerate(layer_names[inr_type]):
                col_num_in_plot = 1 + inr_idx * (5 - len(layer_names[inr_type])) + layer_idx            
            
                data = []
                data_ids = []
                all_kernels = []
            
                for seed_idx, seed in enumerate(seeds_to_consider):
                    all_kernels.append(final_stuff[dataset_to_use][inr_type][seed][layer_name_to_take]['kernels'])
                    
                    seed_data = final_stuff[dataset_to_use][inr_type][seed][layer_name_to_take]['gabor'].copy()
                    data.append(seed_data)
                    if data_id_label_type == 'inr_type':
                        data_ids.extend([inr_idx]*len(seed_data))
                    else:
                        data_ids.extend([seed_idx]*len(seed_data))
                
                # uMAP for multiple seeds
                data = np.concatenate(data, axis=0)
                data_ids = np.array(data_ids)
                all_kernels = np.concatenate(all_kernels, axis=0)
                
                neighbours = num_neighbors[layer_name_to_take]
                num_clusters = 6
                # For block 3, there are too many points so plot only a subset
                sample_points = True if layer_name_to_take == 'nerv_blk_3_output_contrib' else False
                cluster_point_info, cluster_id_data, axs[vid_idx, col_num_in_plot] = draw_umap(
                    sample_points, axs[vid_idx, col_num_in_plot], data, data_ids, n_neighbors=neighbours,  n_components=2, metric='euclidean', title='UMAP projection of the HNeRV vectors', kmeans=True, num_clusters=num_clusters, axis_off=True
                )
                
                if vid_idx == 0:
                    axs[vid_idx, col_num_in_plot].set_title(layer_titles[layer_name_to_take], fontsize=13, fontweight='bold')
        break         
    fig.savefig(f'../outputs/supplementary/umaps_for_seeds.pdf', bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    main()