import glob
import logging
import os
import pickle

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise


def kernel_wise_norm(kernels, norm_type="min_max"):
    if norm_type == "min_max":
        for idx in range(kernels.shape[0]):
            if kernels[idx].max() - kernels[idx].min() != 0:
                kernels[idx] = (kernels[idx] - kernels[idx].min()) / (
                    kernels[idx].max() - kernels[idx].min()
                )
    else:
        raise NotImplementedError(f"{norm_type} not implemented")
    return kernels


def check_distance(contri_1, contri_2):
    contri_1 = rearrange(contri_1, "k h w -> k (h w)")
    contri_2 = rearrange(contri_2, "k h w -> k (h w)")
    distance = pairwise.euclidean_distances(contri_1, contri_2)
    return distance


def get_contrib(head_dict, seed, frame_no, mlp=False):
    if mlp:
        contrib = head_dict[seed]
    else:
        contrib = head_dict[seed][frame_no]
    contrib = contrib.copy()
    for key in contrib.keys():
        if not mlp:
            if len(contrib[key].shape) == 4:
                contrib[key] = rearrange(contrib[key], "ink outk h w -> (ink outk) h w")
        contrib[key] = np.abs(contrib[key])
    return contrib


def get_spatial_clustering(img, clustering_type, num_clusters=None):
    img_h, img_w, _ = img.shape
    img_pixels = rearrange(img, "h w c -> (h w) c")
    if clustering_type == "kmeans":

        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(img_pixels)
        labels = kmeans.labels_
    else:
        raise NotImplementedError(f"{clustering_type} not implemented")
    labels = rearrange(labels, "(h w) -> h w", h=img_h, w=img_w)
    return labels

def cluster_map_to_vec(kernels, cluster_map):
    num_clusters = np.unique(cluster_map).shape[0]
    final_vecs = np.zeros((kernels.shape[0], num_clusters))
    for kernel_num in range(kernels.shape[0]):
        kernel_to_consider = kernels[kernel_num]
        for i in range(num_clusters):
            final_vecs[kernel_num, i] = np.mean(kernel_to_consider[cluster_map == i])
    return final_vecs


def get_spatial_vec(kernels, kernel_size=8):
    kernels_to_consider = kernels.unsqueeze(0)
    avg_pool = torch.nn.AvgPool2d(kernel_size, kernel_size)
    kernels_to_consider = avg_pool(kernels_to_consider).squeeze(0)
    kernels_to_consider = rearrange(kernels_to_consider, "k h w -> k (h w)")
    return kernels_to_consider.numpy()


def get_kernel_vecs_from_layer(
    head_dict,
    vid_layer_maps,
    frame_no=None,
    mlp=False,
    kernel_size=16,
    drop_dead_neurons=True,
):
    all_seed_head_vecs = {}

    for seed in head_dict.keys():
        all_seed_head_vecs[seed] = {}
        seed_contributions_all_layers = get_contrib(
            head_dict.copy(), seed, frame_no, mlp=mlp
        )
        for layer_name in seed_contributions_all_layers.keys():
            all_seed_head_vecs[seed][layer_name] = {}
            seed_contributions = seed_contributions_all_layers[layer_name]
            if drop_dead_neurons:
                sum_over_image_per_neuron = seed_contributions.sum(
                    dim=(1, 2)
                )  # sum over H,W
                seed_contributions = seed_contributions[
                    torch.nonzero(sum_over_image_per_neuron).squeeze()
                ]
            seed_contributions = kernel_wise_norm(
                seed_contributions, norm_type="min_max"
            )
            assert np.isnan(seed_contributions).sum() == 0
            for label_map_type, label_map in vid_layer_maps.items():
                all_seed_head_vecs[seed][layer_name][label_map_type] = (
                    cluster_map_to_vec(seed_contributions.numpy(), label_map)
                )
            all_seed_head_vecs[seed][layer_name][f"spatial_{kernel_size}"] = (
                get_spatial_vec(seed_contributions, kernel_size=kernel_size)
            )
            all_seed_head_vecs[seed][layer_name]["kernels"] = seed_contributions.numpy()
    return all_seed_head_vecs


def get_label_maps(img_path, pixel_clusters=32):

    img = np.array(Image.open(img_path))
    label_maps = {}
    for clustering_base in ["rgb", "gabor"]:
        label_maps[clustering_base] = get_label_map(
            img, clustering_base, num_clusters=pixel_clusters
        )
    return label_maps


def load_stuff(vid_path, pixel_clusters=32, img_path=None, label_maps=None):
    pickle_path = glob.glob(os.path.join(vid_path, "*.pickle"))
    if len(pickle_path) == 0:
        pickle_path = glob.glob(os.path.join(vid_path, "*.pkl"))
    contri_pickle = pickle_path[0]
    head_dict = pickle.load(open(contri_pickle, "rb"))
    logging.info(f"Loaded pickle file")
    if label_maps is None:
        if img_path is None:
            img_path = glob.glob(os.path.join(vid_path, "*.png"))
            if len(img_path) == 0:
                img_path = glob.glob(os.path.join(vid_path, "*.jpg"))
            img_path = img_path[0]

        label_maps = get_label_maps(img_path, pixel_clusters=pixel_clusters)
    data_dict = {"head_dict": head_dict, "label_maps": label_maps}
    return data_dict


def get_label_map(img, clustering_base, num_clusters=32):
    img_to_cluster = img.copy()
    if clustering_base == "gabor":
        img_to_cluster = np.array(Image.fromarray(img_to_cluster).convert("L"))
        img_to_cluster = compute_gabor(img_to_cluster)
    elif clustering_base == "rgb":
        pass
    else:
        raise NotImplementedError(f"{clustering_base} not implemented")
    label_map = get_spatial_clustering(
        img_to_cluster, clustering_type="kmeans", num_clusters=num_clusters
    )
    return label_map


def get_gabor_kernels():
    kernels = []
    for theta in range(4):
        theta = theta / 4.0 * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25, 0.5):
                kernel = np.real(
                    gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma)
                )
                kernels.append(kernel)
        return kernels


def compute_gabor(image):
    kernels = get_gabor_kernels()
    all_filtered = []
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode="wrap")[..., None]
        all_filtered.append(filtered)
    all_filtered = np.concatenate(all_filtered, axis=-1)
    return all_filtered