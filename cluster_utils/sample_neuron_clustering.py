"""
This script provides an example for computing contribution vectors of neurons,
using contribution vectors to cluster neurons in each layer of an INR, 
and visualizing example maps from resulting clusters.
"""

import glob
import os
import warnings

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib import cm
from sklearn.cluster import KMeans

from cluster_utils import get_kernel_vecs_from_layer, load_stuff

warnings.filterwarnings("ignore")


def save_umap(
    dump_path,
    layer_name,
    data,
    data_ids,
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric="euclidean",
    title="",
    num_clusters=5,
    kmeans=False,
    axis_off=False,
):
    data_ids = np.array(data_ids).copy()
    if n_neighbors is None:
        fit = umap.UMAP()
    else:
        fit = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=42,
        )
    u = fit.fit_transform(data)
    all_ids = np.arange(len(data_ids))
    cmap = cm.get_cmap("viridis")
    norm = cm.colors.Normalize(vmin=np.min(data_ids), vmax=np.max(data_ids))
    if len(np.unique(data_ids)) == 2 or len(np.unique(data_ids)) == 1:
        norm = cm.colors.Normalize(vmin=0, vmax=20)
        data_ids[data_ids == 0] = 15
        data_ids[data_ids == 1] = 10
    ids_to_take = []

    plt.figure(figsize=(6, 3))
    plt.scatter(u[:, 0], u[:, 1], c=cmap(norm(data_ids)), s=8, alpha=0.7)

    if kmeans:
        kmean_labels = KMeans(n_clusters=num_clusters, random_state=0).fit_predict(u)
        cluster_colours = np.random.rand(num_clusters, 3)
        for cluster_id in range(num_clusters):
            cluster_data = u[kmean_labels == cluster_id]
            cluster_id_to_take = np.random.choice(
                all_ids[kmean_labels == cluster_id], 5
            )
            ids_to_take.extend(list(cluster_id_to_take))

            center = np.mean(cluster_data, axis=0)
            radius = np.max(np.linalg.norm(cluster_data - center, axis=1))
            circle = patches.Circle(
                center,
                radius=radius,
                edgecolor=cluster_colours[cluster_id],
                facecolor="none",
            )

    plt.tick_params(
        axis="both",
        left=False,
        top=False,
        right=False,
        bottom=False,
        labelleft=False,
        labeltop=False,
        labelright=False,
        labelbottom=False,
    )

    plt.savefig(
        os.path.join(dump_path, f"{layer_name}_umap.png"),
        bbox_inches="tight",
        dpi=500,
        pad_inches=0,
    )

    return u, kmean_labels


def get_all_data(dataset_name, vid_name, inr_types, base_path, pixel_clusters):
    all_data = {}

    img_path = None
    label_maps = None
    vid_data = {}
    for inr_type in inr_types:
        vid_path = base_path.format(inr_type, dataset_name, vid_name)
        if img_path is None:
            img_path = glob.glob(os.path.join(vid_path, "*.png"))
            if len(img_path) == 0:
                img_path = glob.glob(os.path.join(vid_path, "*.jpg"))
            img_path = img_path[0]
        vid_data[inr_type] = load_stuff(
            vid_path,
            pixel_clusters=pixel_clusters,
            img_path=img_path,
            label_maps=label_maps,
        )
        label_maps = vid_data[inr_type]["label_maps"]
    all_data[f"{dataset_name}"] = vid_data

    return all_data


def get_cluster_results(
    all_data,
    dataset_name,
    inr_types,
):
    cluster_results = {}

    vid_data = all_data[dataset_name]

    cluster_results[dataset_name] = {}
    for inr_type in inr_types:
        if inr_type == "HNeRV":
            mlp = False
        else:
            mlp = True
        head_dict = vid_data[inr_type]["head_dict"]
        vid_label_maps = vid_data[inr_type]["label_maps"]
        # Drop dead neurons as necessary
        vecs = get_kernel_vecs_from_layer(
            head_dict,
            vid_label_maps,
            frame_no=0,
            mlp=mlp,
            kernel_size=16,
            drop_dead_neurons=True,
        )
        cluster_results[dataset_name][inr_type] = vecs

    return cluster_results


def get_sample_img_frames(vidname, dataset_name, num_samples=5):
    if dataset_name == "Cityscapes_VPS_models":
        # the frame for training cityscapes is the first frame with provided mask/annotation
        cityscapes_vps_root = "../../data/cityscapes_vps"
        split = "val"
        panoptic_video_mask_dir = os.path.join(
            cityscapes_vps_root, split, "panoptic_video"
        )

        data_path = os.path.join(cityscapes_vps_root, split, "img_all")

        rgb_mask_names = [
            file
            for file in sorted(os.listdir(panoptic_video_mask_dir))
            if file.startswith(vidname)
        ]
        rgb_mask_names = rgb_mask_names[:: len(rgb_mask_names) // num_samples]

        img_paths = [
            os.path.join(
                data_path, "_".join(rgb_mask_name.split("_")[2:5]) + "_leftImg8bit.png"
            )
            for rgb_mask_name in rgb_mask_names
        ]
        return img_paths

    elif dataset_name == "VIPSeg_models":
        # all vipseg frames are annotated, we will take the first frame
        VIPSeg_720P_root = "../../data/VIPSeg-Dataset/VIPSeg/VIPSeg_720P"

        vid_path = os.path.join(VIPSeg_720P_root, "images", vidname)
        img_paths = [
            os.path.join(vid_path, frame) for frame in sorted(os.listdir(vid_path))
        ]
        img_paths = img_paths[:: len(img_paths) // num_samples]

        return img_paths


def main():
    inr_types = ["image_inr", "HNeRV"]

    cityscapes_vids = ["0015", "0315"]
    cityscapes_vid_pairs = [("Cityscapes_VPS_models", vid) for vid in cityscapes_vids]
    vipseg_vids = ["79_brZzLyzaXbA", "127_-hIVCYO4C90", "1890_FIxEdRs9254"]
    vipseg_vid_pairs = [("VIPSeg_models", vid) for vid in vipseg_vids]

    dataset_vid_name_pairs = cityscapes_vid_pairs + vipseg_vid_pairs

    frame_no = 0
    pixel_clusters = 32
    base_path = "../../{}/output/{}/contributions/{}"

    layer_names = {
        "HNeRV": ["head_layer_output_contrib", "nerv_blk_3_output_contrib", "nerv_blk_2_output_contrib", "nerv_blk_1_output_contrib"],
        "image_inr": ["layer_3_output_contrib", "layer_2_output_contrib", "layer_1_output_contrib"],
    }
    num_neighbors = {
        "head_layer_output_contrib": 5,
        "nerv_blk_3_output_contrib": 25,
        "nerv_blk_2_output_contrib": 75,
        "nerv_blk_1_output_contrib": 150,
        "layer_3_output_contrib": 5,
        "layer_2_output_contrib": 5,
        "layer_1_output_contrib": 5,
    }

    seeds_to_consider = [1]
    data_id_label_type = "seed"
    colors = np.random.rand(len(seeds_to_consider), 3)
    colors = np.hstack([colors, np.ones((len(seeds_to_consider), 1)) * 0.25])

    for dataset_name, vid_name in dataset_vid_name_pairs:
        print(dataset_name, vid_name)

        dump_path = f"../outputs/contrib_maps/{dataset_name}/{vid_name}/"
        os.makedirs(dump_path, exist_ok=True)

        num_samples = 6
        img_paths = get_sample_img_frames(
            vid_name, dataset_name, num_samples=num_samples
        )

        for i in range(num_samples):
            plt.figure()
            rgb_img = plt.imread(img_paths[i])
            # if dataset is "vipseg", center crop the 720x1280 image to 640x1280
            if dataset_name == "VIPSeg_models":
                rgb_img = rgb_img[40:680, :]
            plt.imshow(rgb_img)
            plt.axis("off")
            plt.savefig(
                os.path.join(dump_path, f"{i}.png"),
                bbox_inches="tight",
                dpi=500,
                pad_inches=0,
            )

        all_data = get_all_data(
            dataset_name, vid_name, inr_types, base_path, pixel_clusters
        )

        cluster_results = get_cluster_results(all_data, dataset_name, inr_types)

        inrs_to_consider = ["image_inr", "HNeRV"]
        dataset_to_use = dataset_name

        # Cluster neurons in each layer (of FFN + NeRV) for the current video
        for inr_idx, inr_type in enumerate(inrs_to_consider):

            for layer_idx, layer_name_to_take in enumerate(layer_names[inr_type]):

                seed = seeds_to_consider[0]
                seed_idx = 0

                data = []
                data_ids = []
                all_kernels = []

                all_kernels.append(
                    cluster_results[dataset_to_use][inr_type][seed][layer_name_to_take][
                        "kernels"
                    ]
                )

                seed_data = cluster_results[dataset_to_use][inr_type][seed][
                    layer_name_to_take
                ]["gabor"].copy()
                data.append(seed_data)
                if data_id_label_type == "inr_type":
                    data_ids.extend([inr_idx] * len(seed_data))
                else:
                    data_ids.extend([seed_idx] * len(seed_data))

                # For this sample plot, only one model is trained for each video
                # So we use a single seed
                data = np.concatenate(data, axis=0)
                data_ids = np.array(data_ids)
                all_kernels = np.concatenate(all_kernels, axis=0)

                neighbours = num_neighbors[layer_name_to_take]
                num_clusters = 6
                cluster_point_info, cluster_id_data = save_umap(
                    dump_path,
                    "_".join(layer_name_to_take.split("_")[:2]),
                    data,
                    data_ids,
                    n_neighbors=neighbours,
                    n_components=2,
                    metric="euclidean",
                    title="UMAP projection of the HNeRV vectors",
                    kmeans=True,
                    num_clusters=num_clusters,
                    axis_off=True,
                )

                # Select a few (4) points from 4 clusters to plot
                num_clusters_to_plot = 4
                num_contrib_maps_per_cluster = 4
                for cluster_id in range(num_clusters_to_plot):
                    all_ids = np.arange(len(data_ids))
                    for class_id in [0]:
                        ids = (cluster_id_data == cluster_id) & (data_ids == class_id)
                        ids_to_consider = all_ids[ids]
                        # select 4 random neurons
                        chosen_contrib_maps = all_kernels[ids_to_consider][
                            np.random.choice(
                                len(ids_to_consider), num_contrib_maps_per_cluster
                            )
                        ]

                    # Create a 2x2 grid of contribution maps
                    heatmap_grid = np.vstack(
                        [
                            np.hstack(
                                [chosen_contrib_maps[0], chosen_contrib_maps[1]]
                            ),
                            np.hstack(
                                [chosen_contrib_maps[2], chosen_contrib_maps[3]]
                            ),
                        ]
                    )

                    plt.figure(figsize=(6, 4))
                    plt.imshow(heatmap_grid, cmap="magma")
                    plt.axis("off")
                    layer_name = "_".join(layer_name_to_take.split("_")[:2])
                    plt.savefig(
                        os.path.join(dump_path, f"{layer_name}_c{cluster_id}.png"),
                        bbox_inches="tight",
                        dpi=500,
                        pad_inches=0,
                    )


if __name__ == "__main__":
    main()
