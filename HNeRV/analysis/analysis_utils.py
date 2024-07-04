import json
import os
import random

import numpy as np
import torch
from analysis_utils import *


class Args:
    pass


def load_model_args():
    args = Args()
    args.embed = "pe_1.25_80"
    args.ks = "0_3_3"
    args.num_blks = "1_1"
    args.enc_dim = "64_16"
    args.enc_strds, args.dec_strds = [], [4, 2, 2]
    args.fc_dim = 37
    args.fc_hw = "8_16"
    args.norm = "none"
    args.act = "gelu"
    args.reduce = 1.2
    args.lower_width = 6
    args.conv_type = ["convnext", "pshuffel"]
    args.b = 1
    args.out_bias = 0.0

    args.resize_list = "128_256"
    args.modelsize = 1.0

    args.dump_images = False

    return args


def load_model_checkpoint(model, args):
    checkpoint_path = args.weight
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    orig_ckt = checkpoint["state_dict"]
    new_ckt = {k.replace("blocks.0.", ""): v for k, v in orig_ckt.items()}
    if "module" in list(orig_ckt.keys())[0] and not hasattr(model, "module"):
        new_ckt = {k.replace("module.", ""): v for k, v in new_ckt.items()}
        missing = model.load_state_dict(new_ckt, strict=False)
    elif "module" not in list(orig_ckt.keys())[0] and hasattr(model, "module"):
        missing = model.module.load_state_dict(new_ckt, strict=False)
    else:
        missing = model.load_state_dict(new_ckt, strict=False)
    print(missing)
    print(
        "=> loaded checkpoint '{}' (epoch {})".format(args.weight, checkpoint["epoch"])
    )

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


def convert_annotations_to_numpy(tensor_annotations):
    annotations = []

    for tensor_anno in tensor_annotations:
        annotation = {}

        annotation["id"] = tensor_anno["id"].item()
        annotation["inst_id"] = tensor_anno["inst_id"].item()
        annotation["image_id"] = tensor_anno["image_id"].item()
        annotation["category_id"] = tensor_anno["category_id"].item()
        annotation["area"] = tensor_anno["area"].item()
        annotation["iscrowd"] = tensor_anno["iscrowd"].item()
        annotation["isthing"] = tensor_anno["isthing"].item()
        bbox = [bbox_tensor.item() for bbox_tensor in tensor_anno["bbox"]]
        annotation["bbox"] = bbox
        annotation["binary_mask"] = tensor_anno["binary_mask"].numpy()

        annotations.append(annotation)

    return annotations


def add_other_annotation(annotations):

    # Create a mask that indicates whether a location contains at least one instance
    object_region_mask = None
    for ann in annotations:
        binary_mask = ann["binary_mask"].squeeze()
        if object_region_mask is None:
            object_region_mask = binary_mask.copy()
        else:
            # aggregate
            object_region_mask += binary_mask

    # Binarize
    object_region_mask = object_region_mask != 0

    # Create an annotation denoting "other" for regions that have no objects
    annotations.append(
        {
            "id": -1,
            "inst_id": -1,
            "bbox": compute_bbox(object_region_mask),
            "area": object_region_mask.sum(),
            "binary_mask": object_region_mask,
            "iscrowd": 0,
            "isthing": 0,
            "category_id": -1,
            "image_id": annotations[0]["image_id"],
        }
    )
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
    args.data_split = "1_1_1"
    args.shuffle_data = False

    if dataset_name == "cityscapes":
        # Add cityscapes VPS paths
        args.cityscapes_vps_root = "../data/cityscapes_vps"
        args.split = "val"
        args.panoptic_video_mask_dir = os.path.join(
            args.cityscapes_vps_root, args.split, "panoptic_video"
        )
        args.panoptic_inst_mask_dir = os.path.join(
            args.cityscapes_vps_root, args.split, "panoptic_inst"
        )

        args.data_path = os.path.join(args.cityscapes_vps_root, args.split, "img_all")
        args.anno_path = "../data/cityscapes_vps/panoptic_gt_val_city_vps.json"

        args.vidname = vidname

        with open(args.anno_path, "r") as f:
            panoptic_gt_val_city_vps = json.load(f)

        panoptic_categories = panoptic_gt_val_city_vps["categories"]

        categories = panoptic_categories
        categories.append(
            {"id": -1, "name": "other", "supercategory": "", "color": None}
        )
        categories_dict = {el["id"]: el for el in categories}

    elif dataset_name == "vipseg":
        args.VIPSeg_720P_root = "../data/VIPSeg-Dataset/VIPSeg/VIPSeg_720P"
        args.panomasks_dir = os.path.join(args.VIPSeg_720P_root, "panomasks")
        args.panomasksRGB_dir = os.path.join(args.VIPSeg_720P_root, "panomasksRGB")
        args.data_path = os.path.join(args.VIPSeg_720P_root, "images")
        args.anno_path = (
            "../data/VIPSeg-Dataset/VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg.json"
        )

        args.vidname = vidname

        args.crop_list = "640_1280"

        with open(args.anno_path, "r") as f:
            panoptic_gt_VIPSeg = json.load(f)

        panoptic_categories = panoptic_gt_VIPSeg["categories"]

        categories = panoptic_categories
        categories.append(
            {"id": -1, "name": "other", "supercategory": "", "color": None}
        )
        categories_dict = {el["id"]: el for el in categories}

    return args, categories_dict


def flow_load_dataset_specific_args(args, dataset_name, vidname):
    args.batchSize = args.b
    args.distributed = False
    args.workers = 4
    args.data_split = "1_1_1"
    args.shuffle_data = False

    if dataset_name == "cityscapes":
        args.cityscapes_vps_root = "../data/cityscapes_vps"
        args.split = "val"

        args.vps_subset_path = os.path.join(
            args.cityscapes_vps_root, "vps_subset", args.split
        )
        args.image_dir = os.path.join(args.vps_subset_path, vidname, "images")
        # Path to the directory containing pre-computed frame-to-frame optical flow data
        args.flow_dir = os.path.join(args.vps_subset_path, vidname, "flows")

        args.crop_list = "-1"

        args.vidname = vidname

    elif dataset_name == "vipseg":
        args.VIPSeg_720P_root = "../data/VIPSeg-Dataset/VIPSeg/VIPSeg_720P"

        args.image_dir = os.path.join(args.VIPSeg_720P_root, "images", vidname)
        # Path to the directory containing pre-computed frame-to-frame optical flow data
        args.flow_dir = os.path.join(args.VIPSeg_720P_root, "flows", vidname)

        args.crop_list = "640_1280"

        args.vidname = vidname

    return args
