import glob
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.nn.functional import interpolate
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import center_crop, resize


def compute_bbox(binary_mask):
    (rows, cols) = np.where(binary_mask > 0)
    x_min, x_max, y_min, y_max = min(cols), max(cols), min(rows), max(rows)
    # Create the bbox in COCO format [x, y, width, height]
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    bbox = [x_min, y_min, width, height]
    return bbox


class CityscapesVPSVideoDataSet(Dataset):
    def __init__(self, args):

        vidname = args.vidname

        self.panoptic_categories, self.panoptic_images, self.panoptic_annotations = (
            self.load_json(args.anno_path)
        )
        self.rgb_img_name_suffix = "_leftImg8bit.png"

        rgb_mask_names = [
            file
            for file in sorted(os.listdir(args.panoptic_video_mask_dir))
            if file.startswith(vidname)
        ]

        self.video = [
            os.path.join(
                args.data_path, "_".join(x.split("_")[2:5]) + "_leftImg8bit.png"
            )
            for x in rgb_mask_names
        ]  # e.g. frankfurt_000000_001751

        self.image_ids = [
            "_".join(mask_name.split("_")[:5]) for mask_name in rgb_mask_names
        ]  # e.g. 0005_0025_frankfurt_000000_001751
        self.vid_inst_mask_paths = sorted(
            glob.glob(args.panoptic_inst_mask_dir + "/" + vidname + "*.png")
        )
        self.vid_rgb_mask_paths = sorted(
            glob.glob(args.panoptic_video_mask_dir + "/" + vidname + "*.png")
        )

        # Resize the input video and center crop
        self.crop_list, self.resize_list = args.crop_list, args.resize_list
        first_frame, _ = self.img_transform(self.img_load(0))
        self.final_size = first_frame.size(-2) * first_frame.size(-1)

    def img_load(self, idx):
        img = read_image(self.video[idx])
        return img / 255.0

    def load_json(self, panoptic_gt_val_city_vps_path):
        with open(panoptic_gt_val_city_vps_path, "r") as f:
            panoptic_gt_val_city_vps = json.load(f)

        panoptic_categories = panoptic_gt_val_city_vps["categories"]
        panoptic_images = panoptic_gt_val_city_vps["images"]
        panoptic_annotations = panoptic_gt_val_city_vps["annotations"]

        return panoptic_categories, panoptic_images, panoptic_annotations

    def load_annos(self, inst_mask_path, rgb_mask_path, image_id):
        # Load the panoptic annotation from JSON for this image_id
        panoptic_annos = [
            ann for ann in self.panoptic_annotations if ann["image_id"] == image_id
        ][0]
        segments_info = panoptic_annos["segments_info"]

        panop_mask = np.array(Image.open(inst_mask_path))
        rgb_panop_mask = np.array(Image.open(rgb_mask_path))

        ann_id = 0
        annotations = []

        categories = self.panoptic_categories
        categories.append(
            {"id": -1, "name": "other", "supercategory": "", "color": None}
        )
        categories_dict = {el["id"]: el for el in categories}

        for el in np.unique(panop_mask):
            if el < 1000:
                semantic_id = el
                is_crowd = 1
            else:
                semantic_id = el // 1000
                is_crowd = 0

            if semantic_id not in categories_dict:
                continue
            if categories_dict[semantic_id]["isthing"] == 0:
                is_crowd = 0

            # Binary mask
            binary_mask = panop_mask == el

            # Find instance ID from RGB values of colored mask
            r, g, b = np.unique(rgb_panop_mask[binary_mask], axis=0)[0]
            inst_id = r + g * 256 + b * 256**2

            segment_info = [
                seg_info for seg_info in segments_info if seg_info["id"] == inst_id
            ][0]

            # Compute bbox from binary mask
            bbox = compute_bbox(binary_mask)

            anno = {
                "id": ann_id,
                "inst_id": inst_id,
                "image_id": image_id,
                "category_id": segment_info["category_id"],
                "area": segment_info["area"],
                "iscrowd": is_crowd,
                "isthing": categories_dict[semantic_id]["isthing"],
                "binary_mask": np.array(binary_mask, dtype=np.uint8),
                "bbox": bbox,
            }

            annotations.append(anno)
            ann_id += 1

        return annotations

    def img_transform(self, img, annotations=[]):

        if self.crop_list != "-1":
            crop_h, crop_w = [int(x) for x in self.crop_list.split("_")[:2]]
            if "last" not in self.crop_list:
                img = center_crop(img, (crop_h, crop_w))
        if self.resize_list != "-1":
            if "_" in self.resize_list:
                resize_h, resize_w = [int(x) for x in self.resize_list.split("_")]
                img = interpolate(
                    input=img.unsqueeze(0), size=(resize_h, resize_w), mode="bicubic"
                ).squeeze(0)
            else:
                resize_hw = int(self.resize_list)
                img = resize(img, resize_hw, "bicubic")
        if "last" in self.crop_list:
            img = center_crop(img, (crop_h, crop_w))

        # Transform annotations as needed, if not empty
        if len(annotations) == 0:
            return img, []

        transformed_annos = []

        for anno in annotations:

            binary_mask = anno["binary_mask"]

            if self.crop_list != "-1":
                crop_h, crop_w = [int(x) for x in self.crop_list.split("_")[:2]]
                if "last" not in self.crop_list:
                    binary_mask = (
                        center_crop(torch.from_numpy(binary_mask), (crop_h, crop_w))
                        .squeeze()
                        .numpy()
                    )
            if self.resize_list != "-1":
                if "_" in self.resize_list:
                    resize_h, resize_w = [int(x) for x in self.resize_list.split("_")]
                    # Use nearest interpolation for binary tensors
                    binary_mask = (
                        interpolate(
                            input=torch.from_numpy(binary_mask)
                            .unsqueeze(0)
                            .unsqueeze(0),
                            size=(resize_h, resize_w),
                            mode="nearest",
                        )
                        .squeeze()
                        .squeeze()
                        .numpy()
                    )
                else:
                    resize_hw = int(self.resize_list)
                    # Use nearest interpolation for binary tensors
                    binary_mask = (
                        resize(
                            torch.from_numpy(binary_mask).unsqueeze(0),
                            resize_hw,
                            "nearest",
                        )
                        .squeeze()
                        .numpy()
                    )
            if "last" in self.crop_list:
                binary_mask = (
                    center_crop(torch.from_numpy(binary_mask), (crop_h, crop_w))
                    .squeeze()
                    .numpy()
                )

            # change binary mask back to 0s and 1s after above operations
            binary_mask = binary_mask != 0

            if binary_mask.sum() == 0:
                # If the binary mask has no non-zero values, the instance is lost
                # after applying transformations so skip it
                continue

            # Calculate the bounding box coordinates
            bbox = compute_bbox(binary_mask)

            area = binary_mask.sum()  # width * height

            transformed_anno = anno.copy()
            transformed_anno["bbox"] = bbox
            transformed_anno["area"] = area
            transformed_anno["binary_mask"] = binary_mask.squeeze()

            transformed_annos.append(transformed_anno)

        return img, transformed_annos

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img = self.img_load(idx)

        inst_mask_path = self.vid_inst_mask_paths[idx]
        rgb_mask_path = self.vid_rgb_mask_paths[idx]

        # Load instance annotations for the image
        annotations = self.load_annos(inst_mask_path, rgb_mask_path, image_id)

        # Process the image and instance annotations as needed
        tensor_image, annotations = self.img_transform(img, annotations)

        norm_idx = float(idx) / len(self.video)

        sample = {
            "img": tensor_image,
            "idx": idx,
            "norm_idx": norm_idx,
            "annotations": annotations,
        }

        return sample


class VIPSegVideoDataSet(Dataset):
    def __init__(self, args):
        vidname = args.vidname

        self.panoptic_categories, self.panoptic_videos, self.panoptic_annotations = (
            self.load_json(args.anno_path)
        )

        vid_info = [
            vid_info
            for vid_info in self.panoptic_videos
            if vid_info["video_id"] == vidname
        ][0]
        self.image_ids = [img_info["id"] for img_info in vid_info["images"]]

        self.video = [
            os.path.join(args.data_path, vidname, x + ".jpg") for x in self.image_ids
        ]

        self.vid_inst_mask_paths = [
            os.path.join(args.panomasks_dir, vidname, x + ".png")
            for x in self.image_ids
        ]
        self.vid_rgb_mask_paths = [
            os.path.join(args.panomasksRGB_dir, vidname, x + ".png")
            for x in self.image_ids
        ]
        self.video_annotations = [
            ann for ann in self.panoptic_annotations if ann["video_id"] == vidname
        ][0]["annotations"]

        # Resize the input video and center crop
        self.crop_list, self.resize_list = args.crop_list, args.resize_list
        first_frame, _ = self.img_transform(self.img_load(0))
        self.final_size = first_frame.size(-2) * first_frame.size(-1)

    def img_load(self, idx):
        img = read_image(self.video[idx])
        return img / 255.0

    def load_json(self, panoptic_gt_VIPSeg_path):
        with open(panoptic_gt_VIPSeg_path, "r") as f:
            panoptic_gt_VIPSeg = json.load(f)

        panoptic_categories = panoptic_gt_VIPSeg["categories"]
        panoptic_videos = panoptic_gt_VIPSeg["videos"]
        panoptic_annotations = panoptic_gt_VIPSeg["annotations"]
        return panoptic_categories, panoptic_videos, panoptic_annotations

    def load_annos(self, inst_mask_path, rgb_mask_path, image_id):
        # Load the panoptic annotation for this image_id
        panoptic_annos = [
            ann for ann in self.video_annotations if ann["image_id"] == image_id
        ][0]
        segments_info = panoptic_annos["segments_info"]

        panop_mask = np.array(Image.open(inst_mask_path))
        rgb_panop_mask = np.array(Image.open(rgb_mask_path))

        ann_id = 0
        annotations = []

        categories = self.panoptic_categories
        categories.append(
            {"id": -1, "name": "other", "supercategory": "", "color": None}
        )
        categories_dict = {el["id"]: el for el in categories}

        for el in np.unique(panop_mask):
            # Ignore background
            if el == 0:
                continue
            # VIPSeg has 125 stuff categories
            # For the thing class, divide by 100 to obtain category
            if el < 125:
                semantic_id = el
                is_crowd = 0
            else:
                semantic_id = el // 100
                is_crowd = 0

            # Categories start from 0
            semantic_id = semantic_id - 1

            if semantic_id not in categories_dict:
                continue
            if categories_dict[semantic_id]["isthing"] == 0:
                is_crowd = 0

            # Binary mask
            binary_mask = panop_mask == el

            # Find instance ID from RGB values of colored mask
            r, g, b = np.unique(rgb_panop_mask[binary_mask], axis=0)[0]
            inst_id = r + g * 256 + b * 256**2

            segment_info = [
                seg_info for seg_info in segments_info if seg_info["id"] == inst_id
            ][0]

            # Compute bbox from binary mask
            bbox = compute_bbox(binary_mask)

            anno = {
                "id": ann_id,
                "inst_id": inst_id,
                "image_id": image_id,
                "category_id": segment_info["category_id"],
                "area": segment_info["area"],
                "iscrowd": is_crowd,
                "isthing": categories_dict[semantic_id]["isthing"],
                "binary_mask": np.array(binary_mask, dtype=np.uint8),
                "bbox": bbox,
            }

            annotations.append(anno)
            ann_id += 1

        return annotations

    def img_transform(self, img, annotations=[]):

        if self.crop_list != "-1":
            crop_h, crop_w = [int(x) for x in self.crop_list.split("_")[:2]]
            if "last" not in self.crop_list:
                img = center_crop(img, (crop_h, crop_w))
        if self.resize_list != "-1":
            if "_" in self.resize_list:
                resize_h, resize_w = [int(x) for x in self.resize_list.split("_")]
                img = interpolate(
                    input=img.unsqueeze(0), size=(resize_h, resize_w), mode="bicubic"
                ).squeeze(0)
            else:
                resize_hw = int(self.resize_list)
                img = resize(img, resize_hw, "bicubic")
        if "last" in self.crop_list:
            img = center_crop(img, (crop_h, crop_w))

        # Transform annotations as needed if not empty
        if len(annotations) == 0:
            return img, []

        transformed_annos = []

        for anno in annotations:

            binary_mask = anno["binary_mask"]

            if self.crop_list != "-1":
                crop_h, crop_w = [int(x) for x in self.crop_list.split("_")[:2]]
                if "last" not in self.crop_list:
                    binary_mask = (
                        center_crop(torch.from_numpy(binary_mask), (crop_h, crop_w))
                        .squeeze()
                        .numpy()
                    )
            if self.resize_list != "-1":
                if "_" in self.resize_list:
                    resize_h, resize_w = [int(x) for x in self.resize_list.split("_")]
                    # Use nearest interpolation for binary tensors
                    binary_mask = (
                        interpolate(
                            input=torch.from_numpy(binary_mask)
                            .unsqueeze(0)
                            .unsqueeze(0),
                            size=(resize_h, resize_w),
                            mode="nearest",
                        )
                        .squeeze()
                        .squeeze()
                        .numpy()
                    )
                else:
                    resize_hw = int(self.resize_list)
                    # Use nearest interpolation for binary tensors
                    binary_mask = (
                        resize(
                            torch.from_numpy(binary_mask).unsqueeze(0),
                            resize_hw,
                            "nearest",
                        )
                        .squeeze()
                        .numpy()
                    )
            if "last" in self.crop_list:
                binary_mask = (
                    center_crop(torch.from_numpy(binary_mask), (crop_h, crop_w))
                    .squeeze()
                    .numpy()
                )

            # change binary mask back to 0s and 1s after above operations
            binary_mask = binary_mask != 0

            if binary_mask.sum() == 0:
                # If the binary mask has no non-zero values, the instance is lost
                # after applying transformations so skip it
                continue

            # Calculate the bounding box coordinates
            bbox = compute_bbox(binary_mask)

            area = binary_mask.sum()

            transformed_anno = anno.copy()
            transformed_anno["bbox"] = bbox
            transformed_anno["area"] = area
            transformed_anno["binary_mask"] = binary_mask.squeeze()

            transformed_annos.append(transformed_anno)

        return img, transformed_annos

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img = self.img_load(idx)

        inst_mask_path = self.vid_inst_mask_paths[idx]
        rgb_mask_path = self.vid_rgb_mask_paths[idx]

        # Load instance annotations for the image
        annotations = self.load_annos(inst_mask_path, rgb_mask_path, image_id)

        # Process the image and instance annotations as needed
        tensor_image, annotations = self.img_transform(img, annotations)

        norm_idx = float(idx) / len(self.video)

        sample = {
            "img": tensor_image,
            "idx": idx,
            "norm_idx": norm_idx,
            "annotations": annotations,
        }

        return sample


class VideoFlowDataSet(Dataset):
    def __init__(self, args):

        self.video = [
            os.path.join(args.image_dir, file)
            for file in sorted(os.listdir(args.image_dir))
        ]
        self.flow_images = [
            os.path.join(args.flow_dir, file)
            for file in sorted(os.listdir(args.flow_dir))
        ]  # pre-computed npy files for frame-to-frame flow

        # Resize the input video and center crop
        self.crop_list, self.resize_list = args.crop_list, args.resize_list
        first_frame, _ = self.img_transform(self.img_load(0))
        self.final_size = first_frame.size(-2) * first_frame.size(-1)

    def img_load(self, idx):
        img = read_image(self.video[idx])
        return img / 255.0

    def load_flow(self, idx):
        flow_img = np.load(self.flow_images[idx])
        # Convert to tensor. Note: this is not normalized
        flow_img = torch.tensor(flow_img, dtype=torch.float32).unsqueeze(0)
        return flow_img / 255.0

    def img_transform(self, img, flow_img=None):
        if self.crop_list != "-1":
            crop_h, crop_w = [int(x) for x in self.crop_list.split("_")[:2]]
            if "last" not in self.crop_list:
                img = center_crop(img, (crop_h, crop_w))
                if flow_img is not None:
                    flow_img = center_crop(flow_img, (crop_h, crop_w))
        if self.resize_list != "-1":
            if "_" in self.resize_list:
                resize_h, resize_w = [int(x) for x in self.resize_list.split("_")]
                img = interpolate(
                    input=img.unsqueeze(0), size=(resize_h, resize_w), mode="bicubic"
                ).squeeze(0)
                if flow_img is not None:
                    flow_img = interpolate(
                        input=flow_img.unsqueeze(0),
                        size=(resize_h, resize_w),
                        mode="bicubic",
                    ).squeeze(0)
            else:
                resize_hw = int(self.resize_list)
                img = resize(img, resize_hw, "bicubic")
                if flow_img is not None:
                    flow_img = resize(flow_img, resize_hw, "bicubic")
        if "last" in self.crop_list:
            img = center_crop(img, (crop_h, crop_w))
            if flow_img is not None:
                flow_img = center_crop(flow_img, (crop_h, crop_w))

        return img, flow_img

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):
        img = self.img_load(idx)
        flow_img = self.load_flow(idx)

        # Process the image and flow map as needed
        tensor_image, tensor_flow_img = self.img_transform(img, flow_img)

        norm_idx = float(idx) / len(self.video)

        sample = {
            "img": tensor_image,
            "idx": idx,
            "norm_idx": norm_idx,
            "flow_img": tensor_flow_img,
        }

        return sample
