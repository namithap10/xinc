import glob
import json
import os

import numpy as np
import torch
import torchvision as tv
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import center_crop
from torchvision.transforms.functional import resize as resize_transform
from utils import data_process

"""
    All dataloader outputs should be a dict. 
    Follows a standard format. 
"""
class single_image_cityscape_vps_dataset(Dataset):
    def __init__(self, cfg):

        self.cfg = cfg
        
        vidname = cfg.data.vidname
        self.rgb_img_name_suffix = "_leftImg8bit.png"

        rgb_mask_names = [
            file for file in sorted(os.listdir(cfg.data.panoptic_video_mask_dir)) if file.startswith(vidname)
        ]
        rgb_mask_name = rgb_mask_names[cfg.data.frame_num_in_video]

        img_path = os.path.join(cfg.data.data_path, "_".join(rgb_mask_name.split("_")[2:5]) + "_leftImg8bit.png")

        image_id = "_".join(rgb_mask_name.split("_")[:5])  # e.g. 0005_0025_frankfurt_000000_001751
        print("image_id: ", image_id)
        inst_mask_path = sorted(glob.glob(cfg.data.panoptic_inst_mask_dir + "/" + vidname + "*.png"))[0]
        rgb_mask_path = sorted(glob.glob(cfg.data.panoptic_video_mask_dir + "/" + vidname + "*.png"))[0]
        
        anno_path = cfg.data.anno_path

        self.proc = data_process.DataProcessor(cfg.data)
        self.data = self.load_tensor(img_path, cfg.data.data_shape)
        self.annotations = self.load_annos(
            anno_path, inst_mask_path, rgb_mask_path, image_id, cfg.data.data_shape
        )
        # Converts data to [N,3]
        # Features shape can be different from data shape as we might pad.
        self.features, self.features_shape = self.proc.get_features(self.data)

    def load_json(self, panoptic_gt_val_city_vps_path):
        with open(panoptic_gt_val_city_vps_path, "r") as f:
            panoptic_gt_val_city_vps = json.load(f)

        panoptic_categories = panoptic_gt_val_city_vps["categories"]
        panoptic_images = panoptic_gt_val_city_vps["images"]
        panoptic_annotations = panoptic_gt_val_city_vps["annotations"]

        return panoptic_categories, panoptic_images, panoptic_annotations
    
    def load_tensor(self, path, resize=None):
        """
        Read image from path and return tensor.
        Use PIL
        """
        img = Image.open(path)
        img = img.convert("RGB")
        if resize is None:
            img = transforms.ToTensor()(img)
        else:
            resize = (resize, resize) if isinstance(resize, int) else resize
            custom_transforms = tv.transforms.Compose(
                [tv.transforms.Resize(resize), tv.transforms.ToTensor()]
            )
            img = custom_transforms(img)
        return img

    def load_annos(self, anno_path, inst_mask_path, rgb_mask_path, image_id, resize=None):
        panoptic_categories, panoptic_images, panoptic_annotations = self.load_json(anno_path)
        
        # Load the panoptic annotation from JSON for this image_id        
        panoptic_annos = [ann for ann in panoptic_annotations if ann['image_id'] == image_id][0]
        segments_info = panoptic_annos['segments_info']
        
        panop_mask = np.array(Image.open(inst_mask_path))
        rgb_panop_mask = np.array(Image.open(rgb_mask_path))
        
        ann_id = 0
        annotations = []
        
        categories = panoptic_categories
        categories.append({'id': -1, 'name': 'other', 'supercategory': '', 'color':None})
        categories_dict = {el['id']: el for el in categories}
        
        for el in np.unique(panop_mask):
            if el < 1000:
                semantic_id = el
                is_crowd = 1
            else: 
                semantic_id = el // 1000
                is_crowd = 0
            
            if semantic_id not in categories_dict:
                continue
            if categories_dict[semantic_id]['isthing'] == 0:
                is_crowd = 0
            
            # Binary mask
            binary_mask = (panop_mask == el)
            
            # Find instance ID from RGB values of colored mask
            r, g, b = np.unique(rgb_panop_mask[binary_mask], axis=0)[0]
            inst_id = r + g * 256 + b * 256**2
        
            segment_info = [seg_info for seg_info in segments_info if seg_info['id'] == inst_id][0]
            
            # Compute bbox from binary mask
            bbox = compute_bbox(binary_mask)
            
            anno = {
                "id": ann_id,
                "inst_id": inst_id,
                "image_id": image_id,
                "category_id": segment_info["category_id"],
                "area": segment_info["area"],
                "iscrowd": is_crowd,
                "isthing": categories_dict[semantic_id]['isthing'],
                "binary_mask": np.array(binary_mask, dtype=np.uint8),
                "bbox": bbox,
            }
            
            annotations.append(anno)
            ann_id += 1

        if resize is None:
            return annotations

        resize_hw = (resize,resize) if isinstance(resize,int) else resize
        transformed_annotations = []

        for anno in annotations:
            binary_mask = anno["binary_mask"]
            # Resize the binary mask
            binary_mask = resize_transform(
                torch.from_numpy(binary_mask).unsqueeze(0), resize_hw, interpolation=InterpolationMode.NEAREST
            ).squeeze().numpy()
            
            binary_mask = binary_mask != 0
            if binary_mask.sum() == 0:
                # If the binary mask has no non-zero values, the instance is lost
                # after applying transformations so skip it
                continue
            
            # Calculate the bounding box coordinates
            bbox = compute_bbox(binary_mask)
            area = binary_mask.sum()
            
            transformed_ann = anno.copy()
            transformed_ann["bbox"] = bbox
            transformed_ann["area"] = area
            transformed_ann["binary_mask"] = binary_mask.squeeze()
            
            transformed_annotations.append(transformed_ann)

        
        return transformed_annotations

    def __len__(self):
        return 1

    def __getitem__(self, index):
        output = {
            "data": self.data,
            "features": self.features,
            "features_shape": torch.tensor(self.features_shape),
            "data_shape": torch.tensor(self.data.shape),
            "annotations": self.annotations,
        }
        return output

def compute_bbox(binary_mask):
    (rows, cols) = np.where(binary_mask > 0)
    x_min, x_max, y_min, y_max = min(cols), max(cols), min(rows), max(rows)
    # Create the bbox in COCO format [x, y, width, height]
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    bbox = [x_min, y_min, width, height]
    return bbox

class single_image_vipseg_dataset(Dataset):
    def __init__(self, cfg):

        self.cfg = cfg
        
        vidname = cfg.data.vidname
        self.panoptic_categories, self.panoptic_videos, self.panoptic_annotations = self.load_json(cfg.data.anno_path)        

        vid_info = [vid_info for vid_info in self.panoptic_videos if vid_info["video_id"] == vidname][0]
        frame_num_in_video = cfg.data.frame_num_in_video
        image_id = sorted([img_info["id"] for img_info in vid_info["images"]])[frame_num_in_video] # e.g. 00000166

        img_path = os.path.join(cfg.data.data_path, vidname, image_id + '.jpg')


        inst_mask_path = os.path.join(cfg.data.panomasks_dir, vidname, image_id + '.png')
        rgb_mask_path = os.path.join(cfg.data.panomasksRGB_dir, vidname, image_id + '.png')
        self.video_annotations = [ann for ann in self.panoptic_annotations if ann['video_id'] == vidname][0]['annotations']

        self.proc = data_process.DataProcessor(cfg.data)
        self.data = self.load_tensor(img_path, cfg.data.data_shape, crop=cfg.data.crop)
        self.annotations = self.load_annos(
            inst_mask_path, rgb_mask_path, image_id, resize=cfg.data.data_shape, crop=cfg.data.crop
        )
        # Converts data to [N,3]
        # Features shape can be different from data shape as we might pad.
        self.features, self.features_shape = self.proc.get_features(self.data)

    def load_json(self, panoptic_gt_VIPSeg_path):
        with open(panoptic_gt_VIPSeg_path, 'r') as f:
            panoptic_gt_VIPSeg = json.load(f)
            
        panoptic_categories = panoptic_gt_VIPSeg['categories']
        panoptic_videos = panoptic_gt_VIPSeg['videos']
        panoptic_annotations = panoptic_gt_VIPSeg['annotations']    
        return panoptic_categories, panoptic_videos, panoptic_annotations
    
    def load_tensor(self, path, resize=None, crop=None):
        """
        Read image from path and return tensor.
        Use PIL
        """
        img = Image.open(path)
        img = img.convert("RGB")
        
        if crop is not None:
            (crop_h, crop_w) = (crop, crop) if isinstance(crop,int) else crop
            img = center_crop(img, (crop_h, crop_w))        
        if resize is None:
            img = transforms.ToTensor()(img)
        else:
            resize = (resize, resize) if isinstance(resize, int) else resize
            custom_transforms = tv.transforms.Compose(
                [tv.transforms.Resize(resize), tv.transforms.ToTensor()]
            )
            img = custom_transforms(img)
        return img

    def load_annos(self, inst_mask_path, rgb_mask_path, image_id, resize=None, crop=None):
        # Load the panoptic annotation for this image_id
        panoptic_annos = [ann for ann in self.video_annotations if ann['image_id'] == image_id][0]
        segments_info = panoptic_annos['segments_info']
        
        panop_mask = np.array(Image.open(inst_mask_path))
        rgb_panop_mask = np.array(Image.open(rgb_mask_path))
        
        ann_id = 0
        annotations = []
    
        categories = self.panoptic_categories
        categories.append({'id': -1, 'name': 'other', 'supercategory': '', 'color':None})
        categories_dict = {el['id']: el for el in categories}
        
        for el in np.unique(panop_mask):
            # Ignore background
            if el == 0:
                continue
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
            if categories_dict[semantic_id]['isthing'] == 0:
                is_crowd = 0
            
            # Binary mask
            binary_mask = (panop_mask == el)
            
            # Find instance ID from RGB values of colored mask
            r, g, b = np.unique(rgb_panop_mask[binary_mask], axis=0)[0]
            inst_id = r + g * 256 + b * 256**2
        
            segment_info = [seg_info for seg_info in segments_info if seg_info['id'] == inst_id][0]
            
            # Compute bbox from binary mask
            bbox = compute_bbox(binary_mask)
    
            anno = {
                    "id": ann_id,
                    "inst_id": inst_id,
                    "image_id": image_id,
                    "category_id": segment_info["category_id"],
                    "area": segment_info["area"],
                    "iscrowd": is_crowd,
                    "isthing": categories_dict[semantic_id]['isthing'],
                    "binary_mask": np.array(binary_mask, dtype=np.uint8),
                    "bbox": bbox,
                }
            
            annotations.append(anno)
            ann_id += 1
        
        if crop is None and resize is None:
            return annotations
        
        if crop is not None:
            (crop_h, crop_w) = (crop, crop) if isinstance(crop,int) else crop
        if resize is not None:
            resize_hw = (resize,resize) if isinstance(resize,int) else resize
        
        transformed_annotations = []

        for anno in annotations:
            binary_mask = anno["binary_mask"]
            if crop is not None:
                binary_mask = center_crop(torch.from_numpy(binary_mask), (crop_h, crop_w)).squeeze().numpy()
            if resize is not None:
                # Resize the binary mask
                binary_mask = resize_transform(
                    torch.from_numpy(binary_mask).unsqueeze(0), resize_hw, interpolation=InterpolationMode.NEAREST
                ).squeeze().numpy()
            
            binary_mask = binary_mask != 0
            if binary_mask.sum() == 0:
                # If the binary mask has no non-zero values, the instance is lost
                # after applying transformations so skip it
                continue
            
            # Calculate the bounding box coordinates
            bbox = compute_bbox(binary_mask)
            area = binary_mask.sum()
            
            transformed_ann = anno.copy()
            transformed_ann["bbox"] = bbox
            transformed_ann["area"] = area
            transformed_ann["binary_mask"] = binary_mask.squeeze()
            
            transformed_annotations.append(transformed_ann)

        
        return transformed_annotations

    def __len__(self):
        return 1

    def __getitem__(self, index):
        output = {
            "data": self.data,
            "features": self.features,
            "features_shape": torch.tensor(self.features_shape),
            "data_shape": torch.tensor(self.data.shape),
            "annotations": self.annotations,
        }
        return output