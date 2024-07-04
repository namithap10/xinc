import numpy as np
import torch
import torchvision as tv
from PIL import Image
from pycocotools.coco import COCO
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


class single_image_dataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        path = cfg.data.data_path

        self.proc = data_process.DataProcessor(cfg.data)
        # Crop the data if necessary
        self.data = self.load_tensor(path, cfg.data.data_shape, crop=cfg.data.crop)

        # Converts data to [N,3]
        # Features shape can be different from data shape as we might pad.

        self.features, self.features_shape = self.proc.get_features(
            self.data.unsqueeze(0), patch_shape=cfg.data.patch_shape
        )

    def load_tensor(self, path, resize=None, crop=None):
        """
        Read image from path and return tensor.
        Use PIL
        """
        img = Image.open(path)
        img = img.convert("RGB")
        if crop is not None:
            (crop_h, crop_w) = (crop, crop) if isinstance(crop, int) else crop
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

    def __len__(self):
        return 1

    def __getitem__(self, index):
        output = {
            "data": self.data,
            "features": self.features,
            "features_shape": torch.tensor(self.features_shape),
            "data_shape": torch.tensor(self.data.shape),
        }
        return output


class single_image_dataset_with_annos(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        path = cfg.data.data_path
        anno_path = cfg.data.anno_path
        image_id = cfg.data.image_id_in_anno

        self.proc = data_process.DataProcessor(cfg.data)
        self.data = self.load_tensor(path, cfg.data.data_shape, crop=cfg.data.crop)
        self.annotations = self.load_annotations(
            anno_path, image_id, cfg.data.data_shape, crop=cfg.data.crop
        )
        # Converts data to [N,3]
        # Features shape can be different from data shape as we might pad.
        self.features, self.features_shape = self.proc.get_features(self.data)

    def load_tensor(self, path, resize=None, crop=None):
        """
        Read image from path and return tensor.
        Use PIL
        """
        img = Image.open(path)
        img = img.convert("RGB")
        if crop is not None:
            (crop_h, crop_w) = (crop, crop) if isinstance(crop, int) else crop
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

    def load_annotations(self, anno_path, image_id, resize=None, crop=None):
        """
        Read annotations associated with image.
        Resize the binary mask and associated parameters accordingly.
        """
        coco = COCO(anno_path)
        ann_ids = coco.getAnnIds(imgIds=image_id)
        annotations = coco.loadAnns(ann_ids)
        if crop is None and resize is None:
            return annotations

        if crop is not None:
            (crop_h, crop_w) = (crop, crop) if isinstance(crop, int) else crop
        if resize is not None:
            resize_hw = (resize, resize) if isinstance(resize, int) else resize

        transformed_annotations = []

        for ann in annotations:
            binary_mask = coco.annToMask(ann)
            if crop is not None:
                binary_mask = (
                    center_crop(torch.from_numpy(binary_mask), (crop_h, crop_w))
                    .squeeze()
                    .numpy()
                )
            if resize is not None:
                # Resize the binary mask
                binary_mask = (
                    resize_transform(
                        torch.from_numpy(binary_mask).unsqueeze(0),
                        resize_hw,
                        interpolation=InterpolationMode.NEAREST,
                    )
                    .squeeze()
                    .numpy()
                )

            binary_mask = binary_mask != 0
            if binary_mask.sum() == 0:
                # If the binary mask has no non-zero values, the instance is lost
                # after applying transformations so skip it
                continue

            bbox = self.compute_bbox(binary_mask)
            area = binary_mask.sum()

            transformed_ann = ann.copy()
            transformed_ann["bbox"] = bbox
            transformed_ann["area"] = area
            transformed_ann["binary_mask"] = binary_mask.squeeze()
            del transformed_ann["segmentation"]

            transformed_annotations.append(transformed_ann)

        return transformed_annotations

    def compute_bbox(self, binary_mask):
        (rows, cols) = np.where(binary_mask > 0)
        x_min, x_max, y_min, y_max = min(cols), max(cols), min(rows), max(rows)
        # Create the bbox in COCO format [x, y, width, height]
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        bbox = [x_min, y_min, width, height]
        return bbox

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