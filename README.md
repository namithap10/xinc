# Explaining the Implicit Neural Canvas (XINC): Connecting Pixels to Neurons by Tracing their Contributions (CVPR 2024)


This is the official implementation of [Explaining the Implicit Neural Canvas: Connecting Pixels to Neurons by Tracing their Contributions](https://arxiv.org/abs/2401.10217).

### [Paper](https://arxiv.org/pdf/2401.10217.pdf) | [Project Page](https://namithap10.github.io/xinc)
<br/>


# Setting up the environment
1. Clone this repository and navigate to it in your terminal.
1. Install python==3.9
1. `pip install -r requirements.txt`


# Generating Neuron Contribution Maps

For MLP-based INR analysis, we use a simple implementation of the [FFN](https://bmild.github.io/fourfeat/) network and for CNN-based INR analysis, we build upon the [HNeRV](https://github.com/haochen-rye/HNeRV) repository.

### FFN Neuron Contribution Maps
The script  `image_inr/get_mlp_mappings.py` provides functions for computing the contribution maps of FFN Layer 3, Layer 2 and Layer 1 neurons. It uses the intermediate outputs that are gathered from the FFN model (found in `image_inr/models/ffn.py`).

### NeRV Neuron Contribution Maps
The script `HNeRV/get_mappings.py` provides functions for computing the contribution maps for NeRV head layer and inner block neurons. It uses the intermediate outputs that are gathered from the HNeRV model (found in `HNeRV/model_all.py`).


# Datasets and Dataloaders

Download and place the panoptic video datasets in a folder called `data/` at the root of the code folder. Download the [Cityscapes-VPS](https://github.com/mcahny/vps) dataset into `data/cityscapes_vps` and [VIPSeg](https://github.com/VIPSeg-Dataset/VIPSeg-Dataset) dataset into `data/VIPSeg-Dataset` by following their respective instructions.

The directory structure should be organized as follows:
```
./xinc/
├── checkpoints/
├── cluster_utils/
└── data/
    ├── cityscapes_vps/ 
    └── VIPSeg-Dataset/
├── HNeRV/
└── image_inr/ 

```

### Image datasets - For MLP-based models
Single image dataset class definitions (with panoptic annotations) for Cityscapes-VPS and VIPSeg can be found in `image_inr/image_vps_datasets.py`.

### Video datasets - For NeRV
Cityscapes-VPS and VIPSeg video dataset class definitions (with panoptic annotations) can be found in `HNeRV/vps_datasets.py`.

# Training

### FFN
MLP training framework (allows training various types of MLP-based INRs including FFN) - `image_inr/main.py`. The associated config files can be found in image_inr/configs.

*Example command*
```bash
python main.py data.data_shape=[128,256] data.data_path=<path to image file> network=ffn network.layer_size=104  trainer.num_iters=1000 logging.checkpoint.logdir=<path to save folder>
```

A sample FFN model trained on the first frame of the Cityscapes-VPS video "0005" is provided in [checkpoints/](checkpoints/). For a fair comparison between NeRV and FFN, we set the FFN layer size to be 104 in order to match their bpp values, but this can be modified.

### NeRV
The original [NeRV/HNeRV](https://github.com/haochen-rye/HNeRV) training script with modifications necessary for INR analysis experiments - `HNeRV/train_nerv_all.py`.

*Example command*
```bash
python train_nerv_all.py --outf <path to save folder> --data_path <path to video> --conv_type convnext pshuffel --act gelu --norm none  --crop_list 640_1280 --resize_list 128_256 --embed pe_1.25_80 --fc_hw 8_16 --dec_strds 4 2 2 --ks 0_3_3 --reduce 1.2 --skip_mssim --modelsize 1.0  -e <num_epochs> --lower_width 6 -b 2 --lr <lr>
```

A sample NeRV model trained on Cityscapes-VPS video "0005" is provided in [checkpoints](checkpoints/).


# Analysis
Some examples of MLP INR analysis can be found under `image_inr/analysis` and the CNN INR analysis examples are located in `HNeRV/analysis`.
These show examples of analyzing contribution maps to investigate the properties of implicit models. Further processing can be performed on the generated results to create visualizations similar to those presented in our paper.

For generating vector representations of neurons and grouping neurons, please refer to [cluster_utils](cluster_utils/).


# Citation
If you find our work useful, please consider citing:

```
@InProceedings{Padmanabhan_2024_CVPR,
    author    = {Padmanabhan, Namitha and Gwilliam, Matthew and Kumar, Pulkit and Maiya, Shishira R and Ehrlich, Max and Shrivastava, Abhinav},
    title     = {Explaining the Implicit Neural Canvas: Connecting Pixels to Neurons by Tracing their Contributions},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {10957-10967}
}
```