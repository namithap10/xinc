# Explaining the Implicit Neural Canvas (XINC): Connecting Pixels to Neurons by Tracing their Contributions


This is the official implementation of the paper "[Explaining the Implicit Neural Canvas: Connecting Pixels to Neurons by Tracing their Contributions](https://arxiv.org/abs/2401.10217)" 

### [Paper](https://arxiv.org/pdf/2401.10217.pdf) | [Project Page](https://namithap10.github.io/xinc)
<br/>


This README provides information on the scripts for constructing neuron contribution maps, training models and analysis. For MLP-based INR analysis, we use a simple implementation of the [FFN](https://bmild.github.io/fourfeat/) network and for CNN-based INR analysis, we build upon code from the [HNeRV](https://github.com/haochen-rye/HNeRV) repository.


## Setting up the environment
1. Clone this repository and navigate to it in your terminal. 
1. Install python==3.9
1. `pip install -r requirements.txt` 

# Generating Contribution Maps

### FFN Neuron Contribution Maps
The script  `image_inr/get_mlp_mappings.py` provides functions for computing the contribution maps of FFN Layer 3, Layer 2 and Layer 1 neurons. It uses the intermediate results that are gathered from the FFN model class (found in `image_inr/models/ffn.py`).

### NeRV Neuron Contribution Maps
The script `HNeRV/get_mappings.py` provides functions for computing the contribution maps for all NeRV head layer and Block 3 neurons. It uses the intermediate network results that are gathered from the HNeRV model class (found in `HNeRV/model_all.py`).


# Datasets and Dataloaders

To use the dataloaders, please download and place the panoptic video datasets in a folder called `data/` at the root of the code folder. Download the [Cityscapes-VPS](https://github.com/mcahny/vps) dataset into `data/cityscapes_vps` and [VIPSeg](https://github.com/VIPSeg-Dataset/VIPSeg-Dataset) dataset into `data/VIPSeg-Dataset` by following the instructions on their respective websites.

### Image datasets - for MLP-based models
CityscapesVPS and VIPSeg single image dataset definitions (with panoptic annotations) can be found in `image_inr/image_vps_datasets.py`.

### Video datasets - For NeRV
CityscapesVPS, VIPSeg video dataset definitions (with panoptic annotations) and Flow-based dataset definition can be found in `HNeRV/vps_datasets.py`.

# Training

### MLP
MLP training framework (allows training various types of MLP-based INRs including FFN) - `image_inr/main.py`. The associated config files can be found in image_inr/configs.

*Example command*
```bash
python main.py data.data_shape=[128,256] data.data_path=<path to image file> network=ffn  network.layer_size=104  trainer.num_iters=1000 logging.checkpoint.logdir=<path to save folder>
```

### NeRV
The original [NeRV/HNeRV](https://github.com/haochen-rye/HNeRV) training script with modifications necessary for INR analysis experiments - `HNeRV/train_nerv_all.py`.

*Example command*
```bash
python train_nerv_all.py --outf <path to save folder> --data_path <path to video> --conv_type convnext pshuffel --act gelu --norm none  --crop_list 640_1280 --resize_list 128_256 --embed pe_1.25_80 --fc_hw 8_16 --dec_strds 4 2 2 --ks 0_3_3 --reduce 1.2 --skip_mssim --modelsize 1.0  -e <num_epochs> --lower_width 6 -b 2 --lr <lr>
```

# Analysis
The MLP INR analysis iPython notebooks and scripts can be found under `image_inr/analysis`. The CNN INR analysis code is contained in iPython notebooks and scripts in the `HNeRV/analysis` directory. 

## Citation
If you find our work useful, please consider citing:

```
@misc{padmanabhan2024explaining,
    title={Explaining the Implicit Neural Canvas: Connecting Pixels to Neurons by Tracing their Contributions}, 
    author={Namitha Padmanabhan and Matthew Gwilliam and Pulkit Kumar and Shishira R Maiya and Max Ehrlich and Abhinav Shrivastava},
    year={2024},
    eprint={2401.10217},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
