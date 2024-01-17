import os
import pickle

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from matplotlib import gridspec
from torchvision.io import read_image

nerv_save_dir = '../plotting_source_data/NeRV/B-representation_is_distributed'
mlp_save_dir = '../plotting_source_data/MLP/B-representation_is_distributed'

nerv_per_vid_num_kernels_with_meaningful_contrib = pickle.load(open(f'{nerv_save_dir}/per_vid_num_kernels_with_meaningful_contrib.pkl', 'rb'))
mlp_per_vid_num_kernels_with_meaningful_contrib = pickle.load(open(f'{mlp_save_dir}/per_vid_num_kernels_with_meaningful_contrib.pkl', 'rb'))

rgb_img_paths = {
    "0005": "../data/cityscapes_vps/val/img_all/frankfurt_000000_001736_leftImg8bit.png",
    "0175": "../data/cityscapes_vps/val/img_all/frankfurt_000001_049683_leftImg8bit.png",
    "12_n-ytHkMceew": "../data/VIPSeg-Dataset/VIPSeg/VIPSeg_720P/images/12_n-ytHkMceew/00002537.jpg",
    "26_cblDl5vCZnw": "../data/VIPSeg-Dataset/VIPSeg/VIPSeg_720P/images/26_cblDl5vCZnw/00000976.jpg"
}

vipseg_vids = ["12_n-ytHkMceew", "26_cblDl5vCZnw"]

# Without using gridspec
vidnames = ["0005"]
target_areas = [0.1, 0.5]
vidname = vidnames[0]
titles_fontsize = 12


# Set up gridspec
fig = plt.figure(figsize=(16, 4))#, tight_layout=True)
gs = gridspec.GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 1])#, height_ratios=[1, 1])

# Plot the RGB image at the center of the first column
rgb_img = read_image(rgb_img_paths[vidname])
if vidname in vipseg_vids:
    rgb_img = transforms.CenterCrop((640, 1280))(rgb_img)
rgb_ax = fig.add_subplot(gs[:, 0])
rgb_ax.imshow(rgb_img.permute(1, 2, 0).numpy())
# rgb_ax.set_title("Ground Truth Image", fontsize=titles_fontsize)
rgb_ax.axis('off')

# Number of kernels is a percentage of total number in the layer - so that we can compare across layers

# Find the vmin, vmax setting - common across the plot

for target_area_idx, target_area in enumerate(target_areas):
    head_heatmap = nerv_per_vid_num_kernels_with_meaningful_contrib[vidname][target_area]["head"]
    blk_3_heatmap = nerv_per_vid_num_kernels_with_meaningful_contrib[vidname][target_area]["blk_3"]
    layer_1_heatmap = mlp_per_vid_num_kernels_with_meaningful_contrib[vidname][target_area]["layer_1"]
    layer_2_heatmap = mlp_per_vid_num_kernels_with_meaningful_contrib[vidname][target_area]["layer_2"]
    layer_3_heatmap = mlp_per_vid_num_kernels_with_meaningful_contrib[vidname][target_area]["layer_3"]

    if target_area_idx == 0:
        vmin = min(head_heatmap.min(), blk_3_heatmap.min(), layer_1_heatmap.min(), layer_2_heatmap.min(), layer_3_heatmap.min())
        vmax = max(head_heatmap.max(), blk_3_heatmap.max(), layer_1_heatmap.max(), layer_2_heatmap.max(), layer_3_heatmap.max())
    else:
        vmin = min(vmin, head_heatmap.min(), blk_3_heatmap.min(), layer_1_heatmap.min(), layer_2_heatmap.min(), layer_3_heatmap.min())
        vmax = max(vmax, head_heatmap.max(), blk_3_heatmap.max(), layer_1_heatmap.max(), layer_2_heatmap.max(), layer_3_heatmap.max())
        
# Plot other images in the remaining columns
for target_area_idx, target_area in enumerate(target_areas):
    axs = [fig.add_subplot(gs[target_area_idx, col_idx]) for col_idx in range(1, 5)]

    head_heatmap = nerv_per_vid_num_kernels_with_meaningful_contrib[vidname][target_area]["head"]
    blk_3_heatmap = nerv_per_vid_num_kernels_with_meaningful_contrib[vidname][target_area]["blk_3"]
    layer_1_heatmap = mlp_per_vid_num_kernels_with_meaningful_contrib[vidname][target_area]["layer_1"]
    layer_2_heatmap = mlp_per_vid_num_kernels_with_meaningful_contrib[vidname][target_area]["layer_2"]
    layer_3_heatmap = mlp_per_vid_num_kernels_with_meaningful_contrib[vidname][target_area]["layer_3"]
    
    im = axs[0].imshow(layer_3_heatmap, cmap="OrRd", vmin=vmin, vmax=vmax) # GnBu
    im = axs[1].imshow(layer_2_heatmap, cmap="OrRd", vmin=vmin, vmax=vmax)
    im = axs[2].imshow(head_heatmap, cmap="OrRd", vmin=vmin, vmax=vmax)
    im = axs[3].imshow(blk_3_heatmap, cmap="OrRd",vmin=vmin, vmax=vmax)

    if target_area_idx == 0:
        axs[0].set_title(f'FFN Layer 3', fontsize=titles_fontsize, fontweight='bold')
        axs[1].set_title(f'FFN Layer 2', fontsize=titles_fontsize, fontweight='bold')
        axs[2].set_title(f'NeRV Head Layer', fontsize=titles_fontsize, fontweight='bold')
        axs[3].set_title(f'NeRV Block 3', fontsize=titles_fontsize, fontweight='bold')

    save_dir = f'../inr_analysis/plot_figures/outputs/B/npp/thresh_{int(target_area*100)}'

    axs[-1].text(1.05, 0.5, f"Threshold {int(target_area * 100)}%", fontsize=10, transform=axs[-1].transAxes, rotation=-90,
                verticalalignment='center', horizontalalignment='left', fontweight='bold')

    for ax in axs:
        ax.axis('off')

fig.subplots_adjust(right=0.9)
cbar_ax_magma = fig.add_axes([0.925, 0.2, 0.01, 0.6])
cbar_magma = fig.colorbar(im, cax=cbar_ax_magma, cmap='OrRd', orientation='vertical', pad=0.02, aspect=50, ticks=[0.2, 0.5, 0.8])
cbar_magma.ax.set_yticklabels(['0.2', '0.5', '0.8'])
cbar_magma.set_label('% Neurons that Contribute', rotation=270, labelpad=12, fontweight='bold')

plt.subplots_adjust(wspace=0.05, hspace=-0.15)

os.makedirs('outputs/B/', exist_ok=True)
fig.savefig('outputs/B/4.2-neurons_per_pixel_v2.pdf', bbox_inches="tight", pad_inches=0)
