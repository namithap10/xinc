import os
import pickle

import matplotlib.pyplot as plt
from torchvision.io import read_image
import torchvision.transforms as transforms

nerv_save_dir = '../plotting_source_data/NeRV/A-contribution_vs_intensity'
mlp_save_dir = '../plotting_source_data/MLP/A-contribution_vs_intensity'

nerv_per_vid_intensity_vs_heatmap_dict = pickle.load(open(f'{nerv_save_dir}/per_vid_intensity_vs_heatmap_dict.pkl', 'rb'))

mlp_per_vid_intensity_vs_heatmap_dict = pickle.load(open(f'{mlp_save_dir}/per_vid_intensity_vs_heatmap_dict.pkl', 'rb'))


rgb_img_paths = {
    "0005": "../data/cityscapes_vps/val/img_all/frankfurt_000000_001736_leftImg8bit.png",
    "0175": "../data/cityscapes_vps/val/img_all/frankfurt_000001_049683_leftImg8bit.png",
    "12_n-ytHkMceew": "../data/VIPSeg-Dataset/VIPSeg/VIPSeg_720P/images/12_n-ytHkMceew/00002537.jpg",
    "26_cblDl5vCZnw": "../data/VIPSeg-Dataset/VIPSeg/VIPSeg_720P/images/26_cblDl5vCZnw/00000976.jpg"
}

vipseg_vids = ["12_n-ytHkMceew", "26_cblDl5vCZnw"]

# Select the vidnames we want
vidnames = ["0005", "12_n-ytHkMceew"]# "26_cblDl5vCZnw"]

# size is 2*num_videos x rgb+num_layers (3 MLP + 2 NeRV) + one for cbar
fig, axs = plt.subplots(2*len(vidnames), 7, figsize=(16, 7), tight_layout=True, width_ratios=[1,1,1,1,1,1,0.2])

for vid_idx, vidname in enumerate(vidnames):    
    
    rgb_img = read_image(rgb_img_paths[vidname])
    if vidname in vipseg_vids:
        rgb_img = transforms.CenterCrop((640, 1280))(rgb_img)
    
    # Read the stored values
    head_heatmap = nerv_per_vid_intensity_vs_heatmap_dict[vidname]["heatmaps"]["head"]
    blk_3_heatmap = nerv_per_vid_intensity_vs_heatmap_dict[vidname]["heatmaps"]["blk_3"]
    layer_1_heatmap = mlp_per_vid_intensity_vs_heatmap_dict[vidname]["heatmaps"]["layer_1"]
    layer_2_heatmap = mlp_per_vid_intensity_vs_heatmap_dict[vidname]["heatmaps"]["layer_2"]
    layer_3_heatmap = mlp_per_vid_intensity_vs_heatmap_dict[vidname]["heatmaps"]["layer_3"]
    
    head_intensity_contrib_diff = nerv_per_vid_intensity_vs_heatmap_dict[vidname]["diffs_after_normalize"]["head"]
    blk_3_intensity_contrib_diff = nerv_per_vid_intensity_vs_heatmap_dict[vidname]["diffs_after_normalize"]["blk_3"]
    layer_1_intensity_contrib_diff = mlp_per_vid_intensity_vs_heatmap_dict[vidname]["diffs_after_normalize"]["layer_1"]
    layer_2_intensity_contrib_diff = mlp_per_vid_intensity_vs_heatmap_dict[vidname]["diffs_after_normalize"]["layer_2"]
    layer_3_intensity_contrib_diff = mlp_per_vid_intensity_vs_heatmap_dict[vidname]["diffs_after_normalize"]["layer_3"]
    
    
    rgb_intensity = nerv_per_vid_intensity_vs_heatmap_dict[vidname]["rgb_intensity"]
    
    # edge_map = nerv_per_vid_intensity_vs_heatmap_dict[vidname]["edge_map"]
    axs[vid_idx*2][0].imshow(rgb_img.permute(1,2,0).numpy())
    axs[vid_idx*2][0].set_title('Frame', fontsize=14, fontweight='bold')
    
    axs[vid_idx*2+1][0].imshow(rgb_intensity, cmap="viridis") # BuPu, YlOrBr is good
    axs[vid_idx*2+1][0].set_title('Total Intensity Map', fontsize=14, fontweight='bold')
    
    # Heatmaps
    im1 = axs[vid_idx*2][1].imshow(layer_3_heatmap, cmap="magma")
    im1 = axs[vid_idx*2][2].imshow(layer_2_heatmap, cmap="magma")
    im1 = axs[vid_idx*2][3].imshow(layer_1_heatmap, cmap="magma")
    im1 = axs[vid_idx*2][4].imshow(head_heatmap, cmap="magma")
    im1 = axs[vid_idx*2][5].imshow(blk_3_heatmap, cmap="magma")

    # Use diverging cmaps for showing difference (negatives to positives)
    im2 = axs[vid_idx*2+1][1].imshow(layer_3_intensity_contrib_diff, cmap="seismic", vmin=-1.0, vmax=1.0) # seismic or RdYlBu or spectral # Seismic is most clear
    im2 = axs[vid_idx*2+1][2].imshow(layer_2_intensity_contrib_diff, cmap="seismic", vmin=-1.0, vmax=1.0)
    im2 = axs[vid_idx*2+1][3].imshow(layer_1_intensity_contrib_diff, cmap="seismic", vmin=-1.0, vmax=1.0)
    im2 = axs[vid_idx*2+1][4].imshow(head_intensity_contrib_diff, cmap="seismic", vmin=-1.0, vmax=1.0)
    im2 = axs[vid_idx*2+1][5].imshow(blk_3_intensity_contrib_diff, cmap="seismic", vmin=-1.0, vmax=1.0)
    
    axs[0][1].set_title('FFN Layer 3', fontsize=14, fontweight='bold')
    axs[0][2].set_title('FFN Layer 2', fontsize=14, fontweight='bold')
    axs[0][3].set_title('FFN Layer 1', fontsize=14, fontweight='bold')
    
    axs[0][4].set_title('NeRV Head Layer', fontsize=14, fontweight='bold')
    axs[0][5].set_title('NeRV Block 3', fontsize=14, fontweight='bold')
    
    for ax in axs.flatten():
        ax.axis('off')


fig.subplots_adjust(right=0.8)
cbar_ax_magma = fig.add_axes([0.965, 0.54, 0.01, 0.4])
cbar_magma = fig.colorbar(im1, cax=cbar_ax_magma, cmap='magma', orientation='vertical', pad=0.02, aspect=50,) # ticks=[0, 5, 10, 15, 20, 25]
cbar_magma.set_label('Contribution Magnitude', rotation=270, labelpad=15, fontweight='bold')

fig.subplots_adjust(right=0.8)
cbar_ax_seismic = fig.add_axes([0.965, 0.05, 0.01, 0.4])
cbar_seismic = fig.colorbar(im2, cax=cbar_ax_seismic, cmap='seismic', orientation='vertical', pad=0.02, aspect=50,) # ticks=[-1.0, -0.5, 0.0, 0.5, 1.0]
cbar_seismic.set_label('Higher Contribution    Higher Intensity', rotation=270, labelpad=15, fontweight='bold')


from matplotlib import ticker  # Disable ticks on colorbars

cbar_magma.ax.yaxis.set_major_formatter(ticker.NullFormatter())
cbar_seismic.ax.yaxis.set_major_formatter(ticker.NullFormatter())
cbar_magma.ax.set_yticks([])
cbar_seismic.ax.set_yticks([])

os.makedirs('outputs/A/', exist_ok=True)
fig.savefig('outputs/A/4.1-contribution_vs_intensity_map.pdf', bbox_inches="tight", pad_inches=0)