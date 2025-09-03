import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

def visualize_all_in_subplots(raw_echo_np, full_image_np, label, upscaled_graph_structure, full_matrix, echosubidx, filedir='./', cmap_labels=None, norm_labels=None):
    """
    Visualize multiple plots (full image, label patches, graph structure, etc.) in one figure with subplots.
    
    Args:
        full_image_np: The full concatenated image as a NumPy array.
        label: The label patches.
        upscaled_graph_structure: The upscaled graph structure tensor.
        full_matrix: The weighted graph matrix.
        cmap_labels: Colormap for the labels.
        norm_labels: Normalization for the label colormap.
    """
    if cmap_labels is None:
        cmap_labels = mcolors.ListedColormap(['yellow', 'black', 'red', 'green'])
    if norm_labels is None:
        boundaries_labels = [-200, -0.5, 0.5, 1.5, 2.5]
        norm_labels = mcolors.BoundaryNorm(boundaries_labels, cmap_labels.N, clip=True)

    os.makedirs(filedir, exist_ok=True)

    # Define figure size based on image dimensions
    figsize = (full_image_np.shape[1] // 50, full_image_np.shape[0] // 10)
    
    # Create the figure and axes for subplots
    fig, axs = plt.subplots(7, 1, figsize=figsize)
    
    # Subplot 1: Label patches
    axs[0].imshow(label, cmap=cmap_labels, norm=norm_labels)
    axs[0].axis('off')
    axs[0].set_title('Label Patches')
    
    # Subplot 2: RAW image
    axs[1].imshow(raw_echo_np[:, :, -1])
    axs[1].axis('off')
    axs[1].set_title('RAW 330kHz')

    # Subplot 2: Full concatenated image
    axs[2].imshow(full_image_np)
    axs[2].axis('off')
    axs[2].set_title('RGB Image')
    
    # Subplot 3: Upscaled graph structure
    axs[3].imshow(upscaled_graph_structure[:full_image_np.shape[0], :full_image_np.shape[1], :].cpu().numpy())
    axs[3].axis('off')
    axs[3].set_title('Upscaled Graph Structure')
    
    # Subplot 4: Overlay of label patches on full image
    axs[4].imshow(full_image_np)
    axs[4].imshow(label, cmap=cmap_labels, norm=norm_labels, alpha=0.3)
    axs[4].axis('off')
    axs[4].set_title('Labels on Image')
    
    # Subplot 5: Overlay of graph structure on label patches
    axs[5].imshow(upscaled_graph_structure[:full_image_np.shape[0], :full_image_np.shape[1], :].cpu().numpy())
    axs[5].imshow(label, cmap=cmap_labels, norm=norm_labels, alpha=0.3)
    axs[5].axis('off')
    axs[5].set_title('Graph Structure on Labels')

    # Subplot 6: RAW image  + graph
    axs[6].imshow(raw_echo_np[:, :, -1])
    axs[6].imshow(upscaled_graph_structure[:full_image_np.shape[0], :full_image_np.shape[1], :].cpu().numpy(), alpha=0.3)
    axs[6].axis('off')
    axs[6].set_title('RAW 330kHz')

    # Adjust layout and show the figure
    plt.tight_layout()
    plt.savefig(os.path.join(filedir, 'fig_structure_%d.png' %(echosubidx)))

    # Create a new figure for the Weighted Graph Matrix with figsize 10x10
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Subplot 6: Weighted graph matrix
    ax.imshow(full_matrix[0].cpu().numpy(), vmin=0, vmax=1)
    ax.set_title('Weighted Graph Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(filedir, 'fig_kerel_matrix_%d.png' %(echosubidx)))


