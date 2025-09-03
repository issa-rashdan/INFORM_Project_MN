# import numpy as np
# import core ML and utility libraries
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import multiprocessing as mp
from multiprocessing import Semaphore, Process, current_process
from multiprocessing.resource_tracker import unregister
import faulthandler

# Just declare how many concurrent slots you want
# Enable Python fault handler to print a C-level traceback on trap
faulthandler.enable()


# import data loading functions for echograms
from data.echosounder_data.load_data.get_echograms import get_echograms, count_classes_in_echograms
# import utilities to group patches into batches
from data.echosounder_data.dataloader import group_generators_by_patch_limit
from data.echosounder_data.preprocessing.resize_and_crop import generate_patch_batches
from data.echosounder_data.dataloader import BatchGeneratorDataset  # wraps callables into a Dataset
# normalization transforms for echogram values
from data.echosounder_data.preprocessing.normalization import db, clip, standardize_min_max

# import the Autoencoder model and training/validation helpers
from adaptation_method.cnn.autoencoder import Autoencoder, balanced_activation, diversity_loss, train_epoch, test_epoch, epoch_logger



def worker_task(sem: Semaphore, data_chunk):
    sem.acquire()
    try:
        # This section can run only in up to MAX_CONCURRENT processes at once
        print(f"{current_process().name} processing {data_chunk}")
        # … perform work …
    finally:
        sem.release()



def main():

    """
    Entry point for training/testing the Autoencoder on echogram data.
    All of your existing logic lives in the try block; cleanup happens in finally.
    """
    # 1. Create the semaphore once, before any DataLoader or Process starts
    MAX_CONCURRENT = os.cpu_count() #os.cpu_count()  # number of CPU cores for DataLoader workers    
    sem = Semaphore(MAX_CONCURRENT)

    try:

        num_workers = 4 

        # control verbosity of printed output
        verbose = True

        # specify training and test data years (or 'all' for multiple years)
        tr_years = 2014  
        te_years = 2017  

        # minimum dimension constraint for echogram images
        minimum_shape = 224

        # set of frequency tuples to include
        tuple_frequencies = (18, 38, 70, 120, 200, 333)

        # weightings for balancing terms in loss
        balance_weights = [0, 1, 5, 10]
        diversity_weights = [0, 0.1, 0.2, 0.5, 1]

        # stopping criteria for reconstruction loss
        rec_loss_threshold = 0.01

        # maximum number of epochs to train
        max_epochs = 100  

        # normalization/transforms to apply to each patch
        data_transforms = [db, clip, standardize_min_max]

        # batch control: maximum patches per batch 
        batch_limit = 400  

        # patch sizes before and after resizing
        split_patch_size = 224  
        output_patch_size = 224  


        # load lists of echogram objects for training and testing
        tr_echograms = get_echograms(
            years=tr_years,
            tuple_frequencies=tuple_frequencies,
            minimum_shape=minimum_shape
        )
        te_echograms = get_echograms(
            years=te_years,
            tuple_frequencies=tuple_frequencies,
            minimum_shape=minimum_shape
        )
        print(f"Number of TR echograms: {len(tr_echograms)}")
        print(f"Number of TE echograms: {len(te_echograms)}")

        # optionally display class distribution in labels
        if verbose:
            print("TR: ", count_classes_in_echograms(tr_echograms))
            print("TE: ", count_classes_in_echograms(te_echograms))

        # generate per-echogram patch batch generators (full batches)
        tr_generators, tr_num_patches_per_echogram = generate_patch_batches(
            tr_echograms,
            split_patch_size=split_patch_size,
            output_patch_size=output_patch_size,
            data_transforms=data_transforms,
            batch_size="full",
            verbose=False
        )
        te_generators, te_num_patches_per_echogram = generate_patch_batches(
            te_echograms,
            split_patch_size=split_patch_size,
            output_patch_size=output_patch_size,
            data_transforms=data_transforms,
            batch_size="full",
            verbose=False
        )

        # len(tr_grouped_gen_fns) = number of batches regarding batch_limit
        tr_grouped_gen_fns, tr_grouped_patch_counts = group_generators_by_patch_limit(tr_generators,
                                                                                        tr_num_patches_per_echogram,
                                                                                        batch_limit)
        te_grouped_gen_fns, te_grouped_patch_counts = group_generators_by_patch_limit(te_generators,
                                                                                        te_num_patches_per_echogram,
                                                                                        batch_limit)


        # Suppose tr_grouped_gen_fns is your list of callables
        tr_dataset = BatchGeneratorDataset(tr_grouped_gen_fns)
        te_dataset = BatchGeneratorDataset(te_grouped_gen_fns)

        # keep default batch size of 1, as it includes all patches from the grouped generators
        tr_loader = DataLoader(
            tr_dataset,
            shuffle=True,       # shuffle the order of generator‐batches each epoch
            num_workers=num_workers,      # adjust to your CPU cores
            pin_memory=False     # speeds up GPU transfer if you’re on CUDA
        )

        te_loader = DataLoader(
            te_dataset,
            shuffle=True,       # shuffle the order of generator‐batches each epoch
            num_workers=num_workers,      # adjust to your CPU cores
            pin_memory=False     # speeds up GPU transfer if you’re on CUDA
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[INFO] Using device: {device}")

        for balance_weight in balance_weights:
            for diversity_weight in diversity_weights:
                results_dir = "/Users/cch031/Documents/GitHub/INFORM_marine/results/decorrelated_but_balanced"
                subdir_name = f"b_{balance_weight}_d_{diversity_weight}"
                result_path = os.path.join(results_dir, subdir_name)

                if os.path.exists(result_path):
                    counter = 1
                    new_result_path = f"{result_path}_{counter}"
                    while os.path.exists(new_result_path):
                        counter += 1
                        new_result_path = f"{result_path}_{counter}"
                    result_path = new_result_path
                os.makedirs(result_path)
                tr_epoch_log = (f"Training with balance_weight: {balance_weight}, diversity_weight: {diversity_weight}\n"
                            f"Results directory created: {result_path}"
                            "\n-----------------------------------------------------------\n")
                te_epoch_log = (f"Test with balance_weight: {balance_weight}, diversity_weight: {diversity_weight}\n"
                            f"Results directory created: {result_path}"
                            "\n-----------------------------------------------------------\n")

                # Training control flag
                rerun_training = True

                # Main training loop with rerun support
                while rerun_training:
                    rerun_training = False  # Will only be set to True again if retraining is triggered

                    # (Re)initialize model and optimizer
                    model = Autoencoder(in_channels=6, latent_channels=3).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

                    for epoch in range(1, max_epochs + 1):

                        tr_avg_rec, tr_recon_stats, tr_latent_stats = train_epoch(
                            model=model,
                            dataloader=tr_loader, 
                            optimizer=optimizer, 
                            div_criterion=diversity_loss, 
                            bal_criterion=balanced_activation, 
                            diversity_weight=diversity_weight, 
                            balance_weight=balance_weight,
                            device=device)

                        tr_epoch_log = epoch_logger("train", tr_epoch_log, epoch, max_epochs, tr_avg_rec, rec_loss_threshold, tr_recon_stats, tr_latent_stats, result_path)
                        if verbose:
                            print(tr_epoch_log)

                        if epoch % 5 == 0 or epoch == max_epochs:
                            te_avg_rec, te_recon_stats, te_latent_stats = test_epoch(
                                model=model,
                                dataloader=te_loader, 
                                div_criterion=diversity_loss, 
                                bal_criterion=balanced_activation, 
                                device=device)
                            te_epoch_log = epoch_logger("test", te_epoch_log, epoch, max_epochs, te_avg_rec, rec_loss_threshold, te_recon_stats, te_latent_stats, result_path)                    

                        # If the reconstruction loss is below the threshold, exit the training loop
                        if tr_avg_rec < rec_loss_threshold:
                            te_avg_rec, te_recon_stats, te_latent_stats = test_epoch(
                                model=model,
                                dataloader=te_loader, 
                                div_criterion=diversity_loss, 
                                bal_criterion=balanced_activation, 
                                device=device)
                            te_epoch_log = epoch_logger("test", te_epoch_log, epoch, max_epochs, te_avg_rec, rec_loss_threshold, te_recon_stats, te_latent_stats, result_path)                    
                            # Save the trained model when reconstruction loss passes the threshold
                            model_file = os.path.join(
                                result_path,
                                f"autoencoder_b{balance_weight}_d{diversity_weight}_epoch{epoch}.pt"
                            )
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'tr_avg_rec': tr_avg_rec,
                                'te_avg_rec': te_avg_rec,
                                'balance_weight': balance_weight,
                                'diversity_weight': diversity_weight
                            }, model_file)

                            print(f"[INFO] Model checkpoint saved to: {model_file}")
                            tr_epoch_log += f"\nModel checkpoint saved to: {model_file}\n"
                            te_epoch_log += f"\nModel checkpoint saved to: {model_file}\n"
                            break

                        if epoch == max_epochs+1 and tr_avg_rec >= 0.01:
                            print(f"Full epoch completed with rec_loss {tr_avg_rec:.4f} (>= 0.01). Restarting training with the same setup.")
                            # Reset any necessary state (if applicable) and restart the epoch loop.
                            # If this training loop is enclosed in an outer while-loop, use 'break' or 'continue' as needed.
                            # For example, if using a while-loop for re-running the experiment, you can:
                            rerun_training = True  # Flag to signal that training must restart
                            break
    finally:
        try:
            unregister(sem._semlock.name, 'semaphore')
        except Exception:
            pass

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()



"""
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################


############################################################################
# visualize the echogram
import matplotlib.pyplot as plt
import numpy as np

# Retrieve the number of patch rows and columns defined in the ROI
n_rows_per_type = split_and_resize.ROI['num_patch_height']  # number of sub-rows per type
n_cols_per_type = split_and_resize.ROI['num_patch_width']     # number of patches per sub-row
total_patches = n_rows_per_type * n_cols_per_type           # total patches per type
img_scale = 2

# We have 5 figure types: 
#   0: label, 1: input data, 2: reconstruction, 3: latent (compressed image), 4: mask
n_types = 5
total_fig_rows = n_types * n_rows_per_type
total_fig_cols = n_cols_per_type

# Create one figure that will display all patches for all 5 types
fig, axs = plt.subplots(total_fig_rows, total_fig_cols, 
                        figsize=(total_fig_cols * img_scale, total_fig_rows * img_scale))

for channel in [5]: # [0, 1, 2, 3, 4, 5]:
    # Iterate over the 5 types
    for t in range(n_types):
        # For each type, iterate over the total patches in order
        for idx in range(total_patches):
            # Determine the row and column location inside the subgrid for this type
            sub_row = idx // n_cols_per_type     # row index within the current type block
            sub_col = idx % n_cols_per_type      # column index within the current type block
            
            # Compute the global row in the figure
            global_row = t * n_rows_per_type + sub_row
            ax = axs[global_row, sub_col]
            
            if t == 0:
                # Display the label using viridis colormap
                ax.imshow(label[idx], cmap='viridis', vmin=-1, vmax=2)
            elif t == 1:
                # Display the compressed 3-channel latent image
                latent_img = latent[idx].cpu().detach().numpy().transpose(1, 2, 0)
                ax.imshow(latent_img, vmin=0, vmax=1)
            elif t == 2:
                # Display reconstruction (choosing channel 0) in grayscale
                ax.imshow(recon[idx, channel, :, :].cpu().detach().numpy(), cmap='gray', vmin=0, vmax=1)
            elif t == 3:
                # Display input data (choosing channel 0) in grayscale
                ax.imshow(data[idx, channel, :, :].cpu().detach().numpy(), cmap='gray', vmin=0, vmax=1)
            elif t == 4:
                # Display the mask in grayscale
                ax.imshow(mask[idx], cmap='gray', vmin=0, vmax=1)
        
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(result_path, 'echogram_updated_%d.jpg' % channel), dpi=300)
    plt.close()


# Create subplots: total rows = num_types * n_rows_per_type, columns = n_cols
num_types = 5  # 0: full, 1: red, 2: green, 3: blue
fig, axs = plt.subplots(num_types * n_rows_per_type, n_cols_per_type, 
                        figsize=(n_cols_per_type * img_scale, num_types * n_rows_per_type * img_scale))

# Loop over each type and place images in the subgrid
for t in range(num_types):
    for idx in range(n_rows_per_type * n_cols_per_type):
        global_row = t * n_rows_per_type + (idx // n_cols_per_type)
        col = idx % n_cols_per_type

        # If idx exceeds the number of latent images, disable the axis
        if idx >= total_patches:
            axs[global_row, col].axis("off")
            continue

        # Convert latent to a NumPy image: (H, W, 3)
        latent_np = latent[idx].cpu().detach().numpy().transpose(1, 2, 0)
        
        if t == 0:  # Label
            axs[global_row, col].imshow(label[idx], cmap='viridis', vmin=-1, vmax=2)
        elif t == 1:  # Full 3-channel image
            axs[global_row, col].imshow(latent_np, vmin=0, vmax=1)
        elif t == 2:  # Red channel only
            axs[global_row, col].imshow(latent_np[..., 0], cmap='Reds', vmin=0, vmax=1)
        elif t == 3:  # Green channel only
            axs[global_row, col].imshow(latent_np[..., 1], cmap='Greens', vmin=0, vmax=1)
        elif t == 4:  # Blue channel only
            axs[global_row, col].imshow(latent_np[..., 2], cmap='Blues', vmin=0, vmax=1)
            

        axs[global_row, col].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(result_path, 'latent_channels_highlight.jpg'), dpi=300)
plt.close()




####################################################################################################
'''
# Visualize Cumulative Histogram of Echogram Data
#
# This section generates and displays the cumulative histogram of the preprocessed echogram data.
# The histogram helps visualize the distribution of data values, where only the specified percentile
# (e.g., 99th) is emphasized.
from data.echosounder_data.preprocessing.visualize_cmf import visualize_cumulative_histogram
import os
import numpy as np
visualize_cumulative_histogram(data=data.cpu().detach().numpy(), min_val=0, max_val=1, percentile=99)
# visualize_cumulative_histogram(data=data, min_val=0, max_val=1, percentile=99)
'''

####################################################################################################
# Choose pretrained ViT architecture: S, B, L
# 
# 


####################################################################################################
# Download weights trained by SSL: DINO, DINOv2, MAE, and so on
# 
# 


####################################################################################################
# Choose inference method: Louvain clustering, K-NN, linear probing, etc.





####################################################################################################


"""