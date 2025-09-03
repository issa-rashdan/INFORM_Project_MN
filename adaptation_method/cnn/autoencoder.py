import torch
import os
import torch.nn as nn
import torch.nn.functional as F

def diversity_loss(latent):
    B, C, H, W = latent.size()
    # Permute to shape B x H x W x C and flatten: each row is a pixel's channel activations.
    latent_flat = latent.permute(0, 2, 3, 1).reshape(-1, C)
    
    # Since the latent output is already softmaxed, take the log of the probabilities.
    log_p = torch.log(latent_flat)
    
    # Create a uniform distribution: each channel has probability 1/C.
    uniform = torch.full_like(log_p, 1.0 / C)
       
    # Compute the KL divergence between the latent log-probabilities and the uniform distribution.
    kl_div = torch.nn.functional.kl_div(log_p, uniform, reduction='batchmean')
    
    # Maximum possible KL divergence for a one-hot distribution relative to a uniform distribution is log(C).
    # Subtract kl_div from log(C) so that minimizing the loss encourages maximized diversity.
    # max_kl_div = torch.log(torch.tensor(C, dtype=latent.dtype, device=latent.device))
    # return max_kl_div - kl_div    
    return - kl_div

def balanced_activation(latent):
    B, C, H, W = latent.size()
    # Aggregate over batch and spatial dimensions: shape (C, B*H*W)
    latent_agg = latent.transpose(0, 1).contiguous().view(C, -1)

    # Balanced activation: enforce similar average activation across channels using the entire batch
    channel_means = latent_agg.mean(dim=1)
    target_mean = channel_means.mean()
    bal_loss = F.mse_loss(channel_means, torch.full_like(channel_means, target_mean.item()))
    
    return bal_loss

def epoch_logger(type, epoch_log, epoch, max_epochs, avg_rec, rec_loss_threshold, recon_stats, latent_stats, result_path):
    if type == "train":
        filename = "training_log.txt"
    elif type == "test":
        filename = "testing_log.txt"
    else:
        raise ValueError("Type must be either 'train' or 'test'.")

    epoch_log += (
    f"\n[Epoch {epoch:03d}/{max_epochs}]\n"
    "-----------------------------------------------------------\n"
    )
    epoch_log += "\n".join(recon_stats) + "\n" + "\n".join(latent_stats) + "\n"

    # Build the log message depending on the reconstruction loss
    if avg_rec < rec_loss_threshold:
        message = f"\n-------------------- Early stopping at epoch {epoch:03d} --------------------\n"
    else:
        message = f"\n-------------------- End of Epoch {epoch:03d} --------------------\n"
    epoch_log += message

    log_file = os.path.join(result_path, filename)
    with open(log_file, "w") as f:
        f.write(epoch_log)
    return epoch_log



def train_epoch(model, dataloader, optimizer, div_criterion, bal_criterion, diversity_weight, balance_weight, device):
    # Ensure the model is in training mode      

    model.train()
    rec_losses, bal_losses, div_losses = [], [], []
    r_activations_min, g_activations_min, b_activations_min = [], [], []
    r_activations_mean, g_activations_mean, b_activations_mean = [], [], []
    r_activations_max, g_activations_max, b_activations_max = [], [], []
    latent_stats = []
    recon_stats = []

    for (batch_labels, batch_data, batch_masks) in dataloader:
        batch_labels, batch_data, batch_masks = batch_labels.squeeze(0).to(device), batch_data.squeeze(0).to(device), batch_masks.squeeze(0).to(device)
        recon, latent = model(batch_data)
        rec_loss = F.mse_loss(recon, batch_data)
        div_loss = div_criterion(latent)
        bal_loss = bal_criterion(latent)

        total_loss = rec_loss + bal_loss * balance_weight + div_loss * diversity_weight

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Accumulate stats
        rec_losses.append(rec_loss.item())
        bal_losses.append(bal_loss.item())
        div_losses.append(div_loss.item())

        # Latent channel stats
        for i, color in enumerate(["R", "G", "B"]):
            ch = latent[:, i, :, :]
            if i == 0:
                r_activations_min.append(ch.min().item())
                r_activations_mean.append(ch.mean().item())
                r_activations_max.append(ch.max().item())
            elif i == 1:
                g_activations_min.append(ch.min().item())
                g_activations_mean.append(ch.mean().item())
                g_activations_max.append(ch.max().item())
            elif i == 2:
                b_activations_min.append(ch.min().item())
                b_activations_mean.append(ch.mean().item())
                b_activations_max.append(ch.max().item())

    # Summary per epoch
    avg_rec = sum(rec_losses) / len(rec_losses)
    avg_bal = sum(bal_losses) / len(bal_losses)
    avg_div = sum(div_losses) / len(div_losses)

    recon_stats.append(
        f"Losses -> Rec: {avg_rec:.4f}, Bal: {avg_bal:.4f}, Div: {avg_div:.4f}\n"
    )

    # Compute latent channel statistics using a loop to avoid code repetition.
    channels = {
        "R": (r_activations_min, r_activations_mean, r_activations_max),
        "G": (g_activations_min, g_activations_mean, g_activations_max),
        "B": (b_activations_min, b_activations_mean, b_activations_max),
    }
    for ch, (mins, means, maxs) in channels.items():
        avg_min = sum(mins) / len(mins)
        avg_mean = sum(means) / len(means)
        avg_max = sum(maxs) / len(maxs)
        latent_stats.append(
            f"{ch}: min={avg_min:.4f}, mean={avg_mean:.4f}, max={avg_max:.4f}"
        )
    return avg_rec, recon_stats, latent_stats


# Evaluate the model on the test dataset every 5 epochs or at the final epoch
def test_epoch(model, dataloader, div_criterion, bal_criterion, device):
    # Ensure the model is in training mode      

    model.eval()
    rec_losses, bal_losses, div_losses = [], [], []
    r_activations_min, g_activations_min, b_activations_min = [], [], []
    r_activations_mean, g_activations_mean, b_activations_mean = [], [], []
    r_activations_max, g_activations_max, b_activations_max = [], [], []
    latent_stats = []
    recon_stats = []

    for batch_labels, batch_data, batch_masks in dataloader:
        batch_labels, batch_data, batch_masks = batch_labels.squeeze(0).to(device), batch_data.squeeze(0).to(device), batch_masks.squeeze(0).to(device)
        recon, latent = model(batch_data)
        rec_loss = F.mse_loss(recon, batch_data)
        div_loss = div_criterion(latent)
        bal_loss = bal_criterion(latent)
        
        # Accumulate stats
        rec_losses.append(rec_loss.item())
        bal_losses.append(bal_loss.item())
        div_losses.append(div_loss.item())

        # Latent channel stats
        for i, color in enumerate(["R", "G", "B"]):
            ch = latent[:, i, :, :]
            if i == 0:
                r_activations_min.append(ch.min().item())
                r_activations_mean.append(ch.mean().item())
                r_activations_max.append(ch.max().item())
            elif i == 1:
                g_activations_min.append(ch.min().item())
                g_activations_mean.append(ch.mean().item())
                g_activations_max.append(ch.max().item())
            elif i == 2:
                b_activations_min.append(ch.min().item())
                b_activations_mean.append(ch.mean().item())
                b_activations_max.append(ch.max().item())

    # Summary per epoch
    avg_rec = sum(rec_losses) / len(rec_losses)
    avg_bal = sum(bal_losses) / len(bal_losses)
    avg_div = sum(div_losses) / len(div_losses)

    recon_stats.append(
        f"Losses -> Rec: {avg_rec:.4f}, Bal: {avg_bal:.4f}, Div: {avg_div:.4f}\n"
    )

    # Compute latent channel statistics using a loop to avoid code repetition.
    channels = {
        "R": (r_activations_min, r_activations_mean, r_activations_max),
        "G": (g_activations_min, g_activations_mean, g_activations_max),
        "B": (b_activations_min, b_activations_mean, b_activations_max),
    }
    for ch, (mins, means, maxs) in channels.items():
        avg_min = sum(mins) / len(mins)
        avg_mean = sum(means) / len(means)
        avg_max = sum(maxs) / len(maxs)
        latent_stats.append(
            f"{ch}: min={avg_min:.4f}, mean={avg_mean:.4f}, max={avg_max:.4f}"
        )
    return avg_rec, recon_stats, latent_stats





# --- Autoencoder ---
class Autoencoder(nn.Module):
    def __init__(self, in_channels=6, latent_channels=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, latent_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, latent_channels),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=1)  # Apply softmax over channel dimension
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        # Randomly permute the channels of the latent representation.
        perm = torch.randperm(z.size(1), device=z.device)
        z_perm = z[:, perm, :, :]
        x_hat = self.decoder(z_perm)
        return x_hat, z_perm

