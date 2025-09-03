import torch
import numpy as np

def split_rgb_features(img_tensor, patch_size=224):
    if img_tensor.shape[2] == 3:
        img_tensor = img_tensor.permute(2, 0, 1)  # Now [C, H, W]

    # Padding if necessary
    pad_h = (patch_size - (img_tensor.size(1) % patch_size)) % patch_size
    pad_w = (patch_size - (img_tensor.size(2) % patch_size)) % patch_size

    # Pad the image
    padded_img = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h))

    # Unfold to get patches
    patches = padded_img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(3, -1, patch_size, patch_size)
    patches = patches.permute(1, 0, 2, 3)  # Shape [num_patches, 3, 224, 224]

    # Calculate indices for reconstruction
    indices = []
    num_patches_h = padded_img.size(1) // patch_size
    num_patches_w = padded_img.size(2) // patch_size

    for i in range(num_patches_h):
        for j in range(num_patches_w):
            indices.append((i * patch_size, j * patch_size))

    return patches, indices, (padded_img.size(1), padded_img.size(2))  # return the shape instead of padded_img


def split_label(label, patch_size=224):
    """
    Splits a single-channel label image into patches of the specified size.

    Parameters:
    label (torch.Tensor): The label tensor of shape [H, W] or [1, H, W].
    patch_size (int): The size of each patch. Default is 224.

    Returns:
    patches (np.ndarray): The patches array of shape [num_patches, 1, patch_size, patch_size].
    indices (list): List of top-left indices of each patch in the original image.
    padded_size (tuple): The height and width of the padded label image.
    """

    # Ensure label is [1, H, W] by adding a channel dimension if necessary
    if label.ndim == 2:
        label = np.expand_dims(label, axis=0)  # Shape becomes [1, H, W]

    # Padding if necessary
    H, W = label.shape[1], label.shape[2]
    pad_h = (patch_size - (H % patch_size)) % patch_size
    pad_w = (patch_size - (W % patch_size)) % patch_size

    # Pad the label
    padded_label = np.pad(label, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    # Unfold to get patches
    num_patches_h = padded_label.shape[1] // patch_size
    num_patches_w = padded_label.shape[2] // patch_size
    patches = padded_label.reshape(1, num_patches_h, patch_size, num_patches_w, patch_size)
    patches = patches.transpose(1, 3, 0, 2, 4).reshape(-1, 1, patch_size, patch_size)  # Shape: [num_patches, 1, patch_size, patch_size]

    # Calculate indices for reconstruction
    indices = [(i * patch_size, j * patch_size) for i in range(num_patches_h) for j in range(num_patches_w)]

    return patches, indices, (padded_label.shape[1], padded_label.shape[2])


# def split_rgb_features(img_tensor, patch_size=224):

#     if len(img_tensor.shape[2]) == 3:
#         img_tensor = img_tensor.permute(2, 0, 1)  # Now [C, H, W]

#     # Padding if necessary
#     pad_h = (patch_size - (img_tensor.size(1) % patch_size)) % patch_size
#     pad_w = (patch_size - (img_tensor.size(2) % patch_size)) % patch_size

#     # Pad the image
#     padded_img = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h))

#     # Unfold to get patches
#     patches = padded_img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
#     patches = patches.contiguous().view(3, -1, patch_size, patch_size)
#     patches = patches.permute(1, 0, 2, 3)  # Shape [num_patches, 3, 224, 224]

#     return patches, padded_img


def recon_rgb_features(patches, indices, padded_shape, patch_size=224):
    # Unpack padded_shape to get the height and width of the padded image
    padded_h, padded_w = padded_shape

    # Initialize an empty tensor for the reconstruction
    reconstructed = torch.zeros((3, padded_h, padded_w))

    # Iterate through patches and their corresponding indices to place them back
    for patch, (i, j) in zip(patches, indices):
        reconstructed[:, i:i+patch_size, j:j+patch_size] = patch

    # Permute back to [H, W, C]
    reconstructed = reconstructed.permute(1, 2, 0)

    # Crop back to the original image size if padding was added
    return reconstructed


# def recon_rgb_features(padded_img, patch_size=224):

#     num_patches_h = padded_img.size(1) // patch_size
#     num_patches_w = padded_img.size(2) // patch_size

#     patches = patches.permute(1, 0, 2, 3).contiguous().view(3, num_patches_h, num_patches_w, patch_size, patch_size)
#     reconstructed = patches.contiguous().view(3, num_patches_h * patch_size, num_patches_w * patch_size)

#     # Permute back to [H, W, C]
#     reconstructed = reconstructed.permute(1, 2, 0)

#     # Crop back to the original image size if padding was added
#     reconstructed = reconstructed[:img_tensor.size(1), :img_tensor.size(2), :]

#     # Now `reconstructed` should match the original image tensor
#     print(reconstructed.shape)