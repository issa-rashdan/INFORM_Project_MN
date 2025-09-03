import torch
import torch.nn.functional as F

def pagerank(A, num_iterations=100, d=0.85, tolerance=1e-6):
    """
    Computes the PageRank of nodes in a graph represented by matrix A using the power method.

    Parameters:
    - A: square, column-stochastic matrix representing the graph (web link structure).
    - num_iterations: maximum number of iterations to perform.
    - d: damping factor, usually set to 0.85.
    - tolerance: convergence tolerance.

    Returns:
    - rank: the PageRank vector, representing the importance of each node.
    """
    if len(A.shape)==3 and A.shape[0]==1:
        A.squeeze_(0)
    A = A / A.sum(dim=0, keepdim=True)  # Normalize columns to sum to 1
    N = A.size(0)
    # Initialize with equal probability to each page
    rank = torch.ones(N, dtype=A.dtype, device=A.device) / N
    # Teleportation vector (to deal with dead-ends and spider traps)
    teleport = torch.ones(N, dtype=A.dtype, device=A.device) / N
    
    for _ in range(num_iterations):
        new_rank = d * torch.mv(A, rank) + (1 - d) * teleport
        # Check for convergence
        if torch.norm(new_rank - rank) < tolerance:
            break
        rank = new_rank

    return rank

def masking_out_features(mask_tensor, all_embeddings, patch_size, token_size):
    # Step 1: Zero-pad the mask to match a size that can be split into 448x448 patches
    pad_x = (patch_size - (mask_tensor.shape[1] % patch_size)) % patch_size
    pad_y = (patch_size - (mask_tensor.shape[0] % patch_size)) % patch_size

    # Apply padding to the mask
    mask_tensor = F.pad(mask_tensor, (0, pad_x, 0, pad_y), mode='constant', value=0)

    # Split the padded mask into patches of 448x448
    mask_tensor = mask_tensor.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size).reshape(-1, patch_size, patch_size)

    # Downsample each 448x448 mask patch to 28x28 using max-pooling
    mask_tensor = F.max_pool2d(mask_tensor.unsqueeze(1), kernel_size=token_size).squeeze(1)

    # Reshape the mask patches to match the flattened structure (7, 784)
    mask_tensor_flattened = mask_tensor.view(-1)

    # Step 2: Select embeddings corresponding to mask == 1
    # Flatten all_embeddings to (7 * 784, 768) to match with mask
    flattened_embeddings = all_embeddings.view(-1, all_embeddings.shape[-1])
    embeddings_selcted = flattened_embeddings[mask_tensor_flattened == 1].unsqueeze(0)
    return embeddings_selcted, mask_tensor_flattened
