import torch
import networkx as nx
import community as community_louvain  # This is the Louvain method implementation
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

'''
# = UNDIRECTED: CHOOSE KERNELS AND BANDWIDTH ======================================
# = DIRECTED (knn): CHOOSE KERNELS, BANDWIDTH, AND # OF NEIGHBORS (K) ===================
# OPERATIONAL ORDER:
#   1. CHOOSE DISTANCE/SIMILARITY MEASURE AND BANDWIDTH CRITERIA
#   2. COMPUTE PAIRWISE DISTANCES FOR ALL EMBEDDINGS AND CONVERT TO THE KERNEL SCORE
'''

def upscale_graph_structure(graph_structure, token_size):
    """
    Upscale the graph structure only in height and width using the scale factor.
    
    Args:
        graph_structure: The original graph structure tensor of size [H, W, C].
        scale_factor: The scale factor for height and width (default is (16, 16)).
    
    Returns:
        Upscaled graph structure tensor with scaled H and W dimensions.
    """
    scale_factor=(token_size, token_size)
    # Permute from [H, W, C] to [C, H, W]
    graph_structure_torch = graph_structure.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension [1, C, H, W]
    
    # Perform interpolation (upscale H and W) using nearest neighbor or bilinear interpolation
    upscaled_graph = F.interpolate(graph_structure_torch, scale_factor=scale_factor, mode='nearest')

    # Permute back to [H, W, C] and remove batch dimension
    upscaled_graph = upscaled_graph.squeeze(0).permute(1, 2, 0)  # [H, W, C]
    
    return upscaled_graph

def tensor_to_networkx_graph(tensor):
    """Convert a PyTorch tensor representing an adjacency matrix to a NetworkX graph."""
    G = nx.from_numpy_array(tensor.cpu().numpy())
    return G

def detect_communities_and_create_subgraphs(graphs_tensor):
    """Detect communities using Louvain and create subgraphs."""
    all_partitions = []
    subgraphs = []

    for i in range(graphs_tensor.size(0)):  # Iterate over each graph
        # Convert the tensor to a NetworkX graph
        G = tensor_to_networkx_graph(graphs_tensor[i])

        # Apply Louvain community detection
        partition = community_louvain.best_partition(G)
        all_partitions.append(partition)
        
        # Create subgraphs based on the communities
        community_subgraphs = []
        for community in set(partition.values()):
            nodes_in_community = [node for node, community_id in partition.items() if community_id == community]
            subgraph = G.subgraph(nodes_in_community).copy()
            community_subgraphs.append(subgraph)
        
        subgraphs.append(community_subgraphs)
    
    return all_partitions, subgraphs

def visualize_subgraph_structure_as_image(partition, indices, padded_shape, patch_size, image_size=28):
    """Visualize the community structure on a full image grid."""

    token_size = patch_size // image_size

    # Calculate the output shape, considering the downscaling from the original patch size to the reduced image size
    reduced_height = padded_shape[0] // token_size
    reduced_width = padded_shape[1] // token_size
    
    # Initialize the full image with the calculated shape and 3 channels (RGB)
    full_image = torch.zeros((reduced_height, reduced_width, 3))
    
    # Define a colormap
    cmap = plt.get_cmap("tab20")

    # Calculate the number of nodes per patch (28x28 = 784 nodes)
    tokens_per_patch = image_size ** 2

    # Iterate over each node in the partition
    for node, community in partition.items():
        block_idx, subnode = divmod(node, tokens_per_patch)  # Determine block and subnode within the block
        start_row, start_col = indices[block_idx]  # Get the start indices for this block
        
        # Calculate the start position in the reduced grid
        reduced_start_row = start_row // token_size
        reduced_start_col = start_col // token_size
        
        # Calculate the position of the subnode in the reduced grid
        row_offset, col_offset = divmod(subnode, image_size)
        
        # Determine the final position in the reduced image
        reduced_row = reduced_start_row + row_offset
        reduced_col = reduced_start_col + col_offset
        
        # Assign color based on the community
        color = cmap(community % cmap.N)  # Get color from colormap
        full_image[reduced_row, reduced_col] = torch.tensor(color[:3])
    return full_image


# def visualize_subgraph_structure_as_image(partition, indices, padded_shape, image_size=28):
    """Visualize the community structure on a 28x28 image grid."""

    # compute the output shape, based on the padded_shape, indices, and image_size.
    # the output image will be 3 channels, rgb. 
    # the node will show the location, and the community will show the color.
    # block indeices will show which block (0, 1, 2, 3, 4, 5, 6, 7, 8, 9) shows, and subnode will show the location in the block.

    n_blocks = partition.keys().__len__()//(image_size**2)
    image = np.zeros((image_size, image_size, 3))

    # Define a colormap
    cmap = plt.get_cmap("tab20")
    
    # Fill the image with colors based on the community each node belongs to
    for node, community in partition.items():
        block, subnode = divmod(node, image_size**2)  # Map node index to 2D grid
        row, col = divmod(subnode, image_size)  # Map node index to 2D grid
        color = cmap(community % cmap.N)  # Get color from colormap
        image[row, col] = color[:3]  # Assign RGB color to the pixel
    return image

class BuildGraphBlockwise:
    def __init__(self, embeddings, block_size=2000) -> None:
        pass
        self.shape = embeddings.shape
        self.embeddings = embeddings # (1, #token of the full image, dim)
        self.block_size=block_size

    def build_undirected_graph_blockwise(self, kernelblock):
        symmetric_matrix = []
        for i in range(0, self.shape[1], self.block_size):
            end_i = min(i + self.block_size, self.shape[1])
            block1 = self.embeddings[0, i:end_i, :]

            for j in range(0, self.shape[1], self.block_size):
                end_j = min(j + self.block_size, self.shape[1])
                block2 = self.embeddings[0, j:end_j, :]
                symmetric_matrix.append(kernelblock(block1, block2))
        return symmetric_matrix

    def reconstruct_full_matrix(self, symmetric_matrix):
        # Initialize the full matrix with zeros
        full_matrix = torch.zeros((self.shape[1], self.shape[1]), device=symmetric_matrix[0].device)
        
        block_idx = 0  # Index to track the position in weighted_graph_block
        for i in range(0, self.shape[1], self.block_size):
            end_i = min(i + self.block_size, self.shape[1])
            
            for j in range(0, self.shape[1], self.block_size):
                end_j = min(j + self.block_size, self.shape[1])
                
                # Retrieve the block from the weighted_graph_block list
                block = symmetric_matrix[block_idx]
                block_idx += 1
                
                # Place the block into the correct position in the full matrix
                full_matrix[i:end_i, j:end_j] = block
                
                # # If it's not on the diagonal, ensure symmetry
                # if i != j:
                #     full_matrix[j:end_j, i:end_i] = block.T
        return full_matrix.unsqueeze(0)

class BuildGraph:
    def __init__(self, embeddings) -> None:
        pass
        self.shape = embeddings.shape
        self.embeddings = embeddings # (patch, #token per patch, dim)

    def build_undirected_graph(self, kernel, self_loop=True):
        graph_list = []

        # Iterate over each patch to build the graph individually
        for i in range(self.embeddings.shape[0]):
            # Extract embeddings for the current patch
            patch_embeddings = self.embeddings[i]  # Shape: (#token per patch, dim)

            # Compute the similarity matrix using the kernel function
            symmetric_matrix = kernel(patch_embeddings)

            # Optionally remove self-loops by setting the diagonal to 0
            if not self_loop:
                symmetric_matrix.fill_diagonal_(0)

            # Append the graph for this patch to the list
            graph_list.append(symmetric_matrix)

        # Stack all graphs into a single tensor (optional, depending on how you need the output)
        graphs_tensor = torch.stack(graph_list)  # Shape: (patch, #token per patch, #token per patch)

        return graphs_tensor

    def build_knn_graph(self, kernel, k, self_loop=True):
        # Build the undirected graph using the specified kernel
        graphs_tensor = self.build_undirected_graph(kernel, self_loop)

        knn_weighted_graph_list = []
        knn_binary_graph_list = []

        # Process each patch to apply k-NN filtering
        for i in range(graphs_tensor.shape[0]):
            symmetric_matrix = graphs_tensor[i]  # Get the graph for this patch

            # Create a binary matrix to store the k-NN graph structure
            binary_matrix = torch.zeros_like(symmetric_matrix)

            # Perform top-k selection for each node to find the k-nearest neighbors
            values, indices = torch.topk(symmetric_matrix, k=k, dim=1, largest=True, sorted=True)

            # Set the k-nearest neighbors in the binary matrix
            binary_matrix.scatter_(1, indices, 1)

            # Create the weighted k-NN graph by element-wise multiplying the binary matrix with the original graph
            weighted_matrix = symmetric_matrix * binary_matrix

            # Append the k-NN graph for this patch to the list
            knn_weighted_graph_list.append(weighted_matrix)
            knn_binary_graph_list.append(binary_matrix)

        # Stack all k-NN graphs into a single tensor
        knn_weighted_graphs_tensor = torch.stack(knn_weighted_graph_list)  # Shape: (patch, #token per patch, #token per patch)
        knn_binary_graphs_tensor = torch.stack(knn_binary_graph_list)  # Shape: (patch, #token per patch, #token per patch)

        return knn_weighted_graphs_tensor, knn_binary_graphs_tensor