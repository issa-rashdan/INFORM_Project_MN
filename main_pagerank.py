from data.load_data import get_echograms
from data.load_rgb_features import EchogramRGB

from data.split_rgb_features import split_rgb_features, split_label #, recon_rgb_features
from models.get_embeddings import get_embeddings
from torchvision import transforms as pth_transforms
import torch.nn.functional as F

from models.select_model import select_model
import parseargs
import torch
import numpy as np

from method.build_graph import BuildGraphBlockwise, detect_communities_and_create_subgraphs, visualize_subgraph_structure_as_image, upscale_graph_structure 
from method.kernels import KernelsDual
from functools import partial
import matplotlib.pyplot as plt
from visualize import visualize_all_in_subplots
from method.converge_graph import pagerank, masking_out_features

echograms = get_echograms()

patch_size = 448 # 28: one embedding for a pixel (1x1), 224: one embeding for a patch (8x8)
num_tokens = 28
batch_size = 128
token_size = patch_size // num_tokens

args = parseargs.parse_args_vit()
model = select_model(args)

transform = pth_transforms.Compose([
    pth_transforms.Resize((224, 224)),
    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

for echo_idx, e in enumerate(echograms):
    e_rgb = EchogramRGB(e)
    e_rgb.extract_pca_features()
    for medianof in [0.10, 0.40, 0.80]:
        for echosubidx in range(len(e_rgb.pca_features)):
            img = e_rgb.pca_features[echosubidx]['data_rgb_tensor'] # size (436, 3830, 3) or similar. considered as one image chunk
            label = e_rgb.pca_features[echosubidx]['label']
            raw_echo = (e_rgb.pca_features[echosubidx]['raw_data'] - e_rgb.pca_features[echosubidx]['raw_data'].min())/((e_rgb.pca_features[echosubidx]['raw_data'].max() - e_rgb.pca_features[echosubidx]['raw_data'].min()))
            mask = e_rgb.pca_features[echosubidx]['mask']
            patches_rgb, indices, padded_shape = split_rgb_features(img, patch_size=patch_size) # spleted to several chunks after padding, e.g., 9 chunks of 448x448x3 
            label_patches, _, _ = split_label(label, patch_size=patch_size)

            patches = transform(patches_rgb.cpu())
            patches = patches.to('mps') #torch.Size([N, 3, 224, 224])

            embeddings_list = []
            cls_tokens_list = []    
            for start_idx in range(0, len(patches), batch_size):
                end_idx = min(start_idx + batch_size, len(patches))
                minibatch = patches[start_idx:end_idx]  # Extract minibatch

                # Process the minibatch to get embeddings
                cls_token, embeddings = get_embeddings(model, minibatch) # (patch, #token per patch, dim)

                # Collect the embeddings
                embeddings_list.append(embeddings)
                cls_tokens_list.append(cls_token)   

            # Concatenate all the embeddings into a single tensor
            all_embeddings = torch.cat(embeddings_list, dim=0)
            all_cls_tokens = torch.cat(cls_tokens_list, dim=0)

            mask_tensor = torch.tensor(mask, device=args.device)
            flattened_embeddings, mask_tensor_flattened = masking_out_features(mask_tensor, all_embeddings, patch_size, token_size)
            graph_builder_block = BuildGraphBlockwise(flattened_embeddings)

            weighted_graph_block = graph_builder_block.build_undirected_graph_blockwise(partial(KernelsDual().rbf, medianof=medianof))
            full_matrix = graph_builder_block.reconstruct_full_matrix(weighted_graph_block)
            ranking = pagerank(full_matrix, num_iterations=100, d=0.85, tolerance=1e-6)

            output_tensor = torch.zeros_like(mask_tensor_flattened)
            output_tensor[mask_tensor_flattened == 1] = ranking
            reshaped_ranking = output_tensor.view(7, 28, 28).permute(1, 0, 2).reshape(28, 7 * 28)

            plt.figure()
            plt.imshow(reshaped_ranking.cpu().numpy())
            plt.savefig(f'ranking_{echosubidx}_%f.png' %medianof)
            plt.close()

            ###############################################################


