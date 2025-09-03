from gpu_pca import IncrementalPCAonGPU
import torch    
import torch.nn.functional as F

def rescale_each_channel_for_visualization(data_rgb_tensor):
    channel_mins = data_rgb_tensor.amin(dim=(0,1))
    channel_maxs = data_rgb_tensor.amax(dim=(0,1))    
    return (data_rgb_tensor - channel_mins) / (channel_maxs - channel_mins)  

def PCAonGPU(data, mask, n_components=3, normalize=True, device=None):
    """
    Perform PCA on the data tensor using GPU
    """
    data = torch.tensor(data).to(device)
    mask = torch.tensor(mask).to(device)
    data_vec = data[mask]
    if normalize:
        data_vec = F.normalize(data_vec, p=2, dim=1)
    batch_size = len(data_vec)
    pca = IncrementalPCAonGPU(n_components=n_components, batch_size=batch_size)
    pca.fit(data_vec)
    data_rgb_vec = pca.transform(data_vec)
    data_rgb = torch.zeros(mask.shape[0], mask.shape[1], n_components).to(device)
    data_rgb[mask] = data_rgb_vec
    return data_rgb, pca

    
