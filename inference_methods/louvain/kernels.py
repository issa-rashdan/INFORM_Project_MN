import torch
import torch.nn.functional as F

class Kernels:
    def __init__(self) -> None:
        pass

    def rbf(self, x, medianof=1, sigma=None): 
        # L2 distance (Euclidean) based RBF kernel
        dists = self.minkowski_distance(x, p=2)
        if sigma is None:
            # Rule of thumb for RBF kernel: 15% of the median value of the distance (Jenssen 2010)
            sigma = torch.median(dists).item() * medianof
        return torch.exp(-(dists**2) / (2 * sigma**2))

    def laplace(self, x, medianof=1, sigma=None): 
        # L1 distance (Manhattan) based Laplace kernel
        dists = self.minkowski_distance(x, p=1)
        if sigma is None:
            # Rule of thumb for Laplace kernel: 15% of the median value of the distance (Jenssen 2010)
            sigma = torch.median(dists).item() * medianof
        return torch.exp(-dists / sigma)

    def minkowski_distance(self, x, p): 
        # Lp distance. L1: Manhattan, L2: Euclidean
        diffs = x.unsqueeze(1) - x.unsqueeze(0)  # Shape: (N, N, d)
        return diffs.abs().pow(p).sum(-1).pow(1.0 / p)

    def linear(self, x):
        # Linear kernel
        out = torch.mm(x, x.t())
        return torch.clamp(out, min=0)

    def cosine(self, x):
        # Cosine similarity kernel
        x_norm = F.normalize(x, p=2, dim=1)
        out = torch.mm(x_norm, x_norm.t())
        return torch.clamp(out, min=0)

    def polynomial(self, x, degree=3, c=1):    
        # Polynomial kernel
        out = (torch.mm(x, x.t()) + c).pow(degree)
        return torch.clamp(out, min=0)


class KernelsDual:
    def __init__(self) -> None:
        pass

    def rbf(self, x1, x2, medianof=1, sigma=None): 
        # L2 distance (Euclidean) based RBF kernel between x1 and x2
        dists = self.minkowski_distance(x1, x2, p=2)
        if sigma is None:
            # Rule of thumb for RBF kernel: 15% of the median value of the distance (Jenssen 2010)
            sigma = torch.median(dists).item() * medianof
        return torch.exp(-dists.pow(2) / (2 * sigma**2))

    def laplace(self, x1, x2, medianof=1, sigma=None): 
        # L1 distance (Manhattan) based Laplace kernel between x1 and x2
        dists = self.minkowski_distance(x1, x2, p=1)
        if sigma is None:
            # Rule of thumb for Laplace kernel: 15% of the median value of the distance (Jenssen 2010)
            sigma = torch.median(dists).item() * medianof
        return torch.exp(-dists / sigma)

    def minkowski_distance(self, x1, x2, p): 
        # Lp distance between x1 and x2. L1: Manhattan, L2: Euclidean
        x1 = x1.unsqueeze(1)  # Shape: (N1, 1, d)
        x2 = x2.unsqueeze(0)  # Shape: (1, N2, d)
        diffs = x1 - x2  # Broadcasting to shape: (N1, N2, d)
        return diffs.abs().pow(p).sum(-1).pow(1.0 / p)  # Final shape: (N1, N2)

    def linear(self, x1, x2):
        # Linear kernel between x1 and x2
        out = torch.mm(x1, x2.t())
        return torch.clamp(out, min=0)

    def cosine(self, x1, x2):
        # Cosine similarity kernel between x1 and x2
        x1_norm = F.normalize(x1, p=2, dim=1)
        x2_norm = F.normalize(x2, p=2, dim=1)
        out = torch.mm(x1_norm, x2_norm.t())
        return torch.clamp(out, min=0)

    def polynomial(self, x1, x2, degree=3, c=1):    
        # Polynomial kernel between x1 and x2
        out = (torch.mm(x1, x2.t()) + c).pow(degree)
        return torch.clamp(out, min=0)

    # def rbf(self, x1, x2, medianof=1, sigma=None): 
    #     # L2 distance (Euclidean) based RBF kernel between x1 and x2
    #     dists = self.minkowski_distance(x1, x2, p=2)
    #     if sigma is None:
    #         # Rule of thumb for RBF kernel: 15% of the median value of the distance (Jenssen 2010)
    #         sigma = torch.median(dists).item() * medianof
    #     return torch.exp(-dists.pow(2) / (2 * sigma**2))

    # def laplace(self, x1, x2, medianof=1, sigma=None): 
    #     # L1 distance (Manhattan) based Laplace kernel between x1 and x2
    #     dists = self.minkowski_distance(x1, x2, p=1)
    #     if sigma is None:
    #         # Rule of thumb for Laplace kernel: 15% of the median value of the distance (Jenssen 2010)
    #         sigma = torch.median(dists).item() * medianof
    #     return torch.exp(-dists / sigma)

    # def minkowski_distance(self, x1, x2, p): 
    #     # Lp distance between x1 and x2. L1: Manhattan, L2: Euclidean
    #     diffs = x1.unsqueeze(1) - x2.unsqueeze(0)  # Shape: (N1, N2, d)
    #     return diffs.abs().pow(p).sum(-1).pow(1.0 / p)

    # def linear(self, x1, x2):
    #     # Linear kernel between x1 and x2
    #     out = torch.mm(x1, x2.t())
    #     return torch.clamp(out, min=0)

    # def cosine(self, x1, x2):
    #     # Cosine similarity kernel between x1 and x2
    #     x1_norm = F.normalize(x1, p=2, dim=1)
    #     x2_norm = F.normalize(x2, p=2, dim=1)
    #     out = torch.mm(x1_norm, x2_norm.t())
    #     return torch.clamp(out, min=0)

# class HyperbolicKernels:
    def __init__(self) -> None:
        pass


    def hyperbolic_rbf(self, x, medianof=1, sigma=None): # L2 distance (euclidean) based
        x = self.hyperbolic_to_euclidean(x)
        dists = self.minkowski_distance(x, p=2)
        if sigma is None:
            # Rule of thumb for rbf kernel: 15 % (0.15) of the median value of in the distance (Jenssen 2010)
            # 15 % tends to be too small in the ViT features.
            sigma = torch.median(dists).item() * medianof
        return torch.exp(-(dists**2)/(2*sigma**2))

    def hyperbolic_laplace(self, x, medianof=1, sigma=None): # L1 distance (manhattan) based
        x = self.hyperbolic_to_euclidean(x)
        dists = self.minkowski_distance(x, p=1)
        if sigma is None:
            # Rule of thumb for rbf kernel: 15 % of the median value of in the distance (Jenssen 2010)
            sigma = torch.median(dists).item() * medianof
        return torch.exp(-dists / sigma)
    
    def hyperbolic_tangent(self, x, medianof=1, sigma=None): # L1 distance (manhattan) based
        x = self.hyperbolic_to_euclidean(x)
        out = torch.mm(x, x.t())
        return torch.clamp(out, min=0)

    def hyperbolic_to_euclidean(self, z):
        epsilon = 1e-6
        max_norm_z = torch.norm(z, p=2, dim=-1).max()
        z_scaled = z/(max_norm_z + max_norm_z*epsilon) 
        norm_z_scaled = torch.norm(z_scaled, p=2, dim=-1)*(1-epsilon)
        curvature_tensor = torch.tensor(self.curvature, device=z.device, dtype=z.dtype)
        scaling_factor = torch.atanh(torch.sqrt(curvature_tensor)*norm_z_scaled)/(torch.sqrt(curvature_tensor)*norm_z_scaled)
        return scaling_factor.unsqueeze(1) * z_scaled