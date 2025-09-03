import argparse
import torch

def get_default_device():
    """Determines the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def parse_args_vit():
    parser = argparse.ArgumentParser('Visualize patch rankings')
    # DATA AND NETWORK CHOICE
    parser.add_argument('--data', type=str, default='common', choices=['common', 'oil_spill', 'microfossil', 'echosounder'],
                    help='The dataset to use for visualization. Default is common.')
    parser.add_argument('--device', type=torch.device, default=get_default_device(),
                    help='The device to use for computation (cuda, mps, or cpu). Default is the best available device.')
    parser.add_argument('--arch', default='vit_base', type=str, choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    # parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(224, 224), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='./ranked_patches/', help='Path where to save visualizations.')
    # KERNEL AND BANDWIDTH CHOICE
    parser.add_argument('--kernel', type=str, default='hyperbolic_tangent', choices=['hyperbolic_rbf', 'hyperbolic_laplace', 'hyperbolic_tangent', 'rbf', 'laplace', 'linear', 'cosine', 'polynomial'],
                    help='choose kernels for affinity matrix. See ./graphs/kernels.py.')
    parser.add_argument('--bandwidth', type=float, default=0.15, choices=[0.15, 0.5, 1],
                    help='choose bandwidth of kernels wrt median of all distnaces (medianof). See ./graphs/kernels.py.')
    parser.add_argument('--self_loop', type=bool, default=True,
                    help='Remove the diagonal component if False. See ./graphs/build_graph.py.')
    parser.add_argument('--k', type=int, default=50, help='for knn graph (directed). See ./graphs/build_graph.py.')
    parser.add_argument('--hyperbolic_curvature', type=float, default=1, help='curvature of the hyperbolic space')
    # Parse known arguments to ignore unrecognized ones
    args, unknown = parser.parse_known_args()
  
    return args
    # return parser.parse_args()

'''
ROOM FOR OTHER ARGS, e.g., data-wisely (remote sensing, microfossil, echosounder data) or model-wisely (ViT, CNN)

def parse_args_cnn():
    parser = argparse.ArgumentParser('Visualize patch rankings')
    parser.add_argument('--device', type=torch.device, default=get_default_device(),
                    help='The device to use for computation (cuda, mps, or cpu). Default is the best available device.')

'''