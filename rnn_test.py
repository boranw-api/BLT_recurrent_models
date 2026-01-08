import torch
import argparse
import json
import matplotlib.pyplot as plt
from collections import OrderedDict

import random
import numpy as np

from geometry_path import sample_images, amir_loaders, plot_rep_traj_single_mds, plot_rep_traj_separate_mds, plot_joint_3d_interactive
from analyze_representations import load_model_path

# visualizing the model 
# from tikz_visualizer import visualize_blt, TikzComputationGraphVisualizer

# better traceback hilighting for debugging
from rich.traceback import install
install()

# load custom matplotlib style
plt.style.use('./blt.mplstyle')

def build_args():
    parser = argparse.ArgumentParser(description='Kasper Dataset Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--model-path', type=str,
                    default="./blt_local_cache/imagenet_vggface2_blt_bl_top2linear_run15/blt_full_objects.pt",
                    help='path to the trained model checkpoint')
    parser.add_argument('--layer-categories', type=str,
                        default=json.dumps({
                            "Outputs": ["output_0", "output_1", "output_2", "output_3", "output_4", "output_5"]
                        }),
                        help='JSON string describing the layers to be plotted')
    parser.add_argument('--mds-type', type=str, default='joint_structure', choices=['single', 'multiple', 'joint_structure'],
                        help='Type of MDS plotting: "single" for a shared MDS space, "multiple" for separate MDS per layer, "joint_structure" for all in one 3D plot.')
    parser.add_argument('--plot-dim', type=int, default=3, choices=[2, 3],
                        help='Dimension of the MDS plot (2 or 3)')
    args = parser.parse_args('')

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args.use_cuda = use_cuda
    args.device = device
    args.layer_categories = OrderedDict(json.loads(args.layer_categories))
    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = build_args()
    set_seed(args.seed)

    # train_loader, test_loader = kasper_loaders(args)
    _, test_loader = amir_loaders(args)

    imgs, labels = sample_images(test_loader, n=50)

    if args.mds_type == 'multiple':
        plot_rep_traj_separate_mds(
        args,
        imgs,
        labels
        )
    elif args.mds_type == 'joint_structure':
        plot_joint_3d_interactive(
            args,
            imgs,
            labels
        )
    else:
        plot_rep_traj_single_mds(
            args,
            imgs,
            labels
        )
   


if __name__ == "__main__":
    main()