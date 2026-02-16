import torch
import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict

import random
import numpy as np

from geometry_path import (
    sample_images,
    amir_loaders,
    plot_rep_traj_single_mds,
    plot_rep_traj_separate_mds,
    plot_joint_structure,
    extract_recurrent_steps,
    plot_rdm_per_timestep,
)
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
    parser.add_argument('--input-batch-size', type=int, default=50, metavar='N',
                        help='number of images to sample per class. Theoretical max is ~480 (total test set size / 2).')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--model-path', type=str,
                    default="./blt_local_cache/vggface2_blt_bl_top2linear_run15/blt_full_objects.pt",
                    help='path to the trained model checkpoint')
    parser.add_argument('--layer-categories', type=str,
                        default=json.dumps({
                            "Outputs": ["output_0", "output_1", "output_2", "output_3", "output_4", "output_5"]
                        }),
                        help='JSON string describing the layers to be plotted')
    parser.add_argument('--mds-type', type=str, default='joint_structure_3d', choices=['joint_structure_3d', 'multiple', 'joint_structure'],
                        help='Type of MDS plotting: "joint_structure_3d" for a shared 3D MDS space, "multiple" for separate MDS per layer, "joint_structure" for generic joint structure plot.')

    parser.add_argument('--joint-structure-3d-snapshot-views', type=int, default=4,
                        help='Number of 2D snapshot views to save when plot_dim=3.')
    parser.add_argument('--joint-structure-3d-snapshot-elev', type=float, default=20.0,
                        help='Elevation angle for 3D snapshot views.')
    parser.add_argument('--joint-structure-3d-snapshot-azim-start', type=float, default=0.0,
                        help='Starting azimuth angle for 3D snapshot views.')
    parser.add_argument('--joint-structure-3d-snapshot-azim-step', type=float, default=90.0,
                        help='Azimuth step between successive 3D snapshot views.')

    parser.add_argument('--threads', type=int, default=6,
                        help='Number of threads for PyTorch CPU operations (default: None, uses PyTorch default)')
    parser.add_argument('--plot-rdm-timesteps', action='store_true', default=False,
                        help='Plot RDMs per timestep.')
    parser.add_argument('--split-by-label', action='store_true', default=False,
                        help='Split joint structure trajectories by input label (e.g. faces vs objects).')
    parser.add_argument('--rdm-cmap', type=str, default='Blues',
                        choices=['Blues', 'magma', 'viridis'],
                        help='Colormap for RDM plots (Blues, magma, or viridis).')
    parser.add_argument('--rdm-calc-method', type=str, default='euclidean',
                        choices=['euclidean', 'correlation', 'cosine'],
                        help='Method for RDM calculation (euclidean, correlation, cosine).')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # use_cuda = False

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
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)

    # train_loader, test_loader = kasper_loaders(args)
    # currently we are only loading the test set
    _, test_loader = amir_loaders(args)

    if args.dry_run:
        args.input_batch_size = 2
        print("Dry run: reduced batch size to 2")

    imgs, labels = sample_images(test_loader, n=args.input_batch_size)

    if args.plot_rdm_timesteps:
        plot_rdm_per_timestep(args, imgs, labels, split_by_label=args.split_by_label, rdm_cmap=args.rdm_cmap, rdm_calc_method=args.rdm_calc_method)

    if args.mds_type == 'multiple':
        plot_rep_traj_separate_mds(
        args,
        imgs,
        labels,
        rdm_calc_method=args.rdm_calc_method
        )
    elif args.mds_type == 'joint_structure':
        plot_joint_structure(
            args,
            imgs,
            labels,
            split_by_label=args.split_by_label,
            rdm_calc_method=args.rdm_calc_method
        )
    else:
        args.plot_dim = 3
        plot_rep_traj_single_mds(
            args,
            imgs,
            labels,
            rdm_calc_method=args.rdm_calc_method
        )
   


if __name__ == "__main__":
    main()