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
from analyze_representations import load_model_path, perform_dsa_analysis

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
    parser.add_argument('--mds-type', type=str, default='single', choices=['single', 'multiple', 'joint_structure'],
                        help='Type of MDS plotting: "single" for a shared MDS space, "multiple" for separate MDS per layer, "joint_structure" for all in one 3D plot.')
    parser.add_argument('--plot-dim', type=int, default=3, choices=[2, 3],
                        help='Dimension of the MDS plot (2 or 3)')
    parser.add_argument('--single-mds-snapshot-views', type=int, default=4,
                        help='Number of 2D snapshot views to save when plot_dim=3.')
    parser.add_argument('--single-mds-snapshot-elev', type=float, default=20.0,
                        help='Elevation angle for 3D snapshot views.')
    parser.add_argument('--single-mds-snapshot-azim-start', type=float, default=0.0,
                        help='Starting azimuth angle for 3D snapshot views.')
    parser.add_argument('--single-mds-snapshot-azim-step', type=float, default=90.0,
                        help='Azimuth step between successive 3D snapshot views.')
    parser.add_argument('--skip-dsa', action='store_true', default=True,
                        help='Disable DSA analysis when running the script.')
    parser.add_argument('--dsa-pca-components', type=float, default=80,
                        help='PCA components for DSA trajectories (int) or variance explained (0-1).')
    parser.add_argument('--dsa-n-delays', type=int, default=3,
                        help='Number of delays for DSA Hankel embedding.')
    parser.add_argument('--dsa-delay-interval', type=int, default=1,
                        help='Delay interval for DSA Hankel embedding.')
    parser.add_argument('--dsa-rank', type=int, default=None,
                        help='Rank for DSA HAVOK DMD (optional).')
    # rank explained variance is set to 0.95 to remove high-frequency noise in the Hankel matrix
    parser.add_argument('--dsa-rank-explained-variance', type=float, default=0.95,
                        help='Explained variance threshold for DSA rank selection (optional).')
    parser.add_argument('--dsa-rank-thresh', type=float, default=None,
                        help='Singular value threshold for DSA rank selection (optional).')
    parser.add_argument('--dsa-validate-fit', action='store_true', default=False,
                        help='Warn if DSA fit error is high.')
    parser.add_argument('--dsa-fit-error-threshold', type=float, default=0.2,
                        help='Threshold for DSA fit error warnings.')
    parser.add_argument('--dsa-max-steps', type=int, default=None,
                        help='Optional cap on recurrent steps for DSA.')
    parser.add_argument('--dsa-split-by-label', action='store_true', default=False,
                        help='Compute DSA per label within each layer.')
    parser.add_argument('--dsa-save-path', type=str, default=None,
                        help='Optional path for the DSA heatmap.')
    parser.add_argument('--dsa-cmap', type=str, default='viridis',
                        help='Colormap for DSA heatmap.')
    parser.add_argument('--threads', type=int, default=6,
                        help='Number of threads for PyTorch CPU operations (default: None, uses PyTorch default)')
    parser.add_argument('--plot-rdm-timesteps', action='store_true', default=False,
                        help='Plot RDMs per timestep.')
    parser.add_argument('--split-by-label', action='store_true', default=False,
                        help='Split joint structure trajectories by input label (e.g. faces vs objects).')
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


def run_dsa(args, imgs, labels):
    model, _, _ = load_model_path(args.model_path, print_model=False)
    model.to(args.device)
    model.eval()

    model_steps = getattr(model, "times", getattr(model, "num_recurrence", 1))
    if args.dsa_max_steps is not None:
        model_steps = min(model_steps, args.dsa_max_steps)

    layer_list = []
    for _, layers in args.layer_categories.items():
        layer_list.extend(layers)

    if not layer_list:
        return

    imgs_device = imgs.to(args.device)
    trajectories = OrderedDict()

    if args.dsa_split_by_label:
        unique_labels = torch.unique(labels).tolist()
        for layer_name in layer_list:
            activations = extract_recurrent_steps(model, imgs_device, layer_name, steps=model_steps)
            if args.dsa_max_steps is not None:
                activations = activations[:args.dsa_max_steps]
            for lab in unique_labels:
                mask = labels == lab
                if mask.sum().item() == 0:
                    continue
                traj = [step[mask] for step in activations]
                trajectories[f"{layer_name}_label{int(lab)}"] = traj
    else:
        for layer_name in layer_list:
            activations = extract_recurrent_steps(model, imgs_device, layer_name, steps=model_steps)
            if args.dsa_max_steps is not None:
                activations = activations[:args.dsa_max_steps]
            trajectories[layer_name] = activations

    results_dir = Path("results") / "DSA"
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.dsa_save_path is None:
        suffix = "by_label" if args.dsa_split_by_label else "layers"
        save_path = results_dir / f"dsa_similarity_{suffix}_3_delays.png"
    else:
        save_path = Path(args.dsa_save_path)

    title = "DSA Similarity"
    perform_dsa_analysis(
        trajectories,
        pca_components=args.dsa_pca_components,
        device=args.device,
        plot=True,
        save_path=str(save_path),
        title=title,
        cmap=args.dsa_cmap,
        n_delays=args.dsa_n_delays,
        delay_interval=args.dsa_delay_interval,
        rank=args.dsa_rank,
        rank_explained_variance=args.dsa_rank_explained_variance,
        rank_thresh=args.dsa_rank_thresh,
        validate_fit=args.dsa_validate_fit,
        fit_error_threshold=args.dsa_fit_error_threshold,
    )

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
    _, test_loader = amir_loaders(args)

    imgs, labels = sample_images(test_loader, n=1000)

    if not args.skip_dsa:
        run_dsa(args, imgs, labels)

    if args.plot_rdm_timesteps:
        plot_rdm_per_timestep(args, imgs, labels)

    if args.mds_type == 'multiple':
        plot_rep_traj_separate_mds(
        args,
        imgs,
        labels
        )
    elif args.mds_type == 'joint_structure':
        plot_joint_structure(
            args,
            imgs,
            labels,
            split_by_label=args.split_by_label
        )
    else:
        plot_rep_traj_single_mds(
            args,
            imgs,
            labels
        )
   


if __name__ == "__main__":
    main()