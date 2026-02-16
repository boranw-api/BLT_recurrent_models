import io
import pickle
import logging
import contextlib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import OrderedDict
from pathlib import Path

# PyTorch related imports
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, random_split
from torchvision import datasets, transforms
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision.utils import make_grid

from sklearn import manifold
from scipy.spatial.distance import pdist
import plotly.graph_objects as go

# RSA toolbox specific imports
import rsatoolbox
from rsatoolbox.data import Dataset
from rsatoolbox.rdm.calc import calc_rdm

# new imports for our plotting 
from analyze_representations import kasper_dataset, load_model_path
from matplotlib.collections import LineCollection





def amir_loaders(
    args,
    *,
    pickle_path: str = "./blt_local_cache/face_object_dataset.pkl",
    label_key: str = "group",
):
    path = Path(pickle_path)

    with path.open("rb") as handle:
        data = pickle.load(handle)

    images = data.get("images")
    metadata = data.get("metadata")
    base_config = data.get("config") or data.get("config") or {}

    metadata = list(metadata)

    label_values = [entry[label_key] for entry in metadata]
    unique_labels = sorted(set(label_values))
    label_lookup = {value: idx for idx, value in enumerate(unique_labels)}
    labels = torch.tensor([label_lookup[val] for val in label_values], dtype=torch.long)

    if isinstance(images, list):
        images = np.stack(images)
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)

    imgs = images.float()
    enforce_channels_first = True

    if enforce_channels_first and imgs.ndim == 4 and imgs.shape[-1] in (1, 3, 4) and imgs.shape[1] not in (1, 3, 4):
        imgs = imgs.permute(0, 3, 1, 2)
    elif imgs.ndim == 3:
        imgs = imgs.unsqueeze(1)

    # resize to (224, 224) 
    if imgs.shape[-2:] != (224, 224):
        imgs = F.interpolate(imgs, size=(224, 224), mode='bilinear', align_corners=False)  

    config = dict(base_config)
    config.update({
        "label_key": label_key,
        "label_lookup": label_lookup,
    })
    imgs = imgs.contiguous()

    dataset = TensorDataset(imgs, labels)
    dataset_fraction = float(getattr(args, "train_split", 0.8))
    dataset_fraction = min(max(dataset_fraction, 0.0), 1.0)

    total_items = len(dataset)

    train_size = max(1, int(round(total_items * dataset_fraction)))
    if train_size >= total_items:
        train_size = total_items - 1
    test_size = total_items - train_size

    # Set batch size to the full size of each split
    train_kwargs = {"batch_size": train_size, "shuffle": True}
    test_kwargs = {"batch_size": test_size, "shuffle": False}
    if args.use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    seed = getattr(args, "seed", None)
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    setattr(args, "amir_config", config)

    return train_loader, test_loader

# loading kasper dataset from Vinken SA paper 
def kasper_loaders(args):

    imgs, labels, neuro_data = kasper_dataset()

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    if not torch.is_tensor(imgs):
        imgs = torch.as_tensor(imgs)
    imgs = imgs.float()

    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    elif not torch.is_tensor(labels):
        labels = torch.as_tensor(labels)
    labels = labels.long()

    if isinstance(neuro_data, np.ndarray):
        neuro_data = torch.from_numpy(neuro_data)
    elif not torch.is_tensor(neuro_data):
        neuro_data = torch.as_tensor(neuro_data)
    neuro_data = neuro_data.float()

    mask = labels != 1
    imgs = imgs[mask]
    labels = labels[mask]
    neuro_data = neuro_data[mask]

    num_faces = int((labels == 0).sum().item())
    setattr(args, "num_faces", num_faces)

    dataset = TensorDataset(imgs, labels)

    dataset_fraction = float(getattr(args, "train_split", 0.8))
    dataset_fraction = min(max(dataset_fraction, 0.0), 1.0)

    total_items = len(dataset)
    train_size = int(round(total_items * dataset_fraction))
    train_size = min(max(train_size, 1), total_items - 1)
    test_size = total_items - train_size

    seed = getattr(args, "seed", None)
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    return train_loader, test_loader

def calc_rdms(args, model_features, method='correlation'):
    """
    Calculates representational dissimilarity matrices (RDMs) for model features.

    Inputs:
    - model_features (dict): A dictionary where keys are layer names and values are features of the layers.
    - method (str): The method to calculate RDMs, e.g., 'correlation'. Default is 'correlation'.

    Outputs:
    - rdms (pyrsa.rdm.RDMs): RDMs object containing dissimilarity matrices.
    - rdms_dict (dict): A dictionary with layer names as keys and their corresponding RDMs as values.
    """
    ds_list = []
    kept_layers = []
    for l in range(len(model_features)):
        layer = list(model_features.keys())[l]
        feats = model_features[layer]

        if isinstance(feats, (int, float)) or not torch.is_tensor(feats):
            logging.warning("Skipping non-tensor feature (%s)", layer)
            continue

        if type(feats) is list:
            feats = feats[-1]

        if args.use_cuda:
            feats = feats.cpu()

        if len(feats.shape) > 2:
            feats = feats.flatten(1)

        feats = feats.detach().numpy()
        ds = Dataset(feats, descriptors=dict(layer=layer))
        ds_list.append(ds)
        kept_layers.append(layer)

    rdms = calc_rdm(ds_list, method=method)
    rdms_dict = {layer: rdms.get_matrices()[i] for i, layer in enumerate(kept_layers)}

    # before skipping non tensor features this works
    # rdms_dict = {list(model_features.keys())[i]: rdms.get_matrices()[i] for i in range(len(model_features))}

    return rdms, rdms_dict



# created for potting recurrent steps of a given layer
def extract_recurrent_steps(model, imgs, target_layer, steps=15):
    module = dict(model.named_modules())[target_layer]
    activations = []

    def hook(_module, _input, output):
        activations.append(output.detach().cpu())

    handle = module.register_forward_hook(hook)
    original_num_recurrence = getattr(model, "num_recurrence", steps)
    original_times = getattr(model, "times", original_num_recurrence)
    setattr(model, "num_recurrence", steps)
    if hasattr(model, "times"):
        setattr(model, "times", steps)

    with torch.no_grad():
        model(imgs)

    setattr(model, "num_recurrence", original_num_recurrence)
    if hasattr(model, "times"):
        setattr(model, "times", original_times)
    handle.remove()
    return activations

def sample_images(data_loader, n=5, plot=True):
    """
    Samples a specified number of images from a data loader.

    Inputs:
    - data_loader (torch.utils.data.DataLoader): Data loader containing images and labels.
    - n (int): Number of images to sample per class.
    - plot (bool): Whether to plot the sampled images using matplotlib.

    Outputs:
    - imgs (torch.Tensor): Sampled images.
    - labels (torch.Tensor): Corresponding labels for the sampled images.
    """
    batch = next(iter(data_loader))
    imgs, targets = batch[:2]  # Unpack only the first two elements (images and labels)

    imgs_o = []
    labels = []
    unique_targets = torch.unique(targets)
    for value in unique_targets:
        class_imgs = imgs[targets == value][:n]
        if class_imgs.size(0) == 0:
            continue
        imgs_o.append(class_imgs)
        labels.append(torch.full((class_imgs.size(0),), value, dtype=torch.long, device=targets.device))

    if not imgs_o:
        raise ValueError("No samples were collected from the provided data loader.")

    imgs = torch.cat(imgs_o, dim=0)
    labels = torch.cat(labels, dim=0)

    if plot:
        plt.imshow(torch.moveaxis(make_grid(imgs, nrow=5, padding=0, normalize=False, pad_value=0), 0,-1))
        plt.axis('off')

    return imgs, labels







# beginning here: new functions for Fig.4 
def plot_shepard_diagram(data, dissimilarity_matrix=None, save_path=None, layout=None, title='Shepard Diagram'):
    """
    Plots Shepard diagram(s) to examine the goodness of fit for MDS transformer(s).

    Inputs:
    - data: Either an MDS transformer (single plot) or a list of tuples 
            (mds_transformer, dissimilarity_matrix, title) for multiple plots.
    - dissimilarity_matrix: The original dissimilarity matrix (only for single plot).
    - save_path: Optional path to save the figure.
    - layout: Tuple (rows, cols) for grid layout (only for multiple plots).
    - title: Title for the plot (only for single plot).

    Outputs:
    - fig: The matplotlib figure object.
    """

    if isinstance(data, list):
        # Multiple plots case
        plot_list = data
        if layout is None:
            n_plots = len(plot_list)
            cols = int(np.ceil(np.sqrt(n_plots)))
            rows = int(np.ceil(n_plots / cols))
            layout = (rows, cols)
    else:
        # Single plot case
        plot_list = [(data, dissimilarity_matrix, title)]
        layout = (1, 1)

    rows, cols = layout
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(plot_list):
            mds_transformer, matrix, plot_title = plot_list[i]

            embedded = mds_transformer.embedding_
            fitted_distances = pdist(embedded, metric='euclidean')
            triu_indices = np.triu_indices_from(matrix, k=1)
            original_dissimilarities = matrix[triu_indices]

            ax.scatter(original_dissimilarities, fitted_distances, alpha=0.5, s=10, c='#41B6E6')
            ax.plot([original_dissimilarities.min(), original_dissimilarities.max()],
                    [original_dissimilarities.min(), original_dissimilarities.max()], color='#EF002B', linewidth=2, linestyle='--')
            ax.set_xlabel('Original Dissimilarities')
            ax.set_ylabel('Fitted Distances')
            ax.set_title(plot_title)

            for spine in ax.spines.values():
                spine.set_color('black')
            ax.set_facecolor('white')
            ax.grid(False)

            if hasattr(mds_transformer, 'stress_'):
                ax.text(0.05, 0.95, f'Stress: {mds_transformer.stress_:.4f}',
                        transform=ax.transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            ax.axis('off')

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    return fig

def plot_rep_traj_single_mds(
    args,
    imgs,
    labels,
    cache_dir="blt_local_cache",
    steps=None,
    target_layers=None,
    max_steps=None,
    rdm_calc_method="euclidean",
    rdm_comp_method="cosine",
    figsize_per_panel=4.0,
):
    """
    Generates a combined single-figure trajectory plot where each layer is
    shown in a distinct color and time evolution is indicated by hollow-to-solid dots.
    When plot_dim == 3, also saves multiple 2D snapshot views for appendix-style panels.
    """
    model_path = Path(getattr(args, 'model_path'))
    if not model_path.exists():
        print(f"Model path does not exist: {model_path}")
        return []
        
    model_files = [model_path]
    
    if getattr(args, "dry_run", False):
        print("Dry run: processing specified model")

    output_paths = []
    if not model_files:
        return output_paths

    all_flat_features = OrderedDict()
    model_layer_maps = OrderedDict()

    for model_path in model_files:
        model_name = model_path.parent.name
        model_name = "_".join(model_name.split("_")[:2])
        print(f"Processing (1x6): {model_name}")
        model, _, _ = load_model_path(str(model_path), print_model=False)
        model.to(args.device)
        model.eval()

        model_steps = steps
        if getattr(args, "dry_run", False):
            model_steps = 2
            print("Dry run: reduced steps to 2")
        elif model_steps is None:
            model_steps = getattr(model, "times", getattr(model, "num_recurrence", 1))

        layer_list = target_layers
        if layer_list is None:
            layer_list = ["output_0", "output_1", "output_2", "output_3", "output_4", "output_5"]

        imgs_device = imgs.to(args.device)
        layer_step_features = OrderedDict()
        for layer_name in layer_list:
            activations = extract_recurrent_steps(model, imgs_device, layer_name, steps=model_steps)
            if max_steps is not None:
                activations = activations[:max_steps]
            step_features = OrderedDict(
                (f"{layer_name}_t{idx}", feat) for idx, feat in enumerate(activations)
            )
            layer_step_features[layer_name] = step_features

        flat_features = OrderedDict()
        prefixed_layer_keys = OrderedDict()
        for layer_name, step_features in layer_step_features.items():
            keys = list(step_features.keys())
            for key in keys:
                flat_features[key] = step_features[key]
            prefixed_keys = [f"{model_name}_{key}" for key in keys]
            prefixed_layer_keys[layer_name] = prefixed_keys
            for pref_key, orig_key in zip(prefixed_keys, keys):
                all_flat_features[pref_key] = step_features[orig_key]

        model_layer_maps[model_name] = {
            "layer_order": layer_list,
            "prefixed_layer_keys": prefixed_layer_keys,
        }

    rdms_flat, _ = calc_rdms(args, all_flat_features, method=rdm_calc_method)
    rdms = rdms_flat
    ax_ticks = list(all_flat_features.keys())

    include_labels = labels is not None
    include_labels = False
    if include_labels:
        label_rdm, _ = calc_rdms(
            args,
            {"labels": F.one_hot(labels).float().to(args.device)},
            method=rdm_calc_method,
        )
        rdms = rsatoolbox.rdm.concat((rdms, label_rdm))
        ax_ticks.append("labels")

    rdms_comp = rsatoolbox.rdm.compare(rdms, rdms, method=rdm_comp_method)
    if rdm_comp_method == "cosine":
        rdms_comp = np.clip(rdms_comp, -1, 1)
        rdms_comp = np.arccos(rdms_comp)
    rdms_comp = np.nan_to_num(rdms_comp, nan=0.0)
    rdms_comp = (rdms_comp + rdms_comp.T) / 2.0

    plot_dim = getattr(args, "plot_dim", 3)
    # Comment: A single MDS transformer is fitted across all layers so every subplot
    # reuses the same embedding frame, allowing direct visual comparison of trajectories.
    transformer = manifold.MDS(
        n_components=plot_dim,
        # to metric or to non-metric?
        metric=True,
        max_iter=3000,
        n_init=30,
        normalized_stress=True,
        dissimilarity="precomputed",
    )
    dims = transformer.fit_transform(rdms_comp)

    results_dir = Path("results") / ("3D" if plot_dim == 3 else "2D")
    results_dir.mkdir(parents=True, exist_ok=True)

    shepard_path = results_dir / "shepard_single_mds.png"
    plot_shepard_diagram(transformer, rdms_comp, save_path=shepard_path)
    output_paths.append(str(shepard_path))

    coord_map = {tick: dims[idx] for idx, tick in enumerate(ax_ticks)}

    coords_array = np.vstack(list(coord_map.values()))
    amin, amax = coords_array.min(), coords_array.max()
    center = (amin + amax) / 2.0
    half_span = max((amax - amin) / 2.0, 1e-6)
    axis_min = center - half_span * 1.4
    axis_max = center + half_span * 1.4

    snapshot_views = int(getattr(args, "shared_space_snapshot_views", 4))
    snapshot_elev = float(getattr(args, "shared_space_snapshot_elev", 20.0))
    snapshot_azim_start = float(getattr(args, "shared_space_snapshot_azim_start", 0.0))
    snapshot_azim_step = float(getattr(args, "shared_space_snapshot_azim_step", 90.0))

    for model_path in model_files:
        model_name = model_path.parent.name
        model_name = "_".join(model_name.split("_")[:2])
        # Filter removed to allow explicit model path selection
        # if "vggface" not in model_name or "imagenet" in model_name:
        #     continue  # Skip models that do not contain "vggface" or contain "imagenet" in their name
        layer_order = model_layer_maps[model_name]["layer_order"]
        prefixed_layer_keys = model_layer_maps[model_name]["prefixed_layer_keys"]
        layer_cmap = plt.get_cmap("tab10")
        layer_colors = {
            layer_name: layer_cmap(idx % 10)
            for idx, layer_name in enumerate(layer_order)
        }

        fig = plt.figure(figsize=(8.5, 7.5))
        if plot_dim == 3:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)

        for layer_name in layer_order:
            prefixed_keys = prefixed_layer_keys.get(layer_name, [])
            coords = [coord_map[k] for k in prefixed_keys if k in coord_map]
            if not coords:
                continue
            coords = np.vstack(coords)
            num_steps = len(coords)
            color = layer_colors[layer_name]

            if plot_dim == 3:
                ax.plot(
                    coords[:, 0],
                    coords[:, 1],
                    coords[:, 2],
                    color=color,
                    linewidth=1.5,
                    alpha=0.6,
                )
                for step_idx in range(num_steps):
                    alpha = step_idx / (num_steps - 1) if num_steps > 1 else 1.0
                    face_color = (*color[:3], alpha)
                    ax.scatter(
                        coords[step_idx, 0],
                        coords[step_idx, 1],
                        coords[step_idx, 2],
                        s=36,
                        facecolors=[face_color],
                        edgecolors=[color],
                        linewidths=1.2,
                    )
            else:
                ax.plot(
                    coords[:, 0],
                    coords[:, 1],
                    color=color,
                    linewidth=1.5,
                    alpha=0.6,
                )
                for step_idx in range(num_steps):
                    alpha = step_idx / (num_steps - 1) if num_steps > 1 else 1.0
                    face_color = (*color[:3], alpha)
                    ax.scatter(
                        coords[step_idx, 0],
                        coords[step_idx, 1],
                        s=36,
                        facecolors=[face_color],
                        edgecolors=[color],
                        linewidths=1.2,
                    )

        ax.set_xlim([axis_min, axis_max])
        ax.set_ylim([axis_min, axis_max])
        if plot_dim == 3:
            ax.set_zlim([axis_min, axis_max])
            ax.set_zlabel("dim 3")

        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")

        for spine in ax.spines.values():
            spine.set_color("black")
        ax.set_facecolor("white")
        ax.grid(False)

        ax.set_title(f"Representational Trajectories - {model_name}")
        if layer_order:
            ax.legend(layer_order, loc="upper right", fontsize=8, frameon=False)

        fig.tight_layout()
        save_path = results_dir / f"rep_geo_single_mds_{model_name}_combined.png"
        fig.savefig(save_path, bbox_inches="tight")
        output_paths.append(str(save_path))

        if plot_dim == 3 and snapshot_views > 0:
            for view_idx in range(snapshot_views):
                azim = snapshot_azim_start + snapshot_azim_step * view_idx
                ax.view_init(elev=snapshot_elev, azim=azim)
                snapshot_path = results_dir / (
                    f"rep_geo_single_mds_{model_name}_view{view_idx + 1}.png"
                )
                fig.savefig(snapshot_path, bbox_inches="tight")
                output_paths.append(str(snapshot_path))

        plt.close(fig)

    return output_paths, transformer, rdms_comp

def plot_rep_traj_separate_mds(
    args,
    imgs,
    labels,
    cache_dir="blt_local_cache",
    steps=None,
    target_layers=None,
    max_steps=None,
    rdm_calc_method="euclidean",
    rdm_comp_method="cosine",
    figsize_per_panel=4.0,
):
    """
    For vggface2 BLT model in the cache, fit a separate MDS per layer
    (rather than one shared embedding across layers) and produce two figures:
    1) a 1xN grid of trajectories (one subplot per layer) styled like
       `plot_cache_models_joint_embedding_merged_layers`, and
    2) a matching 1xN grid of Shepard diagrams summarizing each layer's fit.
    """

    cache_root = Path(cache_dir)
    model_files = sorted(cache_root.glob("*/blt_full_objects.pt"))
    if getattr(args, "dry_run", False):
        valid_models = []
        for p in model_files:
            mn = "_".join(p.parent.name.split("_")[:2])
            if "vggface" in mn and "imagenet" not in mn:
                valid_models.append(p)
                break
        model_files = valid_models
        print("Dry run: processing only first valid model")

    trajectory_paths = []
    shepard_paths = []

    plot_dim = getattr(args, "plot_dim", 3)
    results_dir = Path("results") / ("3D" if plot_dim == 3 else "2D")
    results_dir.mkdir(parents=True, exist_ok=True)

    if not model_files:
        return trajectory_paths, shepard_paths

    for model_path in model_files:
        model_name = model_path.parent.name
        model_name = "_".join(model_name.split("_")[:2])
        # Restrict to vggface variants as in the shared-MDS function
        if "vggface" not in model_name or "imagenet" in model_name:
            continue

        print(f"Processing (separate MDS): {model_name}")
        model, _, _ = load_model_path(str(model_path), print_model=False)
        model.to(args.device)
        model.eval()

        model_steps = steps
        if getattr(args, "dry_run", False):
            model_steps = 2
            print("Dry run: reduced steps to 2")
        elif model_steps is None:
            model_steps = getattr(model, "times", getattr(model, "num_recurrence", 1))

        print(model_steps)
        layer_list = target_layers or [
            "output_0",
            "output_1",
            "output_2",
            "output_3",
            "output_4",
            "output_5",
        ]

        imgs_device = imgs.to(args.device)
        layer_step_features = OrderedDict()
        for layer_name in layer_list:
            activations = extract_recurrent_steps(
                model,
                imgs_device,
                layer_name,
                steps=model_steps,
            )
            if max_steps is not None:
                activations = activations[:max_steps]
            if not activations:
                continue
            step_features = OrderedDict(
                (f"{layer_name}_t{idx}", feat) for idx, feat in enumerate(activations)
            )
            layer_step_features[layer_name] = step_features

        if not layer_step_features:
            continue

        layer_results = OrderedDict()
        for layer_name, step_features in layer_step_features.items():
            if not step_features:
                continue

            flat_features = OrderedDict(step_features)
            rdms_flat, _ = calc_rdms(args, flat_features, method=rdm_calc_method)
            rdms = rdms_flat
            ax_ticks = list(flat_features.keys())

            rdms_comp = rsatoolbox.rdm.compare(rdms, rdms, method=rdm_comp_method)
            if rdm_comp_method == "cosine":
                rdms_comp = np.clip(rdms_comp, -1, 1)
                rdms_comp = np.arccos(rdms_comp)
            rdms_comp = np.nan_to_num(rdms_comp, nan=0.0)
            rdms_comp = (rdms_comp + rdms_comp.T) / 2.0
            print(f"RDM dimension for {layer_name}: {rdms_comp.shape}")

            transformer = manifold.MDS(
                n_components=plot_dim,
                metric=True,
                max_iter=3000,
                n_init=30,
                normalized_stress=True,
                dissimilarity="precomputed",
            )
            dims = transformer.fit_transform(rdms_comp)
            coord_map = {tick: dims[idx] for idx, tick in enumerate(ax_ticks)}
            coords = np.vstack(list(coord_map.values())) if coord_map else np.zeros((0, plot_dim))

            layer_results[layer_name] = {
                "coords": coords,
                "transformer": transformer,
                "rdm": rdms_comp,
            }

        if not layer_results:
            continue

        coords_arrays = [res["coords"] for res in layer_results.values() if res["coords"].size]
        if not coords_arrays:
            continue

        stacked = np.vstack(coords_arrays)
        amin, amax = stacked.min(), stacked.max()
        center = (amin + amax) / 2.0
        half_span = max((amax - amin) / 2.0, 1e-6)
        axis_min = center - half_span * 1.4
        axis_max = center + half_span * 1.4

        num_layers = len(layer_results)
        rows = 2
        cols = 3
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(figsize_per_panel * cols, figsize_per_panel * rows * 1.2),
            squeeze=False,
            subplot_kw={'projection': '3d'} if plot_dim == 3 else None,
        )

        for idx, (layer_name, result) in enumerate(layer_results.items()):
            ax = axes[idx // cols, idx % cols]
            coords = result["coords"]
            if coords.size == 0:
                ax.axis("off")
                ax.set_title(f"{layer_name}\n(no activations)")
                continue

            num_steps = len(coords)
            cmap = plt.get_cmap("viridis_r")
            colors = cmap(np.linspace(0, 1, num_steps))

            if plot_dim == 3:
                for step_idx in range(num_steps - 1):
                    ax.plot(
                        coords[step_idx : step_idx + 2, 0],
                        coords[step_idx : step_idx + 2, 1],
                        coords[step_idx : step_idx + 2, 2],
                        color=colors[step_idx],
                        linewidth=2,
                    )
                ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colors, s=30, edgecolors="none")
                ax.plot(coords[0, 0], coords[0, 1], coords[0, 2], color="k", marker="s", markersize=5)
                ax.set_zlim([axis_min, axis_max])
                if idx == 0:
                    ax.set_zlabel("dim 3")
            else:
                for step_idx in range(num_steps - 1):
                    ax.plot(
                        coords[step_idx : step_idx + 2, 0],
                        coords[step_idx : step_idx + 2, 1],
                        color=colors[step_idx],
                        linewidth=2,
                    )
                ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=30, edgecolors="none")
                ax.plot(coords[0, 0], coords[0, 1], color="k", marker="s", markersize=5)

            ax.set_xlim([axis_min, axis_max])
            ax.set_ylim([axis_min, axis_max])
            ax.set_title(layer_name)
            if idx == 0:
                ax.set_ylabel("dim 2")
            ax.set_xlabel("dim 1")

            for spine in ax.spines.values():
                spine.set_color("black")
            ax.set_facecolor("white")
            ax.grid(False)

            distances = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
            total_path_length = np.sum(distances)
            avg_step_length = total_path_length / (num_steps - 1) if num_steps > 1 else 0.0
            text_str = f"Total: {total_path_length:.2f}\nAvg: {avg_step_length:.2f}"
            text_kwargs = dict(transform=ax.transAxes, fontsize=8, verticalalignment="top",
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            if plot_dim == 3:
                ax.text2D(0.05, 0.95, text_str, **text_kwargs)
            else:
                ax.text(0.05, 0.95, text_str, **text_kwargs)

        colorbar_width = 2.5 / num_layers
        colorbar_left = (1.0 - colorbar_width) / 2.0
        cbar_ax = fig.add_axes([colorbar_left, 0.08, colorbar_width, 0.02])
        sm = plt.cm.ScalarMappable(cmap="viridis_r")
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("Timestep")
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['early', 'mid', 'late'])

        fig.suptitle(f"Separate MDS Trajectories - {model_name}", y=1.02)
        fig.tight_layout(rect=[0, 0.15, 1, 1])
        traj_path = results_dir / f"rep_geo_separate_mds_{model_name}_2x3.png"
        fig.savefig(traj_path, bbox_inches="tight")
        plt.close(fig)
        trajectory_paths.append(str(traj_path))

        # Shepard diagrams for each layer using helper
        mds_results = [
            (result["transformer"], result["rdm"], f"{layer_name}")
            for layer_name, result in layer_results.items()
        ]
        shepard_path = results_dir / f"shepard_separate_mds_{model_name}_2x3.png"
        plot_shepard_diagram(
            mds_results,
            save_path=shepard_path,
            layout=(2, 3),
        )
        shepard_paths.append(str(shepard_path))

    return trajectory_paths, shepard_paths

def plot_joint_structure(
    args,
    imgs,
    labels,
    cache_dir="blt_local_cache",
    steps=None,
    target_layers=None,
    max_steps=None,
    rdm_calc_method="euclidean",
    rdm_comp_method="cosine",
    split_by_label=False,
):
    """
    Plots all layers and time steps in a single 2D MDS space.
    Each layer is shown in a distinct color, with a gradient indicating time flow.
    Trajectories are drawn as lines with gradient color, plus dots at each step.
    If split_by_label is True, separates trajectories by input class (e.g. Face vs Object).
    """
    # cache_root = Path(cache_dir)
    # model_files = sorted(cache_root.glob("*/blt_full_objects.pt"))
    
    # Use the specific model path provided in args
    model_path = Path(getattr(args, 'model_path'))
    if not model_path.exists():
        print(f"Model path does not exist: {model_path}")
        return []
        
    model_files = [model_path]
    
    if getattr(args, "dry_run", False):
        print("Dry run: processing specified model")

    output_paths = []

    if not model_files:
        return output_paths

    for model_path in model_files:
        model_name = model_path.parent.name
        model_name = "_".join(model_name.split("_")[:2])
        # Filter removed to allow explicit model path selection
        # if "vggface" not in model_name or "imagenet" in model_name:
        #     continue

        print(f"Processing (Joint MDS): {model_name}")
        model, _, _ = load_model_path(str(model_path), print_model=False)
        model.to(args.device)
        model.eval()

        model_steps = steps
        if getattr(args, "dry_run", False):
            model_steps = 2
            print("Dry run: reduced steps to 2")
        elif model_steps is None:
            model_steps = getattr(model, "times", getattr(model, "num_recurrence", 1))

        layer_list = target_layers
        if layer_list is None:
            layer_list = ["output_0", "output_1", "output_2", "output_3", "output_4", "output_5"]

        imgs_device = imgs.to(args.device)
        
        # Prepare features logic
        if split_by_label and labels is not None:
            # SEPARATE MDS PER LABEL logic
            unique_labels = sorted(torch.unique(labels).tolist())
            
            # Extract once for all images
            all_layer_activations = {}
            for layer_name in layer_list:
                activations = extract_recurrent_steps(model, imgs_device, layer_name, steps=model_steps)
                if max_steps is not None:
                    activations = activations[:max_steps]
                all_layer_activations[layer_name] = activations
            
            # Loop over labels and create a plot for each
            for lbl in unique_labels:
                lbl_mask = (labels == lbl).cpu().numpy()
                lbl_count = np.sum(lbl_mask)
                if lbl_count < 2:
                    print(f"Skipping label {lbl}: insufficient samples ({lbl_count})")
                    continue
                
                print(f"Processing Separate MDS for Label {lbl}...")
                
                # Build flat features for this label only
                flat_features = OrderedDict()
                for layer_name in layer_list:
                    act_list = all_layer_activations[layer_name]
                    for t_idx, feat in enumerate(act_list):
                        # feat is (Batch, D), slice by mask
                        sub_feat = feat[lbl_mask]
                        key = f"{layer_name}_t{t_idx}"
                        flat_features[key] = sub_feat

                # Calc RDMs
                rdms_flat, _ = calc_rdms(args, flat_features, method=rdm_calc_method)
                
                # Compare
                rdms_comp = rsatoolbox.rdm.compare(rdms_flat, rdms_flat, method=rdm_comp_method)
                if rdm_comp_method == "cosine":
                    rdms_comp = np.arccos(np.clip(rdms_comp, -1, 1))
                rdms_comp = np.nan_to_num(rdms_comp, nan=0.0)
                rdms_comp = (rdms_comp + rdms_comp.T) / 2.0
                
                # MDS
                transformer = manifold.MDS(
                    n_components=2,
                    metric=True,
                    max_iter=3000,
                    n_init=30,
                    normalized_stress=True,
                    dissimilarity="precomputed",
                )
                dims = transformer.fit_transform(rdms_comp)
                
                # Rotate
                dims_rot = np.zeros_like(dims)
                dims_rot[:, 0] = dims[:, 1]
                dims_rot[:, 1] = -dims[:, 0]
                dims = dims_rot
                
                # Map
                keys = list(flat_features.keys())
                coord_map = {k: dims[i] for i, k in enumerate(keys)}
                
                # Plotting Config
                num_layers = len(layer_list)
                cmap = plt.get_cmap("tab10") 
                layer_base_colors = [cmap(i) for i in range(num_layers)]

                results_dir = Path("results") / "Joint_Structure"
                results_dir.mkdir(parents=True, exist_ok=True)

                fig, ax = plt.subplots(figsize=(10, 8))
                
                all_coords = dims
                amin, amax = all_coords.min(), all_coords.max()
                center = (amin + amax) / 2.0
                half_span = max((amax - amin) / 2.0, 1e-6)
                axis_min = center - half_span * 1.1
                axis_max = center + half_span * 1.1
                
                legend_handles = []
                legend_labels = []

                for l_idx, layer_name in enumerate(layer_list):
                    base_color = layer_base_colors[l_idx]
                    
                    # Gather coords
                    keys_for_layer = [k for k in keys if k.startswith(f"{layer_name}_t")]
                    keys_for_layer.sort(key=lambda x: int(x.split('_t')[1]))
                    
                    layer_coords = []
                    for key in keys_for_layer:
                        if key in coord_map:
                            layer_coords.append(coord_map[key])
                    
                    layer_coords = np.array(layer_coords)
                    n_steps = len(layer_coords)
                    
                    if n_steps > 0:
                        layer_colors_grad = []
                        for t in range(n_steps):
                            alpha = 0.2 + 0.8 * (t / (n_steps - 1)) if n_steps > 1 else 1.0
                            c = list(base_color)
                            c[3] = alpha 
                            layer_colors_grad.append(tuple(c))
                        
                        if n_steps > 1:
                            points = layer_coords.reshape(-1, 1, 2)
                            segments = np.concatenate([points[:-1], points[1:]], axis=1)
                            seg_colors = layer_colors_grad[:-1]
                            lc = LineCollection(segments, colors=seg_colors, linewidths=2, alpha=1.0)
                            ax.add_collection(lc)

                        ax.scatter(layer_coords[:, 0], layer_coords[:, 1], c=layer_colors_grad, s=40, edgecolors='none', zorder=10)
                        
                        # Legend Entry
                        from matplotlib.lines import Line2D
                        handle = Line2D([0], [0], marker='o', color='w', label=layer_name,
                                        markerfacecolor=base_color, markersize=15, linestyle='None')
                        legend_handles.append(handle)
                        legend_labels.append(layer_name)

                ax.set_xlim([axis_min, axis_max])
                ax.set_ylim([axis_min, axis_max])
                ax.set_aspect('equal')
                ax.set_xlabel("MDS Dim 1")
                ax.set_ylabel("MDS Dim 2")
                label_str = "face only" if lbl == 0 else "object only" if lbl == 1 else f"Label {lbl}"
                ax.set_title(f"Joint MDS: {model_name} ({label_str})")
                
                legend = ax.legend(handles=legend_handles, labels=legend_labels, 
                                   loc='center left', bbox_to_anchor=(1, 0.5), 
                                   frameon=True, fontsize=14)
                legend.get_frame().set_facecolor('#e0e0e0')
                legend.get_frame().set_edgecolor('none')
                
                for spine in ax.spines.values():
                    spine.set_color("black")
                ax.set_facecolor("white")
                ax.grid(True, linestyle='--', alpha=0.5)

                save_path = results_dir / f"joint_structure_{model_name}_label_{lbl}.png"
                fig.savefig(save_path, bbox_inches="tight", dpi=300)
                plt.close(fig)
                output_paths.append(str(save_path))
            
            return output_paths

        else: 
            # Original Logic for Single Joint Plot (or Combined if non-split)
            flat_features = OrderedDict()
            for layer_name in layer_list:
                activations = extract_recurrent_steps(model, imgs_device, layer_name, steps=model_steps)
                if max_steps is not None:
                    activations = activations[:max_steps]
                
                for t_idx, feat in enumerate(activations):
                    key = f"{layer_name}_t{t_idx}"
                    flat_features[key] = feat

            if not flat_features:
                continue

            # Calc RDMs
            rdms_flat, _ = calc_rdms(args, flat_features, method=rdm_calc_method)
            
            # Compare RDMs
            rdms_comp = rsatoolbox.rdm.compare(rdms_flat, rdms_flat, method=rdm_comp_method)
            if rdm_comp_method == "cosine":
                rdms_comp = np.arccos(np.clip(rdms_comp, -1, 1))
            rdms_comp = np.nan_to_num(rdms_comp, nan=0.0)
            rdms_comp = (rdms_comp + rdms_comp.T) / 2.0

            # MDS - 2 Components for 2D plot
            transformer = manifold.MDS(
                n_components=2,
                metric=True,
                max_iter=3000,
                n_init=30,
                normalized_stress=True,
                dissimilarity="precomputed",
            )
            dims = transformer.fit_transform(rdms_comp)
            
            # Rotate 90 degrees clockwise
            dims_rot = np.zeros_like(dims)
            dims_rot[:, 0] = dims[:, 1]
            dims_rot[:, 1] = -dims[:, 0]
            dims = dims_rot

            # Map keys to coords
            keys = list(flat_features.keys())
            coord_map = {k: dims[i] for i, k in enumerate(keys)}

            # Plotting Config
            num_layers = len(layer_list)
            cmap = plt.get_cmap("tab10") 
            layer_base_colors = [cmap(i) for i in range(num_layers)]

            results_dir = Path("results") / "Joint_Structure"
            results_dir.mkdir(parents=True, exist_ok=True)

            fig, ax = plt.subplots(figsize=(10, 8))
            
            all_coords = dims
            amin, amax = all_coords.min(), all_coords.max()
            center = (amin + amax) / 2.0
            half_span = max((amax - amin) / 2.0, 1e-6)
            axis_min = center - half_span * 1.1
            axis_max = center + half_span * 1.1
            
            legend_handles = []
            legend_labels = []

            for l_idx, layer_name in enumerate(layer_list):
                base_color = layer_base_colors[l_idx]
                
                # Gather coords
                keys_for_layer = [k for k in keys if k.startswith(f"{layer_name}_t")]
                keys_for_layer.sort(key=lambda x: int(x.split('_t')[1]))
                
                layer_coords = []
                for key in keys_for_layer:
                    if key in coord_map:
                        layer_coords.append(coord_map[key])
                
                layer_coords = np.array(layer_coords)
                n_steps = len(layer_coords)
                
                if n_steps > 0:
                    layer_colors_grad = []
                    for t in range(n_steps):
                        alpha = 0.2 + 0.8 * (t / (n_steps - 1)) if n_steps > 1 else 1.0
                        c = list(base_color)
                        c[3] = alpha 
                        layer_colors_grad.append(tuple(c))
                    
                    if n_steps > 1:
                        points = layer_coords.reshape(-1, 1, 2)
                        segments = np.concatenate([points[:-1], points[1:]], axis=1)
                        seg_colors = layer_colors_grad[:-1]
                        lc = LineCollection(segments, colors=seg_colors, linewidths=2, alpha=1.0)
                        ax.add_collection(lc)

                    ax.scatter(layer_coords[:, 0], layer_coords[:, 1], c=layer_colors_grad, s=40, edgecolors='none', zorder=10)
                    
                    # Legend Entry
                    from matplotlib.lines import Line2D
                    handle = Line2D([0], [0], marker='o', color='w', label=layer_name,
                                    markerfacecolor=base_color, markersize=15, linestyle='None')
                    legend_handles.append(handle)
                    legend_labels.append(layer_name)

            ax.set_xlim([axis_min, axis_max])
            ax.set_ylim([axis_min, axis_max])
            ax.set_aspect('equal')
            ax.set_xlabel("MDS Dim 1")
            ax.set_ylabel("MDS Dim 2")
            ax.set_title(f"Joint MDS: {model_name}")
            
            legend = ax.legend(handles=legend_handles, labels=legend_labels, 
                               loc='center left', bbox_to_anchor=(1, 0.5), 
                               frameon=True, fontsize=14)
            legend.get_frame().set_facecolor('#e0e0e0')
            legend.get_frame().set_edgecolor('none')
            
            for spine in ax.spines.values():
                spine.set_color("black")
            ax.set_facecolor("white")
            ax.grid(True, linestyle='--', alpha=0.5)

            save_path = results_dir / f"joint_structure_{model_name}.png"
            fig.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close(fig)
            output_paths.append(str(save_path))
        
    return output_paths

def plot_rdm_per_timestep(
    args,
    imgs,
    labels,
    cache_dir="blt_local_cache",
    steps=None,
    target_layers=None,
    max_steps=None,
    rdm_calc_method="euclidean",
    rdm_comp_method="cosine", # Added comparison method
    split_by_label=False,
    rdm_cmap="Blues",  # Colormap option: "Blues", "magma", "viridis"
):
    """
    Computes/Plots a 'Second-order RDM' (RDM of RDMs).
    The resulting matrix (approx 75x75) shows the similarity between the representation 
    at (Layer X, Time Y) and (Layer A, Time B).
    """
    
    # cache_root = Path(cache_dir)
    # model_files = sorted(cache_root.glob("*/blt_full_objects.pt"))
    
    # Use the specific model path provided in args
    model_path = Path(getattr(args, 'model_path'))
    if not model_path.exists():
        print(f"Model path does not exist: {model_path}")
        return []

    model_files = [model_path]
    
    if getattr(args, "dry_run", False):
        print("Dry run: processing specified model")
    
    if not model_files:
        return []

    output_paths = []

    for model_path in model_files:
        model_name = model_path.parent.name
        model_name = "_".join(model_name.split("_")[:2])
        # Filter removed to allow explicit model path selection
        # if "vggface" not in model_name or "imagenet" in model_name:
        #     continue

        print(f"Processing (RDM of RDMs): {model_name}")
        model, _, _ = load_model_path(str(model_path), print_model=False)
        model.to(args.device)
        model.eval()

        model_steps = steps
        if getattr(args, "dry_run", False):
            model_steps = 2
            print("Dry run: reduced steps to 2")
        elif model_steps is None:
            model_steps = getattr(model, "times", getattr(model, "num_recurrence", 1))
        
        if max_steps is not None:
            model_steps = min(model_steps, max_steps)

        layer_list = target_layers
        if layer_list is None:
            layer_list = ["output_0", "output_1", "output_2", "output_3", "output_4", "output_5"]

        imgs_device = imgs.to(args.device)
        
        # 1. Collect all "Layer_Time" features into a single ordered dictionary
        # Extract all features once
        all_layer_activations = {}
        for layer_name in layer_list:
            activations = extract_recurrent_steps(model, imgs_device, layer_name, steps=model_steps)
            if max_steps is not None:
                activations = activations[:max_steps]
            all_layer_activations[layer_name] = activations

        # Determine Groups to Process
        groups_to_process = []
        
        # Always process All/Joint data first
        groups_to_process.append( ("", None, "Joint") )
        
        if split_by_label and labels is not None:
            unique_labels = sorted(torch.unique(labels).tolist())
            # Map index to name (assuming 0=face, 1=object based on check)
            label_map = {0: "Face", 1: "Object"}
            
            for lbl in unique_labels:
                mask = (labels == lbl).cpu().numpy()
                count = np.sum(mask)
                if count < 2:
                    print(f"Skipping label {lbl} for RDM: insufficient samples ({count})")
                    continue
                
                # Use descriptive name if available, else generic
                name = label_map.get(lbl, f"Label{lbl}")
                # Tuple: (Suffix for filename, Mask, Display Name)
                groups_to_process.append( (f"_{name}", mask, name) )

        # Process each group
        for suffix_lbl, mask, display_name in groups_to_process:
            if mask is not None:
                print(f"Processing RDM-of-RDMs for: {display_name}")
            
            # Build flat features for this group
            flat_features = OrderedDict()
            # Also rebuild axis tracking logic as it depends on flattened sequence
            current_idx = 0
            layer_boundaries = []
            
            for layer_name in layer_list:
                layer_boundaries.append(current_idx)
                act_list = all_layer_activations[layer_name]
                
                for t_idx, feat in enumerate(act_list):
                    # feat is (Batch, D)
                    if mask is not None:
                        sub_feat = feat[mask]
                    else:
                        sub_feat = feat
                    
                    key = f"{layer_name}_t{t_idx}"
                    flat_features[key] = sub_feat
                    current_idx += 1
            
            layer_boundaries.append(current_idx)

            if not flat_features:
                continue

            # 2. Compute RDMs
            rdms_flat, _ = calc_rdms(args, flat_features, method=rdm_calc_method)
            
            # 3. Perform Comparisons and Plotting (Two Loop Passes)
            # Different limits for Joint vs Face/Object plots
            # Joint uses higher limits, Face/Object use tighter limits
            is_joint = (display_name == "Joint")
            
            comparisons = [
                {
                    "method": "cosine",
                    "label": "Dissimilarity (1 - cosine)",
                    "cmap": rdm_cmap,  # Use parameter instead of hardcoded
                    "suffix": "cosine",
                    "process_func": lambda x: np.clip(1.0 - x, 0, 2),
                    "limits": (0.0, 0.25) if is_joint else (0.0, 0.13)
                },
                {
                    "method": "spearman",
                    "label": "Dissimilarity (1 - Spearman)",
                    "cmap": rdm_cmap,  # Use parameter instead of hardcoded
                    "suffix": "spearman",
                    "process_func": lambda x: 1.0 - x,
                    "limits": (0.0, 1.2) if is_joint else (0.0, 1.1)
                }
            ]

            for comp in comparisons:
                method_name = comp["method"]
                rdms_comp = rsatoolbox.rdm.compare(rdms_flat, rdms_flat, method=method_name)
                
                # Symmetrize and Fix NaNs
                rdms_comp = np.nan_to_num(rdms_comp, nan=0.0)
                rdms_comp = (rdms_comp + rdms_comp.T) / 2.0
                
                # Custom processing
                rdms_comp = comp["process_func"](rdms_comp)
                
                # Log actual max for reference
                actual_max = np.max(rdms_comp)
                print(f"  {comp['suffix']}: actual max dissimilarity = {actual_max:.4f}, using vmax = {comp['limits'][1]}")
                
                # 4. Plot
                save_dir = Path("results") / "RDM_Timesteps"
                save_dir.mkdir(parents=True, exist_ok=True)
                
                fig, ax = plt.subplots(figsize=(12, 10))
                
                # Apply fixed limits based on plot type
                vargs = {"vmin": comp["limits"][0], "vmax": comp["limits"][1]}

                im = ax.imshow(rdms_comp, cmap=comp["cmap"], origin='upper', **vargs)
                
                # Colorbar
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(comp["label"], rotation=270, labelpad=15)
                
                # Axis labels and layer boundaries
                layer_centers = []
                for i in range(len(layer_boundaries) - 1):
                    start = layer_boundaries[i]
                    end = layer_boundaries[i+1]
                    layer_centers.append((start + end - 1) / 2.0)
                    
                    # Add black grid lines at layer boundaries
                    if i > 0:
                        ax.axhline(start - 0.5, color='black', linewidth=1)
                        ax.axvline(start - 0.5, color='black', linewidth=1)
                
                ax.set_xticks(layer_centers)
                ax.set_xticklabels(layer_list, rotation=45, ha='right')
                ax.set_yticks(layer_centers)
                ax.set_yticklabels(layer_list)
                
                # Remove tick marks but keep tick labels
                ax.tick_params(axis='both', which='both', length=0)
                
                # Remove default grid lines
                ax.grid(False)
                
                # Set Title - Joint without "only", Face/Object with "only"
                if display_name == "Joint":
                    ax.set_title(f"RDM of RDMs ({display_name})")
                elif display_name in ["Face", "Object"]:
                    ax.set_title(f"RDM of RDMs ({display_name} only)")
                    
                ax.set_xlabel(u"Layer (Time \u2192)")
                ax.set_ylabel(u"Layer (Time \u2193)")
        
                # Include colormap name in filename for easy identification
                cmap_suffix = f"_{rdm_cmap}" if rdm_cmap != "Blues" else ""
                save_path = save_dir / f"RDM_of_RDMs_{model_name}{suffix_lbl}_{comp['suffix']}{cmap_suffix}.png"
                fig.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close(fig)
                output_paths.append(str(save_path))
                print(f"Saved {comp['label']} plot to {save_path}")

    return output_paths
