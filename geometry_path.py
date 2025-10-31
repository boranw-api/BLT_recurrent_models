
import contextlib
import io
# Standard library imports
import logging
from collections import OrderedDict

# External libraries: General utilities
import numpy as np

# PyTorch related imports
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision.utils import make_grid

# Matplotlib for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt

# Scikit-Learn for machine learning utilities
from sklearn.decomposition import PCA
from sklearn import manifold

# RSA toolbox specific imports
import rsatoolbox
from rsatoolbox.data import Dataset
from rsatoolbox.rdm.calc import calc_rdm

# new imports for our plotting 
from analyze_representations import kasper_dataset, load_model_path
from torch.utils.data import TensorDataset, random_split

from pathlib import Path

def fetch_dataloaders(args):
    """
    Fetches the data loaders for training and testing datasets.

    Inputs:
    - args (Namespace): Parsed arguments with training configuration.

    Outputs:
    - train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
    - test_loader (torch.utils.data.DataLoader): DataLoader for the test data.
    """
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if args.use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    with contextlib.redirect_stdout(io.StringIO()): #to suppress output
        dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
        dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
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

def extract_features(model, imgs, return_layers, plot='none'):
    """
    Extracts features from specified layers of the model.

    Inputs:
    - model (torch.nn.Module): The model from which to extract features.
    - imgs (torch.Tensor): Batch of input images.
    - return_layers (list): List of layer names from which to extract features.
    - plot (str): Option to plot the features. Default is 'none'.

    Outputs:
    - model_features (dict): A dictionary with layer names as keys and extracted features as values.
    """
    if return_layers == 'all':
        return_layers, _ = get_graph_node_names(model)
    elif return_layers == 'layers':
        layers, _ = get_graph_node_names(model)
        return_layers = [l for l in layers if 'input' in l or 'conv' in l or 'fc' in l]

    feature_extractor = create_feature_extractor(model, return_nodes=return_layers)
    model_features = feature_extractor(imgs)

    return model_features

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

def sample_images(data_loader, n=5, plot=False):
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

def plot_rdms(model_rdms):
    """
    Plots the Representational Dissimilarity Matrices (RDMs) for each layer of a model.

    Inputs:
    - model_rdms (dict): A dictionary where keys are layer names and values are the corresponding RDMs.
    """
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, len(model_rdms))
    fig.subplots_adjust(wspace=0.2, hspace=0.2)

    for l in range(len(model_rdms)):
        layer = list(model_rdms.keys())[l]
        rdm = np.squeeze(model_rdms[layer])

        if len(rdm.shape) < 2:
            rdm = rdm.reshape((int(np.sqrt(rdm.shape[0])), int(np.sqrt(rdm.shape[0]))))

        rdm = rdm / np.max(rdm)

        ax = plt.subplot(gs[0, l])
        ax_ = ax.imshow(rdm, cmap='magma_r')
        ax.set_title(f'{layer}')

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.18, 0.01, 0.53])
    cbar_ax.text(-2.3, 0.05, 'Normalized euclidean distance', size=10, rotation=90)
    fig.colorbar(ax_, cax=cbar_ax)

    plt.show()

def rep_path(
    model_features,
    model_colors,
    labels=None,
    rdm_calc_method='euclidean',
    rdm_comp_method='cosine',
    ax=None,
    legend=True,
    title=None,
    save_path=None,
):
    """
    Represents paths of model features in a reduced-dimensional space.

    Inputs:
    - model_features (dict): Dictionary containing model features for each model.
    - model_colors (dict): Dictionary mapping model names to colors for visualization.
    - labels (array-like, optional): Array of labels corresponding to the model features.
    - rdm_calc_method (str, optional): Method for calculating RDMs ('euclidean' or 'correlation').
    - rdm_comp_method (str, optional): Method for comparing RDMs ('cosine' or 'corr').
    - ax (matplotlib.axes.Axes, optional): Axis to draw on. If None, a new figure is created.
    - legend (bool, optional): Whether to display the legend. Default is True when creating a standalone figure.
    - title (str, optional): Custom subplot title. Defaults to a generic title when None.
    - save_path (str, optional): Optional path for saving the figure when `ax` is None.
    """

    model_names = list(model_features.keys())
    path_len = []
    path_colors = []
    rdms_list = []
    ax_ticks = []
    tick_colors = []

    for model_name in model_names:
        features = model_features[model_name]
        path_colors.append(model_colors[model_name])
        path_len.append(len(features))
        ax_ticks.append(list(features.keys()))
        tick_colors.append([model_colors[model_name]] * len(features))
        rdms, _ = calc_rdms(features, method=rdm_calc_method)
        rdms_list.append(rdms)

    path_len = np.insert(np.cumsum(path_len), 0, 0)

    include_labels = labels is not None
    if include_labels:
        rdms, _ = calc_rdms({'labels': F.one_hot(labels).float().to(args.device)}, method=rdm_calc_method)
        rdms_list.append(rdms)
        ax_ticks.append(['labels'])
        tick_colors.append(['m'])

    rdms = rsatoolbox.rdm.concat(rdms_list)

    ax_ticks = [tick for layer_ticks in ax_ticks for tick in layer_ticks]
    tick_colors = [color for layer_colors in tick_colors for color in layer_colors]
    tick_colors = ['k' if tick == 'input' else color for tick, color in zip(ax_ticks, tick_colors)]

    rdms_comp = rsatoolbox.rdm.compare(rdms, rdms, method=rdm_comp_method)
    if rdm_comp_method == 'cosine':
        rdms_comp = np.clip(rdms_comp, -1, 1)
        rdms_comp = np.arccos(rdms_comp)
    rdms_comp = np.nan_to_num(rdms_comp, nan=0.0)
    rdms_comp = (rdms_comp + rdms_comp.T) / 2.0

    remove_indices = np.where(np.array(ax_ticks) == 'input')[0][1:]
    for index in sorted(remove_indices, reverse=True):
        del ax_ticks[index]
        del tick_colors[index]
        rdms_comp = np.delete(np.delete(rdms_comp, index, axis=0), index, axis=1)
        for m in range(len(path_len) - 1):
            if path_len[m] <= index < path_len[m + 1]:
                for k in range(m + 1, len(path_len)):
                    path_len[k] -= 1
                break

    transformer = manifold.MDS(
        n_components=2,
        max_iter=1000,
        n_init=10,
        normalized_stress='auto',
        dissimilarity="precomputed",
    )
    dims = transformer.fit_transform(rdms_comp)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        created_fig = True
    else:
        fig = ax.figure

    amin, amax = dims.min(), dims.max()
    amin, amax = (amin + amax) / 2 - (amax - amin) * 5 / 8, (amin + amax) / 2 + (amax - amin) * 5 / 8

    for spine in ax.spines.values():
        spine.set_color('#7f7f7f')
    ax.set_facecolor('white')

    for i in range(len(rdms_list) - (1 if include_labels else 0)):
        path_indices = np.arange(path_len[i], path_len[i + 1])
        path_indices = path_indices[path_indices < len(dims)]
        if len(path_indices) == 0:
            continue
        ax.plot(
            dims[path_indices, 0],
            dims[path_indices, 1],
            color=path_colors[i],
            marker='.',
        )

    idx_input = ax_ticks.index('input') if 'input' in ax_ticks else None
    if idx_input is not None and idx_input < len(dims):
        ax.plot(dims[idx_input, 0], dims[idx_input, 1], color='k', marker='s')

    if include_labels:
        idx_labels = ax_ticks.index('labels')
        if idx_labels < len(dims):
            ax.plot(dims[idx_labels, 0], dims[idx_labels, 1], color='m', marker='*')

    default_title = title or 'Representational Geometry Path'
    ax.set_title(default_title, fontdict={'fontsize': 12})
    ax.set_xlim([amin, amax])
    ax.set_ylim([amin, amax])
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")

    if legend and model_names:
        ax.legend(model_names, fontsize=8)

    if created_fig:
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight')

    return dims

def plot_recurrence_grid(
    model,
    imgs,
    labels,
    layer_categories,
    steps=None,
    category_colors=None,
    max_steps=None,
    save_path="representational_geometry_grid.png",
):
    """Plots a 3x3 grid of representational paths across layer categories.

    Args:
        model (torch.nn.Module): Recurrent BLT model.
        imgs (torch.Tensor): Batch of images used to probe the model.
        labels (torch.Tensor): Ground-truth labels for the images.
        layer_categories (OrderedDict): Mapping of category name to list of layer names.
        steps (int, optional): Number of recurrence steps to capture. Defaults to model.times when None.
        category_colors (dict, optional): Mapping of category name to color hex codes.
        max_steps (int, optional): Optional cap on the number of time steps plotted per layer.
        save_path (str, optional): Where to save the resulting figure.
    """

    if steps is None:
        steps = getattr(model, "times", getattr(model, "num_recurrence", 1))

    if category_colors is None:
        category_colors = {
            "Early feature extractors": "#1f77b4",
            "Intermediate representations": "#59402a",
            "Late identity readout": "#2ca02c",
        }

    imgs_device = imgs.to(args.device)
    layer_step_features = {}
    for layers in layer_categories.values():
        for layer_name in layers:
            if layer_name in layer_step_features:
                continue
            activations = extract_recurrent_steps(model, imgs_device, layer_name, steps=steps)
            if max_steps is not None:
                activations = activations[:max_steps]
            step_features = {
                f"{layer_name}_t{idx}": feat for idx, feat in enumerate(activations)
            }
            layer_step_features[layer_name] = step_features

    fig, axes = plt.subplots(len(layer_categories), 3, figsize=(18, 18))

    for row_idx, (category_name, layers) in enumerate(layer_categories.items()):
        row_color = category_colors.get(category_name, '#7f7f7f')
        for col_idx, layer_name in enumerate(layers):
            ax = axes[row_idx, col_idx]
            step_features = layer_step_features.get(layer_name)
            if not step_features:
                ax.axis('off')
                ax.set_title(f"{layer_name}\n(no activations)")
                continue

            features = {layer_name: step_features}
            model_colors = {layer_name: row_color}
            legend = (row_idx == 0 and col_idx == 0)
            rep_path(
                features,
                model_colors,
                labels,
                ax=ax,
                legend=legend,
                title=layer_name,
            )

            if col_idx == 0:
                ax.set_ylabel(category_name, fontsize=14, rotation=90, labelpad=20)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
    return fig
    
def plot_dim_reduction(model_features, labels, transformer_funcs):
    """
    Plots the dimensionality reduction results for model features using various transformers.

    Inputs:
    - model_features (dict): Dictionary containing model features for each layer.
    - labels (array-like): Array of labels corresponding to the model features.
    - transformer_funcs (list): List of dimensionality reduction techniques to apply ('PCA', 'MDS', 't-SNE').
    """

    transformers = []
    for t in transformer_funcs:
        if t == 'PCA': transformers.append(PCA(n_components=2))
        if t == 'MDS': transformers.append(manifold.MDS(n_components = 2, normalized_stress='auto'))
        if t == 't-SNE': transformers.append(manifold.TSNE(n_components = 2, perplexity=40, verbose=0))

    fig = plt.figure(figsize=(8, 2.5*len(transformers)))
    # and we add one plot per reference point
    gs = fig.add_gridspec(len(transformers), len(model_features))
    fig.subplots_adjust(wspace=0.2, hspace=0.2)

    return_layers = list(model_features.keys())

    for f in range(len(transformer_funcs)):

        for l in range(len(return_layers)):
            layer =  return_layers[l]
            feats = model_features[layer].detach().cpu().flatten(1)
            feats_transformed= transformers[f].fit_transform(feats)

            amin, amax = feats_transformed.min(), feats_transformed.max()
            amin, amax = (amin + amax) / 2 - (amax - amin) * 5/8, (amin + amax) / 2 + (amax - amin) * 5/8
            ax = plt.subplot(gs[f,l])
            ax.set_xlim([amin, amax])
            ax.set_ylim([amin, amax])
            ax.axis("off")
            #ax.set_title(f'{layer}')
            if f == 0: ax.text(0.5, 1.12, f'{layer}', size=16, ha="center", transform=ax.transAxes)
            if l == 0: ax.text(-0.3, 0.5, transformer_funcs[f], size=16, ha="center", transform=ax.transAxes)
            # Create a discrete color map based on unique labels
            num_colors = len(np.unique(labels))
            cmap = plt.get_cmap('viridis_r', num_colors) # 10 discrete colors
            norm = mpl.colors.BoundaryNorm(np.arange(-0.5,num_colors), cmap.N)
            ax_ = ax.scatter(feats_transformed[:, 0], feats_transformed[:, 1], c=labels, cmap=cmap, norm=norm)

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([1.01, 0.18, 0.01, 0.53])
    fig.colorbar(ax_, cax=cbar_ax, ticks=np.linspace(0,9,10))
    plt.show()

# plotting in this way allows cross plot comparison of label (star) location
def plot_recurrence_grid_joint_embedding(
    args,
    model,
    imgs,
    labels,
    layer_categories,
    steps=None,
    category_colors=None,
    max_steps=None,
    rdm_calc_method="euclidean",
    rdm_comp_method="cosine",
    save_path="representational_geometry_grid_aligned.png",
):
    if steps is None:
        steps = getattr(model, "times", getattr(model, "num_recurrence", 1))

    if category_colors is None:
        category_colors = {
            "Early Layers": "#1f77b4",
            "Intermediate Layers": "#ff7f0e",
            "Late Layers": "#f70909",
        }

    imgs_device = imgs.to(args.device)

    layer_step_features = OrderedDict()
    for layers in layer_categories.values():
        for layer_name in layers:
            if layer_name in layer_step_features:
                continue
            activations = extract_recurrent_steps(model, imgs_device, layer_name, steps=steps)
            if max_steps is not None:
                activations = activations[:max_steps]
            step_features = OrderedDict(
                (f"{layer_name}_t{idx}", feat) for idx, feat in enumerate(activations)
            )
            layer_step_features[layer_name] = step_features

    flat_features = OrderedDict()
    layer_key_map = OrderedDict()
    for layer_name, step_features in layer_step_features.items():
        keys = list(step_features.keys())
        layer_key_map[layer_name] = keys
        for key in keys:
            flat_features[key] = step_features[key]

    if not flat_features:
        raise ValueError("No activations collected; check layer names or steps.")

    rdms_flat, _ = calc_rdms(args, flat_features, method=rdm_calc_method)
    rdms = rdms_flat
    ax_ticks = list(flat_features.keys())

    include_labels = labels is not None
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

    transformer = manifold.MDS(
        n_components=2,
        max_iter=1000,
        n_init=10,
        normalized_stress="auto",
        dissimilarity="precomputed",
    )
    dims = transformer.fit_transform(rdms_comp)

    coord_map = {tick: dims[idx] for idx, tick in enumerate(ax_ticks)}

    coords_array = np.vstack(list(coord_map.values()))
    amin, amax = coords_array.min(), coords_array.max()
    center = (amin + amax) / 2.0
    half_span = max((amax - amin) / 2.0, 1e-6)
    axis_min = center - half_span * 1.25
    axis_max = center + half_span * 1.25

    fig, axes = plt.subplots(len(layer_categories), 3, figsize=(18, 18))

    for row_idx, (category_name, layers) in enumerate(layer_categories.items()):
        row_color = category_colors.get(category_name, "#b3cde0")
        for col_idx, layer_name in enumerate(layers):
            ax = axes[row_idx, col_idx]
            keys = layer_key_map.get(layer_name, [])
            if not keys:
                ax.axis("off")
                ax.set_title(f"{layer_name}\n(no activations)")
                continue

            coords = np.vstack([coord_map[k] for k in keys if k in coord_map])
            if coords.size == 0:
                ax.axis("off")
                ax.set_title(f"{layer_name}\n(no coords)")
                continue

            ax.plot(coords[:, 0], coords[:, 1], color=row_color, marker=".")
            ax.set_xlim([axis_min, axis_max])
            ax.set_ylim([axis_min, axis_max])
            ax.set_xlabel("dim 1")
            ax.set_ylabel("dim 2")
            ax.set_title(layer_name)

            if keys:
                start_coord = coord_map[keys[0]]
                ax.plot(start_coord[0], start_coord[1], color="k", marker="s")

            if include_labels:
                label_coord = coord_map["labels"]
                ax.plot(label_coord[0], label_coord[1], color="m", marker="*")

            if row_idx == 0 and col_idx == 0:
                ax.legend([category_name], fontsize=8, loc="upper right")

            if col_idx == 0:
                ax.set_ylabel(category_name, fontsize=14, rotation=90, labelpad=20)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig

# plot for all models in the cache
def plot_cache_models_joint_embedding(
    args,
    imgs,
    labels,
    layer_categories,
    cache_dir="blt_local_cache",
    max_steps=None,
):
    cache_root = Path(cache_dir)
    model_files = sorted(cache_root.glob("*/blt_full_objects.pt"))
    output_paths = []

    for model_path in model_files:
        model_name = model_path.parent.name
        model, _, _ = load_model_path(str(model_path), print_model=False)
        model.to(args.device)
        model.eval()

        steps = getattr(model, "times", getattr(model, "num_recurrence", None))
        if max_steps is not None:
            steps = min(steps, max_steps) if steps is not None else max_steps

        save_path = f"representational_geometry_grid_aligned_{model_name}.png"
        fig = plot_recurrence_grid_joint_embedding(
            args,
            model,
            imgs,
            labels,
            layer_categories,
            steps=steps,
            max_steps=max_steps,
            save_path=save_path,
        )
        output_paths.append(save_path)

    return output_paths