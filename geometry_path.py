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
    
def amir_dataset(
    pickle_path: str = "./blt_local_cache/face_object_dataset.pkl",
    label_key: str = "group",
    enforce_channels_first: bool = True,
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

    return imgs.contiguous(), labels, config

def amir_loaders(
    args,
    *,
    pickle_path: str = "./blt_local_cache/face_object_dataset.pkl",
    label_key: str = "group",
):
    imgs, labels, config = amir_dataset(pickle_path=pickle_path, label_key=label_key)

    train_kwargs = {"batch_size": args.batch_size, "shuffle": True}
    test_kwargs = {"batch_size": args.test_batch_size, "shuffle": False}
    if args.use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    dataset = TensorDataset(imgs, labels)
    dataset_fraction = float(getattr(args, "train_split", 0.8))
    dataset_fraction = min(max(dataset_fraction, 0.0), 1.0)

    total_items = len(dataset)

    train_size = max(1, int(round(total_items * dataset_fraction)))
    if train_size >= total_items:
        train_size = total_items - 1
    test_size = total_items - train_size

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
    Generates a 1xN subplot figure (default N=6) where each subplot visualizes
    a single layer's trajectory across recurrent steps in a shared MDS space.
    """
    cache_root = Path(cache_dir)
    model_files = sorted(cache_root.glob("*/blt_full_objects.pt"))
    output_paths = []
    if not model_files:
        return output_paths

    all_flat_features = OrderedDict()
    model_layer_maps = OrderedDict()

    for model_path in model_files:
        model_name = model_path.parent.name
        model_name = "_".join(model_name.split("_")[:2])
        if "vggface" not in model_name or "imagenet" in model_name:
            continue  # Skip models that do not contain "vggface" or contain "imagenet" in their name
        print(f"Processing (1x6): {model_name}")
        model, _, _ = load_model_path(str(model_path), print_model=False)
        model.to(args.device)
        model.eval()

        model_steps = steps
        if model_steps is None:
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
        # rdms_comp = np.clip(rdms_comp, -1, 1)
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

    for model_path in model_files:
        model_name = model_path.parent.name
        model_name = "_".join(model_name.split("_")[:2])
        if "vggface" not in model_name or "imagenet" in model_name:
            continue  # Skip models that do not contain "vggface" or contain "imagenet" in their name
        layer_order = model_layer_maps[model_name]["layer_order"]
        prefixed_layer_keys = model_layer_maps[model_name]["prefixed_layer_keys"]

        num_layers = len(layer_order)
        rows = 2
        cols = 3
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(figsize_per_panel * cols, figsize_per_panel * rows * 1.2),
            squeeze=False,
            subplot_kw={'projection': '3d'} if plot_dim == 3 else None,
        )

        for idx, layer_name in enumerate(layer_order):
            ax = axes[idx // cols, idx % cols]
            prefixed_keys = prefixed_layer_keys.get(layer_name, [])
            coords = [coord_map[k] for k in prefixed_keys if k in coord_map]
            if not coords:
                ax.axis("off")
                ax.set_title(f"{layer_name}\n(no activations)")
                continue

            coords = np.vstack(coords)
            num_steps = len(coords)
            cmap = plt.get_cmap("viridis_r")
            colors = cmap(np.linspace(0, 1, num_steps))

            if plot_dim == 3:
                for step_idx in range(num_steps - 1):
                    ax.plot(
                        coords[step_idx:step_idx + 2, 0],
                        coords[step_idx:step_idx + 2, 1],
                        coords[step_idx:step_idx + 2, 2],
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
                        coords[step_idx:step_idx + 2, 0],
                        coords[step_idx:step_idx + 2, 1],
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

            # Remove grayish grid background
            for spine in ax.spines.values():
                spine.set_color('black')
            ax.set_facecolor('white')
            ax.grid(False)

            # distance plotting
            # Total path length: The sum of Euclidean distances between consecutive points in the MDS space. This measures the overall amount of movement along the trajectory.

            # Average step length: The total path length divided by the number of steps (num_steps - 1). This normalizes the metric by the number of transitions, allowing comparison across layers with different numbers of dots.
            distances = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
            total_path_length = np.sum(distances)
            avg_step_length = total_path_length / (num_steps - 1)
            
            # Add metrics as text on the subplot
            text_str = f'Total: {total_path_length:.2f}\nAvg: {avg_step_length:.2f}'
            text_kwargs = dict(transform=ax.transAxes, fontsize=8, verticalalignment='top', 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            if plot_dim == 3:
                ax.text2D(0.05, 0.95, text_str, **text_kwargs)
            else:
                ax.text(0.05, 0.95, text_str, **text_kwargs)

        colorbar_width = 2.5 / num_layers
        colorbar_left = (1.0 - colorbar_width) / 2.0
        cbar_ax = fig.add_axes([colorbar_left, 0.08, colorbar_width, 0.02])
        
        # maybe not normalize, because the number of steps is not fixed.
        # sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
        sm = plt.cm.ScalarMappable(cmap='viridis_r')
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Timestep')
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['early', 'mid', 'late'])

        fig.suptitle(f"Representational Trajectories - {model_name}", y=1.02)
        fig.tight_layout(rect=[0, 0.15, 1, 1])

        save_path = results_dir / f"rep_geo_single_mds_{model_name}_2x3.png"
        fig.savefig(save_path, bbox_inches="tight")
        output_paths.append(str(save_path))
        
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
        if model_steps is None:
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

def plot_joint_3d_interactive(
    args,
    imgs,
    labels,
    cache_dir="blt_local_cache",
    steps=None,
    target_layers=None,
    max_steps=None,
    rdm_calc_method="euclidean",
    rdm_comp_method="cosine",
):
    """
    Plots all layers and time steps in a single 3D MDS space.
    Produces both a static Matplotlib plot and an interactive Plotly HTML plot.
    """
    cache_root = Path(cache_dir)
    model_files = sorted(cache_root.glob("*/blt_full_objects.pt"))
    output_paths = []

    if not model_files:
        return output_paths

    for model_path in model_files:
        model_name = model_path.parent.name
        model_name = "_".join(model_name.split("_")[:2])
        if "vggface" not in model_name or "imagenet" in model_name:
            continue

        print(f"Processing (Joint 3D Structure): {model_name}")
        model, _, _ = load_model_path(str(model_path), print_model=False)
        model.to(args.device)
        model.eval()

        model_steps = steps
        if model_steps is None:
            model_steps = getattr(model, "times", getattr(model, "num_recurrence", 1))

        layer_list = target_layers
        if layer_list is None:
            layer_list = ["output_0", "output_1", "output_2", "output_3", "output_4", "output_5"]

        imgs_device = imgs.to(args.device)
        
        # Extract features
        layer_step_features = OrderedDict()
        for layer_name in layer_list:
            activations = extract_recurrent_steps(model, imgs_device, layer_name, steps=model_steps)
            if max_steps is not None:
                activations = activations[:max_steps]
            
            step_features = OrderedDict(
                (f"{layer_name}_t{idx}", feat) for idx, feat in enumerate(activations)
            )
            layer_step_features[layer_name] = step_features

        # Flatten for RDM
        flat_features = OrderedDict()
        for layer_name in layer_list:
            for t_idx in range(len(layer_step_features[layer_name])):
                key = f"{layer_name}_t{t_idx}"
                flat_features[key] = layer_step_features[layer_name][key]

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

        # MDS
        transformer = manifold.MDS(
            n_components=3,
            metric=True,
            max_iter=3000,
            n_init=30,
            normalized_stress=True,
            dissimilarity="precomputed",
        )
        dims = transformer.fit_transform(rdms_comp)
        
        # Map keys to coords
        keys = list(flat_features.keys())
        coord_map = {k: dims[i] for i, k in enumerate(keys)}

        # Prepare data for plotting
        num_layers = len(layer_list)
        first_layer_keys = list(layer_step_features[layer_list[0]].keys())
        num_steps = len(first_layer_keys)
        cmap = plt.get_cmap("viridis")
        layer_colors = cmap(np.linspace(0, 1, num_layers))

        results_dir = Path("results") / "3D_structure"
        results_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. Static Matplotlib Plot ---
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for l_idx, layer_name in enumerate(layer_list):
            layer_coords = []
            for t in range(num_steps):
                key = f"{layer_name}_t{t}"
                if key in coord_map:
                    layer_coords.append(coord_map[key])
            
            layer_coords = np.array(layer_coords)
            if len(layer_coords) > 0:
                ax.plot(layer_coords[:, 0], layer_coords[:, 1], layer_coords[:, 2], 
                        color=layer_colors[l_idx], label=f"{layer_name}", linewidth=2, alpha=0.9)
                ax.scatter(layer_coords[:, 0], layer_coords[:, 1], layer_coords[:, 2], 
                           color=layer_colors[l_idx], s=25)
                ax.text(layer_coords[0, 0], layer_coords[0, 1], layer_coords[0, 2], f"{layer_name}", fontsize=9)

                # Connect to the start of the next layer
                if l_idx < num_layers - 1:
                    next_layer_name = layer_list[l_idx + 1]
                    next_layer_start = None
                    for t_next in range(num_steps):
                        key_next = f"{next_layer_name}_t{t_next}"
                        if key_next in coord_map:
                            next_layer_start = coord_map[key_next]
                            break
                    
                    if next_layer_start is not None:
                        start_pt = layer_coords[-1]
                        end_pt = next_layer_start
                        ax.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], [start_pt[2], end_pt[2]],
                                color='gray', linestyle='--', linewidth=1, alpha=0.5)

        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")
        ax.set_title(f"Joint Layer-Time Structure: {model_name}")
        ax.legend()

        save_path = results_dir / f"joint_structure_{model_name}.png"
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(str(save_path))

        # --- 2. Interactive Plotly Plot ---
        fig_interactive = go.Figure()
        
        for l_idx, layer_name in enumerate(layer_list):
            layer_coords = []
            for t in range(num_steps):
                key = f"{layer_name}_t{t}"
                if key in coord_map:
                    layer_coords.append(coord_map[key])
            layer_coords = np.array(layer_coords)
            
            if len(layer_coords) > 0:
                c_hex = mpl.colors.to_hex(layer_colors[l_idx])
                
                # Layer trajectory
                fig_interactive.add_trace(go.Scatter3d(
                    x=layer_coords[:, 0], y=layer_coords[:, 1], z=layer_coords[:, 2],
                    mode='lines+markers',
                    name=layer_name,
                    line=dict(color=c_hex, width=5),
                    marker=dict(size=4, color=c_hex)
                ))
                
                # Connection to next layer
                if l_idx < num_layers - 1:
                    next_layer_name = layer_list[l_idx + 1]
                    next_layer_start = None
                    for t_next in range(num_steps):
                        key_next = f"{next_layer_name}_t{t_next}"
                        if key_next in coord_map:
                            next_layer_start = coord_map[key_next]
                            break
                    
                    if next_layer_start is not None:
                        start_pt = layer_coords[-1]
                        end_pt = next_layer_start
                        fig_interactive.add_trace(go.Scatter3d(
                            x=[start_pt[0], end_pt[0]],
                            y=[start_pt[1], end_pt[1]],
                            z=[start_pt[2], end_pt[2]],
                            mode='lines',
                            line=dict(color='gray', dash='dash', width=3),
                            showlegend=False,
                            hoverinfo='skip'
                        ))

        fig_interactive.update_layout(
            title=f"Joint Layer-Time Structure: {model_name}",
            scene=dict(
                xaxis_title='Dim 1',
                yaxis_title='Dim 2',
                zaxis_title='Dim 3'
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            width=1000,
            height=800
        )
        
        html_path = results_dir / f"joint_structure_{model_name}.html"
        fig_interactive.write_html(str(html_path))
        output_paths.append(str(html_path))
        
    return output_paths