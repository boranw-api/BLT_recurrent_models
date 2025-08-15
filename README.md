# BLT Recurrent Models

> Code accompanying the CCN 2024 submission “Recurrent models optimized for face recognition exhibit representational dynamics resembling the primate brain.”

## Overview

This repository contains training code, recurrent model definitions, and analysis utilities for the BLT (Bottom-up, Lateral, Top-down) family of face-recognition networks. The codebase is organized around reproducible PyTorch experiments that compare model dynamics to primate inferior temporal (IT) cortex responses.

* Implements a configurable suite of BLT recurrent convolutional networks alongside CORnet and ResNet baselines.
* Supports multi-GPU distributed training with mixed objectives (classification vs. contrastive).
* Provides notebooks and analysis scripts for representational similarity analysis (RSA), feature visualization, and temporal tuning studies.

If you are stepping into the project for the first time, start with the quick-start section below and then skim the file guide to understand where to look for training, modeling, or analysis logic.

## Quick start

### Create an environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm pandas pillow matplotlib scikit-learn rsatoolbox antialiased-cnns wandb
```

> **Note:** `analyze_representations.py` additionally depends on `repsim` (for Angular CKA metrics) and other scientific Python packages. Install them as needed for your analysis workflow.

### Train a model

```bash
python main.py \
	--model blt_bl \
	--dataset imagenet \
	--data_path /path/to/imagenet \
	--output_path ./results/ \
	--epochs 90 \
	--batch_size 64 \
	--distributed 1
```

Key flags:

* `--model` switches among BLT variants as well as CORnet and ResNet baselines.
* `--pool` selects the pooling operator used in recurrent blocks (`max`, `average`, or `blur`).
* `--objective` chooses between standard classification and contrastive setups.
* `--data_path` must point to an ImageNet- or VGGFace2-style directory tree.

The script auto-detects the number of GPUs and launches distributed data-parallel training. To run on a single device, pass `--distributed 0`.

### Resume or evaluate

```bash
python main.py --resume path/to/checkpoint.pth --evaluate --distributed 0
```

Running with `--evaluate` skips further optimization and reports validation accuracy and loss. Set `--save_model 1` during training to persist the best-performing checkpoint under `output_path`.

## Directory layout 🗂️

```text
BLT_recurrent_models/
├── .gitignore
├── LICENSE
├── README.md
├── IT_response.png
├── analyze_representations.py
├── blt_tuning_dynamics.ipynb
├── engine.py
├── face_patch_tuning.ipynb
├── fast_models.ipynb
├── main.py
├── run_model.ipynb
├── tikz_visualizer.py
├── tuning_dynamics.ipynb
├── tuning_dynamics_second_version.ipynb
├── utils.py
├── visualize_model.ipynb
├── datasets/
│   ├── __init__.py
│   ├── datasets.py
│   ├── vggface2.py
│   └── vggface2_old.py
├── figures/
│   └── __init__.py
└── models/
		├── __init__.py
		├── activations.py
		├── blt.py
		├── build_model.py
		├── cornet.py
		└── ResNet.py
```

## File-by-file guide

### Root scripts and assets

* `.gitignore` – Standard Git exclusions for checkpoints, logs, and Python artifacts.
* `main.py` – Entry point for training and evaluation. Parses all CLI arguments, spawns distributed workers, builds models, and coordinates optimization.
* `engine.py` – Houses the core training loop (`train_one_epoch`) and validation routine (`evaluate`) used by `main.py`.
* `utils.py` – Utility helpers for distributed setup, metric logging, tensor collation, and general PyTorch niceties (adapted from torchvision references).
* `analyze_representations.py` – Toolkit for representational similarity analyses (RSA, Angular CKA) and dataset sampling utilities for evaluation studies.
* `tikz_visualizer.py` – Generates TikZ diagrams describing BLT connectivity graphs for publication-quality figures.
* `IT_response.png` – Reference figure illustrating inferior temporal cortex response dynamics used in documentation and presentations.

### Experiment notebooks

* `blt_tuning_dynamics.ipynb` – Investigates how recurrent steps shape BLT unit tuning curves.
* `tuning_dynamics.ipynb` & `tuning_dynamics_second_version.ipynb` – Alternative explorations of temporal dynamics across BLT variants.
* `face_patch_tuning.ipynb` – Focused analysis on face-selective patches and their response characteristics.
* `fast_models.ipynb` – Prototyping notebook for building and benchmarking lightweight recurrent configurations.
* `run_model.ipynb` – Interactive playground for loading checkpoints, running inference, and inspecting outputs.
* `visualize_model.ipynb` – Demonstrates how to hook layers, capture activations, and visualize network pathways.

### Datasets package (`datasets/`)

* `datasets.py` – Factory for fetching ImageNet, VGGFace2, hybrid (ImageNet + VGGFace2), and specialized evaluation splits. Handles distributed samplers and augmentation pipelines.
* `vggface2.py` – Modern PyTorch dataset wrapper for VGGFace2 with identity remapping, cropping, and optional class subset selection.
* `vggface2_old.py` – Legacy loader retained for reproducibility with earlier experiments.
* `__init__.py` – Exposes dataset constructors when importing the package.

### Models package (`models/`)

* `build_model.py` – Central dispatcher that instantiates BLT, CORnet, or ResNet architectures based on CLI flags.
* `blt.py` – Definition of the BLT recurrent network, including configurable bottom-up, lateral, and top-down connections plus pooling choices.
* `activations.py` – Convenience functions (e.g., `get_activations_batch`) for capturing intermediate features used in analyses.
* `cornet.py` – Implementations of CORnet baselines (Z/S/R/RT variants) for comparison to BLT models.
* `ResNet.py` – Thin wrapper exposing a ResNet baseline aligned with the rest of the training pipeline.
* `__init__.py` – Enables `models` to be imported as a Python package.

### Figures package (`figures/`)

* `__init__.py` – Namespace placeholder for figure-generation utilities (future-ready for scripted figure exports).

## Research workflow tips

1. **Configure data paths:** Update `--data_path` to point to ImageNet or VGGFace2 directories. For hybrid datasets, the loaders expect both datasets to exist under the same root.
2. **Monitor training:** Enable Weights & Biases logging with `--wandb_p project_name` to stream metrics online (`WANDB_MODE` toggles automatically).
3. **Probe representations:** Use `analyze_representations.py` or the notebooks to compute RSA/CKA against neural data and visualize temporal dynamics.
4. **Visualize connectivity:** Run `tikz_visualizer.py` from a notebook to export TikZ diagrams that document the learned recurrent graph.

## Additional resources

* CCN 2024 poster: <https://drive.google.com/file/d/1VUVOf9AJIQbDwfZTyAOccBW8jXGQV_xv/view?usp=sharing>
* Conference paper: <https://2024.ccneuro.org/pdf/505_Paper_authored_CCN_2024_final_with_authors.pdf>

## License

This work is distributed under the terms of the [GNU General Public License v3.0](LICENSE).