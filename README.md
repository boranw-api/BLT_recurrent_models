# BLT Recurrent Models: Analysis & Visualization

This repository contains the analysis and visualization suite for the BLT (Bottom-up, Lateral, Top-down) family of face-recognition networks. It focuses on characterizing the representational dynamics of recurrent models and comparing them to primate inferior temporal (IT) cortex.

**Note:** The original model training code (`main.py`) has been moved to the `archive/` directory. The root directory is now streamlined for analyzing pre-trained models.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-lab/BLT_recurrent_models.git
    cd BLT_recurrent_models
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
    *Note: The `requirements.txt` includes a git dependency for `pytikz`.*

## Usage

The primary entry point for analysis is `rnn_test.py`. This script supports various analyses including Representational Dissimilarity Matrices (RDMs), Multi-Dimensional Scaling (MDS) trajectories, and Dynamic Stability Analysis (DSA).

### 1. Representational Trajectories (MDS)

Visualize how the model's representations evolve over recurrent time steps.

**Single MDS Space (Shared):**
Project all layers and time steps into a common low-dimensional space to compare them directly.
```bash
python rnn_test.py \
    --model-path "path/to/model.pt" \
    --mds-type single \
    --plot-dim 3
```

**Separate MDS Spaces:**
Visualize the trajectory of each layer in its own independent MDS space.
```bash
python rnn_test.py \
    --model-path "path/to/model.pt" \
    --mds-type multiple
```

**Joint Structure (Interactive 3D):**
Visualize the entire model hierarchy as a unified 3D structure, connecting layers sequentially.
```bash
python rnn_test.py \
    --model-path "path/to/model.pt" \
    --mds-type joint_structure \
    --split-by-label  # Optional: visualize separate trajectories for different classes
```
*   **[Read more about Joint Structure Plotting](readmes/joint_structure.md)**

### 2. Dynamic Stability Analysis (DSA)

Analyze the dynamical stability of the recurrent representations using Hankel matrices and DMD-like techniques.

```bash
python rnn_test.py \
    --model-path "path/to/model.pt" \
    --dsa-save-path ./results/dsa_analysis.png \
    --dsa-n-delays 3
```
*   **[Read more about DSA](readmes/DSA.md)**

### 3. RDM Analysis

Compute and plot RDMs to quantify representational geometry.
```bash
python rnn_test.py --plot-rdm-timesteps
```
*   **[Read more about RDM of RDMs](readmes/rdm_of_rdm.md)**

## Directory Layout

```text
BLT_recurrent_models/
├── archive/                  # Archived training code (main.py, utils.py) and old notebooks
├── blt_local_cache/          # Default storage for downloaded datasets/models
├── datasets/                 # Dataset loaders (VGGFace2, etc.)
├── models/                   # Model definitions (BLT, CORnet, etc.)
├── readmes/                  # Detailed method documentation
├── results/                  # Generated plots and analysis outputs
├── analyze_representations.py # Core analysis logic (RSA, CKA)
├── dsa_standard.py           # Dynamic Stability Analysis implementation
├── engine.py                 # (Legacy) Training engine components
├── geometry_path.py          # Plotting and geometry analysis plotting functions
└── rnn_test.py               # Main CLI for running analyses
```

## Historic / Training Code

The training components (`main.py`, `utils.py`, `blt_tuning_dynamics.ipynb`, etc.) have been moved to the `archive/` folder to clean up the workspace.
To train a new model, you may need to restore these files to the root or adjust imports to reference them from `archive/`.

**Original Training Example:**
```bash
# (Requires main.py in root)
python main.py --model blt_bl --dataset imagenet --epochs 90
```

More often used command:
```bash
/home/savannah/anaconda3/envs/blt/bin/python rnn_test.py --plot-rdm-timesteps --skip-dsa --test-batch-size 10 --batch-size 10
```