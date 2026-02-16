# BLT Recurrent Models

This repository contains the analysis and visualization suite for the BLT (Bottom-up, Lateral, Top-down) family of face-recognition networks. It focuses on characterizing the representational dynamics of recurrent models and comparing the last two layers of them to primate inferior temporal (IT) cortex.

**Note:** The original model training code (`main.py`) has been moved to the `archive/` directory. The root directory is now streamlined for analyzing pre-trained models.

## Usage

The primary entry point for analysis is `rnn_test.py`.

### 1. Representational Trajectories (MDS)

Visualize how the model's representations evolve over recurrent time steps.

**Joint Structure (3D):**
Project all layers and time steps into a common low-dimensional space to compare them directly.
```bash
python rnn_test.py \
    --model-path "path/to/model.pt" \
    --mds-type joint_structure_3d
```

**Separate MDS Spaces:**
Visualize the trajectory of each layer in its own independent MDS space.
```bash
python rnn_test.py \
    --model-path "path/to/model.pt" \
    --mds-type multiple
```

**Joint Structure:**
Generate a 2D MDS plot for each layer, with each layer connected sequentially.
```bash
python rnn_test.py \
    --model-path "path/to/model.pt" \
    --mds-type joint_structure \
    --split-by-label  # if chosen, would generate two MDS plots, where one is object only and the other is face only
```

### 2. RDM Analysis

Compute and plot RDMs to quantify representational geometry (RDM-of-RDMs).
```bash
python rnn_test.py \
    --model-path "path/to/model.pt" \
    --plot-rdm-timesteps
```
*   **[Read more about RDM of RDMs](readmes/rdm_of_rdm.md)**

### 3. Combined Analysis

You can run multiple analyses in a single command. For example, to generate both the interactive joint structure and the RDM analysis:

```bash
python rnn_test.py \
    --model-path "path/to/model.pt" \
    --mds-type joint_structure \
    --plot-rdm-timesteps
```

All generated plots will be saved in the `results/` directory under their respective subfolders (`3D/`, `Joint_Structure/`, `RDM_Timesteps/`).

### 4. Dataset Information (Amir Dataset)

The dataset used (`blt_local_cache/face_object_dataset.pkl`) contains 4,800 images balanced between faces and objects.

| Property | Value |
| :--- | :--- |
| **Total Images** | 4,800 |
| **Classes** | Faces (2,400), Objects (2,400) |
| **Dimensions** | 512 x 512 px (RGB) |
| **Organization** | Grouped by class (All Faces first, then all Objects) |
| **Data Type** | PIL Images |

**Important Note:** The loader uses `random_split`, meaning the test set is a **random shuffle** of the original data. Therefore, even though the source file is grouped, the test loader yields mixed batches. 

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

├── engine.py                 # (Legacy) Training engine components
├── geometry_path.py          # Plotting and geometry analysis plotting functions
└── rnn_test.py               # Main CLI for running analyses
```