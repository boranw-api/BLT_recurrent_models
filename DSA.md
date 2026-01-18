Here is the comprehensive English documentation, adapted from your original text and enhanced with the key technical insights, best practices, and warnings we discussed regarding your specific implementation.

---

# Dynamical Similarity Analysis (DSA): Implementation Guide & Best Practices

This document serves as an onboarding guide for developers and researchers using the DSA framework within this project. It explains the conceptual basis, the specific "hybrid" dimensionality reduction pipeline implemented here, and critical guidelines for parameter tuning to avoid common pitfalls like overfitting.

## 1. What is DSA?

Dynamical Similarity Analysis (DSA) is a method used to compare **"whether the underlying dynamical laws of two systems are similar,"** rather than just comparing their geometric trajectories.

Given two time-series trajectories (e.g., hidden states of a neural network layer evolving over time), DSA performs the following:

1. Fits a linear dynamical system (a matrix ) to each trajectory such that .
2. Compares these  matrices using **Procrustes Analysis on Manifolds**, finding an optimal rotation/transformation to align them.

**Intuition:**
Two physical motions might look different in raw data space. However, if they are governed by the same differential equations (just observed from different coordinate systems), DSA will identify them as having high **dynamical similarity**.

## 2. The Core Algorithm (Simplified)

For each input trajectory:

1. **Delay Embedding:** Stack history to form a Hankel Matrix (creating "Eigen-time-delay coordinates").
2. **Implicit SVD (Spatiotemporal Reduction):** Perform SVD on the Hankel matrix to extract dominant dynamical modes.
3. **HAVOK/DMD:** Fit the linear operator  on these low-rank coordinates.
4. **Alignment:** Calculate the distance between operators  and  allowing for orthogonal transformations.

## 3. Implementation Pipeline: The "Hybrid" Approach

In this project, we analyze the similarity of internal dynamics across network layers (e.g., `output_0` vs `output_5`) or between different object categories.

Our implementation uses a **two-stage dimensionality reduction strategy** to handle high-dimensional neural data efficiently:

### Step 1: Data Extraction

* **Function:** `extract_recurrent_steps`
* **Input:** Neural network activations.
* **Output:** A list of tensors with shape `[Batch, Channels, Height, Width]`.

### Step 2: Preprocessing & Explicit Spatial Reduction

* **Function:** `perform_dsa_analysis`  `_apply_pca_to_trajectory`
* **Action:**
1. Flatten spatial dimensions: `(T, B, C, H, W)`  `(T, B, Features_Flat)`.
2. **Explicit PCA:** Apply standard PCA on the flattened features.
* *Purpose:* Spatial compression to reduce memory usage before constructing the massive Hankel matrix.
* *Shape Change:* `(T, B, 4096)`  `(T, B, 80)` (example).


3. Permute to "Trials-First" format: `(Batch, Time, Reduced_Features)`.



### Step 3: Implicit Spatiotemporal Reduction & DSA

* **Function:** `DSA` Class (Standard Implementation)
* **Action:**
1. **Hankel Construction:** Expands dimensions by `n_delays`.
* *Shape:* `(Features × n_delays)`.


2. **Implicit SVD:** Internally performs SVD on the Hankel matrix.
* *Purpose:* Denoising and extracting dynamical modes.
* *Control:* Managed by `rank` or `rank_explained_variance`.


3. **Score Calculation:** Computes the similarity matrix.



## 4. Key Interface & Parameters

**Primary Entry Point:** `perform_dsa_analysis` (in `analyze_representations.py`).

### 4.1. `pca_components` (Explicit Reduction)

Controls the spatial preprocessing step.

* **Float ():** Retain this fraction of explained variance (e.g., `0.95`). **(Recommended)**
* **Integer ():** Retain exactly  components (e.g., `80`).
* *Note:* If set too low, you lose spatial information. If set too high (or `None`), the subsequent Hankel matrix may cause Out-Of-Memory (OOM) errors.

### 4.2. `n_delays` & `delay_interval`

Controls the depth of the history embedding (HAVOK parameters).

* Typically, `n_delays` ranges from 10 to 100 depending on the complexity of the dynamics.
* **Warning:** Higher delays exponentially increase the size of the internal matrices.

### 4.3. `rank` / `rank_explained_variance` (Critical)

Controls the **Implicit SVD** truncation inside the DSA algorithm.

* **⚠️ CRITICAL WARNING:** The default is often `None` (Full Rank). **Do not use the default for neural data.**
* **Why?** Full rank preserves high-frequency noise in the Hankel matrix. The DMD will try to fit this noise, resulting in a garbage model  and meaningless similarity scores.
* **Recommendation:**
* Set `rank_explained_variance=0.95` (or 0.99).
* Or set `rank` to a hard integer (e.g., `100`).



### 4.4. `validate_fit` & `fit_metric`

Enables quality control for the dynamical models.

* **`validate_fit=True`**: Calculates how well matrix  predicts  from .
* **`fit_metric`**:
* `"nrmse"`: Normalized Root Mean Square Error (Lower is better).
* `"r2"`: Coefficient of Determination (Higher is better, close to 1.0).
* `"vaf"`: Variance Accounted For.


* *Note:* The code internally converts "higher-is-better" metrics (R2, VAF) into an "error" format () so that threshold warnings work consistently.

## 5. Typical Usage Flow

### A. Standard Layer-wise Analysis

In `rnn_test.py`:

1. Sample images and extract recurrent trajectories.
2. Call `perform_dsa_analysis` passing the trajectory dictionary.
3. Result is saved as a heatmap in `results/DSA`.

### B. Category-Split Analysis

Use `dsa_split_by_label` to compare dynamics between different input classes (e.g., "Face Dynamics" vs. "Object Dynamics").

## 6. Common Pitfalls & FAQ

### Q1: Is the output a distance or a similarity?

* The standard DSA algorithm calculates an **Angular Distance** .
* **Our Implementation:** Converts this to a similarity score for easier visualization:


* 1.0 = Identical Dynamics.
* 0.0 = Completely Orthogonal/Different.



### Q2: How is the "Batch" dimension handled?

* The batch dimension represents independent trials/conditions.
* The code correctly handles `(Batch, Time, Features)`. It does **not** concatenate batches into a single long time series. This prevents "phantom dynamics" where the end of one trial is treated as the precursor to the start of the next.

### Q3: Why am I getting "High Fit Error" warnings?

* This means the linear operator  fails to capture the trajectory dynamics ( is low).
* **Likely Causes:**
1. `rank` is too high (fitting noise).
2. `n_delays` is too small (not enough history to unfold the attractor).
3. `pca_components` is too small (spatial features destroyed before analysis).



### Q4: Why is Explicit PCA necessary if DSA does SVD?

* If your input is `(Time=100, Features=4096)` and `n_delays=10`, the Hankel matrix width is . Performing SVD on this is slow and memory-intensive.
* Explicit PCA reduces features to ~80 first, making the Hankel width a manageable .

## 7. Summary Checklist

When running DSA on new data, ensure:

* [ ] **PCA:** `pca_components` is set (e.g., `0.95` or `80`) to handle spatial complexity.
* [ ] **Rank:** `rank_explained_variance` (e.g., `0.95`) or `rank` is set to prevent overfitting noise.
* [ ] **Validation:** `validate_fit=True` is on to catch poor models.
* [ ] **Interpretation:** Remember that results are **Similarities** (1 is good), not Distances.