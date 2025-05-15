# MUSE: A Multi-slice Joint Analysis Method for Spatial Transcriptomics

This repository provides the official implementation of **MUSE**, a computational framework for **multi-slice joint embedding**, **spatial domain identification**, and **gene expression imputation** in **spatial transcriptomics** (ST) experiments. MUSE leverages **optimal transport (OT)** to align cells across slices while maintaining spatial consistency, and incorporates **alignment loss and virtual neighbors** to enhance downstream analysis.

---

## 🔧 Key Features

- **Cross-slice alignment** using **fused Gromov-Wasserstein OT** (`alignment.py`)
- **Alignment-aware optimization** with optional transfer of high-quality signals
- **Virtual neighbor augmentation** for improved gene imputation and domain discovery
- Seamless integration with existing methods (e.g., STAGATE, GraphST)
- Evaluation metrics for alignment accuracy and clustering performance (`evaluate.py`)

---

## 📁 Repository Structure

```bash
MUSE/
│
├── alignment.py         # Core functions for optimal transport-based slice alignment
├── evaluate.py          # Evaluation tools for alignment, clustering, and imputation
├── utils.py             # Utilities for masking, clustering, seeding, and data prep
