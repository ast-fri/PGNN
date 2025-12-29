# Proper Orthogonal Decomposition for Scalable Training of Graph Neural Networks (PGNN)

This repository provides the official implementation of **PGNN**, a POD-based sketching framework that trains GNNs efficiently on large graphs by eliminating online sketch updates, achieving strong performance at much lower memory and computation cost.  

---

## Table of Contents
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Running the Project](#running-the-project)
- [Configuration Details](#configuration-details)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

---

## Requirements

- **Python**: ≥ 3.9  
- **CUDA**: Recommended for GPU training  
- **PyTorch**: CUDA-enabled build recommended  
- Complete dependency list available in `environment.yml` or `requirements.txt`

> **Note**: Conda is recommended for consistent dependency management.

---

## Getting Started

You can set up the environment using either **conda (recommended)** or **pip**.

---

### Option 1: Using Conda (Recommended)

#### Step 1: Create the Environment
```bash
conda env create -f environment.yml
```
Step 2: Activate the Environment
```bash
conda activate pgnn
```

The environment name is defined inside environment.yml.

Option 2: Using pip and Virtual Environment
Step 1: Create a Virtual Environment
  ```bash
python3 -m venv env
source env/bin/activate   # On Windows: .\env\Scripts\activate
```
Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```
Verify Installation (Optional)
```bash
python --version
python -c "import torch; print(torch.__version__)"
```
## Running the Project

We provide two ways to run experiments.

### Option 1: Using main_runner.py

This is the most flexible way to run experiments with custom configurations.
```bash
python main_runner.py <dataset> <arch> <lr> <l2> <ratio> <use_rank_approx> <layers> <hidden_dim> <dropout> <layer_norm> <ratio2> <order> <device> <lr_update_weight> <random_sample> <epochs> <runs> <seed>

Example
python main_runner.py cora GCN 0.01 0.0005 0.02 False 3 128 0 0 0.7 2 0 0.01 2 200 5 2
```
#### Configuration Details

| Argument | Description |
|---------|-------------|
| `dataset` | Dataset name (e.g., cora, citeseer, pubmed) |
| `arch` | GNN architecture (GCN, GAT, etc.) |
| `lr` | Learning rate |
| `l2` | L2 regularization weight |
| `ratio` | Sketching ratio |
| `use_rank_approx` | `True` to enable count-sketch approximation |
| `layers` | Number of GNN layers |
| `hidden_dim` | Hidden dimension size |
| `dropout` | Dropout rate |
| `layer_norm` | `1` to enable layer normalization |
| `ratio2` | Count-sketch ratio (not implemented) |
| `order` | Order of the polynomial activation |
| `device` | CUDA device ID or `cpu` |
| `lr_update_weight` | Learning-rate scaling for updating the ρ₁ matrix |
| `random_sample` | `1` to enable random sampling |
| `epochs` | Number of training epochs |
| `runs` | Number of repeated runs |
| `seed` | Random seed |


### Option 2: Using Dataset-Specific Shell Scripts

Predefined scripts are available for convenience.
```bash
cd shell_scripts
bash run_cora.sh
```

You can modify these scripts to change datasets or hyperparameters.

## Acknowledgements

Parts of this codebase are adapted from the following work:

- **Sketch-GNN-Sublinear** GitHub repository: https://github.com/johnding1996/Sketch-GNN-Sublinear  

- Mucong Ding et al., *Sketch-GNN: Scalable Graph Neural Networks with Sublinear Complexity*

We sincerely thank the authors for making their code publicly available.


## License

This project is licensed under the CC-BY-NC License - see LICENSE file for complete details.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{PGNN-2025,
  title={Proper Orthogonal Decomposition for Scalable Training of Graph Neural Networks},
  author={[Abhishek A, Manohar Chandran, Mohit Meena and Mahesh Chandran]},
  journal={arXiv preprint},
  year={2025}
}
```

## Contact

For questions, issues, or collaborations:

Open a GitHub issue

Or contact the research team directly

---

© 2025 Research Project | Proper Orthogonal Decomposition for Scalable GNNs
