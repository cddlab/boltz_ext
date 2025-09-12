# Boltz Extension with Restraint-Guided Inference

This repository provides an extended version of Boltz-1 with **restraint-guided inference** to improve stereochemical accuracy in protein-ligand complex structure prediction. This method addresses significant limitations in ligand stereochemistry reproduction, including chirality, bond lengths, and bond angles, without requiring model retraining.

## 🚀 Quick Start (No Installation Required)

Try the method directly in Google Colab without any installation:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cddlab/colabfold_boltz_restr/blob/main/Boltz1.ipynb)

## 📋 Key Features

- **100% chirality reproduction** for input molecular structures
- **Significant improvement** in bond lengths and angle geometries
- **No model retraining required** - works with existing Boltz-1 weights
- **GPU acceleration** for restraint calculations
- **Maintains protein structure quality** while fixing ligand stereochemistry

## 🛠️ Installation

### Prerequisites

- Python 3.11+
- PyTorch 2.2.0+
- CUDA-compatible GPU (recommended for performance)

### Step 1: Install torch-cluster

First, install `torch-cluster` with the appropriate CUDA version. For PyTorch 2.8.0:

```bash
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.8.0+${CUDA}.html
```
Replace `${CUDA}` with your CUDA version string (e.g., `cu121` for CUDA 12.1, `cu118` for CUDA 11.8, or `cpu` for CPU-only installation).

Examples:
```bash
# For PyTorch 2.8.0 and CUDA 12.6
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
```

### Step 2: Clone and Install Boltz Extension

```bash
git clone https://github.com/cddlab/boltz_ext.git
cd boltz_ext
git checkout restr_torch
pip install -e .
```

## ⚙️ Configuration

### Basic Usage

To enable restraint-guided inference, modify your configuration YAML file:

#### 1. Enable Chiral Restraints for Ligands

Add `chiral_restraints: true` at the same level as your ligand CCD code or SMILES:

```yaml
sequences:
  - protein:
      id: A
      sequence: "MKFLVL..."
  - ligand:
      ccd: "ATP"  # or smiles: "CC(C)CC..."
      chiral_restraints: true  # Add this line
```

#### 2. Configure Restraint Parameters

Add a top-level `restraints_config` section:

```yaml
restraints_config:
  angle:
    weight: 1      # Weight for bond angle restraints
  bond:
    weight: 1      # Weight for bond length restraints  
  chiral:
    weight: 1      # Weight for chirality restraints
  start_sigma: 1.0 # Noise level threshold for applying restraints
  gpu: true        # Enable GPU acceleration
```

### Complete Configuration Example

```yaml
# Sample configuration with restraint-guided inference
sequences:
  - protein:
      id: A
      sequence: "MKFLVLVLLAIIWLLLPSGGAGARGDFPGTYVEYIHYQVWAISPGDKAWRLAKKDQAEVKLREYRKHLA"
  - ligand:
      ccd: "ATP"
      chiral_restraints: true

restraints_config:
  angle:
    weight: 1
  bond:
    weight: 1
  chiral:
    weight: 1
  start_sigma: 1.0
  gpu: true
```

### Configuration Options

#### Parameters

- **`weight`**: Relative weight for each restraint type (default: 1)
- **`start_sigma`**: Sigma threshold below which restraints are applied (default: 1.0)
- **`gpu`**: Enable GPU-accelerated constraint calculations (default: false)
  - Highly recommended for large ligands or multiple diffusion samples

#### Restraint Combinations

You can use different combinations of restraints:
- All restraints (Boltz R in paper)  
```yaml
restraints_config:
  angle:
    weight: 1
  bond:
    weight: 1
  chiral:
    weight: 1
  start_sigma: 1.0
```

- Chirality only (Boltz Rc in paper)
```yaml
restraints_config:
  chiral:
    weight: 1
  bond:
    weight: 0
  chiral:
    weight: 0
  start_sigma: 1.0
```

- Final step only (Boltz R1 in paper)
```yaml
restraints_config:
  angle:
    weight: 1
  bond:
    weight: 1  
  chiral:
    weight: 1
  start_sigma: 0.005
```

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{ishitani2025improving,
  title={Improving Stereochemical Limitations in Protein-Ligand Complex Structure Prediction},
  author={Ishitani, Ryuichiro and Moriwaki, Yoshitaka},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.03.25.645362v2}
}
```

---

<h1 align="center">Boltz-1:

Democratizing Biomolecular Interaction Modeling
</h1>

![](docs/boltz1_pred_figure.png)

Boltz-1 is the state-of-the-art open-source model that predicts the 3D structure of proteins, RNA, DNA, and small molecules; it handles modified residues, covalent ligands and glycans, as well as condition the generation on pocket residues. 

For more information about the model, see our [technical report](https://doi.org/10.1101/2024.11.19.624167).

## Installation
Install boltz with PyPI (recommended):

```
pip install boltz -U
```

or directly from GitHub for daily updates:

```
git clone https://github.com/jwohlwend/boltz.git
cd boltz; pip install -e .
```
> Note: we recommend installing boltz in a fresh python environment

## Inference

You can run inference using Boltz-1 with:

```
boltz predict input_path --use_msa_server
```

Boltz currently accepts three input formats:

1. Fasta file, for most use cases

2. A comprehensive YAML schema, for more complex use cases

3. A directory containing files of the above formats, for batched processing

To see all available options: `boltz predict --help` and for more information on these input formats, see our [prediction instructions](docs/prediction.md).

## Evaluation

To encourage reproducibility and facilitate comparison with other models, we provide the evaluation scripts and predictions for Boltz-1, Chai-1 and AlphaFold3 on our test benchmark dataset as well as CASP15. These datasets are created to contain biomolecules different from the training data and to benchmark the performance of these models we run them with the same input MSAs and same number  of recycling and diffusion steps. More details on these evaluations can be found in our [evaluation instructions](docs/evaluation.md).

![Test set evaluations](docs/plot_test.png)
![CASP15 set evaluations](docs/plot_casp.png)


## Training

If you're interested in retraining the model, see our [training instructions](docs/training.md).

## Contributing

We welcome external contributions and are eager to engage with the community. Connect with us on our [Slack channel](https://join.slack.com/t/boltz-community/shared_invite/zt-2w0bw6dtt-kZU4png9HUgprx9NK2xXZw) to discuss advancements, share insights, and foster collaboration around Boltz-1.

## Coming very soon

- [x] Auto-generated MSAs using MMseqs2
- [x] More examples
- [x] Support for custom paired MSA
- [x] Confidence model checkpoint
- [x] Chunking for lower memory usage
- [x] Pocket conditioning support
- [x] Full data processing pipeline
- [ ] Colab notebook for inference
- [ ] Kernel integration

## License

Our model and code are released under MIT License, and can be freely used for both academic and commercial purposes.


## Cite

If you use this code or the models in your research, please cite the following paper:

```bibtex
@article{wohlwend2024boltz1,
  author = {Wohlwend, Jeremy and Corso, Gabriele and Passaro, Saro and Reveiz, Mateo and Leidal, Ken and Swiderski, Wojtek and Portnoi, Tally and Chinn, Itamar and Silterra, Jacob and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-1: Democratizing Biomolecular Interaction Modeling},
  year = {2024},
  doi = {10.1101/2024.11.19.624167},
  journal = {bioRxiv}
}
```

In addition if you use the automatic MSA generation, please cite:

```bibtex
@article{mirdita2022colabfold,
  title={ColabFold: making protein folding accessible to all},
  author={Mirdita, Milot and Sch{\"u}tze, Konstantin and Moriwaki, Yoshitaka and Heo, Lim and Ovchinnikov, Sergey and Steinegger, Martin},
  journal={Nature methods},
  year={2022},
}
```
