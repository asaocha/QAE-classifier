# Quantum autoencoders for image classification

This is the official code repository of [Quantum autoencoders for image classification](https://arxiv.org/abs/2502.15254). This project implements a quantum autoencoder (QAE) approach for classifying the MNIST dataset. It also supports baseline comparisons with classical methods such as Nonnegative Matrix Factorization (NBMF), Fully Connected Neural Networks (FCNN), and Singular Value Decomposition (SVD).

## Installation Guide

Use the following command to create the environment:
```bash
conda create -n qae-classifier python=3.10
conda activate qae-classifier
pip install qiskit==1.0.2 qiskit-aer==0.14.0.1 qiskit-algorithms==0.3.0 qiskit-machine-learning==0.7.2 pyqubo==1.4.0
conda install -y numpy matplotlib tensorflow scikit-learn seaborn keras opencv h5py psutil tqdm pylatexenc
```

## Usage

### Train QAE classification model
---

Prepare `train_parameter.json`:
```json
{
    "parameter": {
        "seed": 123,
        "label_list": [0, 1, 2, 3],
        "trash_bit_num": 3,
        "train_data_num": 500,
        "test_data_num": 500,
        "height": 16,
        "width": 16,
        "epoch": 5000,
        "ansatz_dict": {
            "ansatz_reps": 20,
            "A": "A-1",
            "A_gate": "Ry",
            "B": "B-1",
            "B_gate": "Ry"
        }
    }
}
```

Run `main.py`:
```bash
python main.py
```

#### Optional arguments:

- `--load_parameter`: Use pre-trained parameters from a folder
- `--savefolder`: Specify a directory path contains existing `train_parameter.json`
- `--skip_training`: Skip training (only do classification)
- `--load_epoch`: Load model parameters from a specific epoch

Example:

```bash
python main.py --load_parameter --savefolder output/train_result_20250323_142300 --load_epoch 100
```


### Reproduce the results in paper
---
The table shows the correspondence between the published data and the results in the paper:
| Table number | Row number |Test accuracy | Directory path |
| :---: | :---: | :---: |  :---: |  
|1| 1 | 80.6% | paper_result/train_result_20250204_103229 |
|1| 2 | 87.6% | paper_result/train_result_20250203_013109 |
|1| 3 | 90.4% | paper_result/train_result_20250203_111648 |
|1| 4 | 82.0% | paper_result/train_result_20250110_134718 |
|1| 5 | 73.6% | paper_result/train_result_20250110_134821 |
|2| 1 | 90.4% | paper_result/train_result_20250203_111648 |
|2| 2 | 49.4% | paper_result/train_result_20250208_122523 |
|2| 3 | 55.2% | paper_result/train_result_20250208_122624 |
|2| 4 | 63.6% | paper_result/train_result_20250208_122726 |
|2| 5 | 56.4% | paper_result/train_result_20250208_122827 |

All results are obtained at epoch 5000. Test accuracy can be reproduced by:
```bash
python main.py --load_parameter --savefolder PATH --skip_training --load_epoch 5000
```

### Comparison method
---

To run a baseline method (e.g., `nbmf`, `fcnn`, or `svd`), add `"method": "nbmf"` (or `"fcnn"` / `"svd"`) to `train_parameter.json`:
```json
{
    "parameter": {
        "method": "nbmf",
        "seed": 123,
        "label_list": [0, 1, 2, 3],
        "trash_bit_num": 3,
        "train_data_num": 500,
        "test_data_num": 500,
        "height": 16,
        "width": 16,
        "epoch": 5000
    }
}
```

### Calculate the expressibility of Ansatz
---
Calculations in the code is based on https://arxiv.org/abs/1905.10876, https://arxiv.org/abs/2003.09887, https://github.com/Saesun-Kim/Quantum_Machine_Learning_Express.

```bash
python -m expressibility.calc --savefolder PATH
```
#### Arguments:

- `--savefolder`: Specify a directory path contains existing `train_parameter.json` and trained parameters

Example:

```bash
python -m expressibility.calc --savefolder output/train_result_20250323_142300
```
