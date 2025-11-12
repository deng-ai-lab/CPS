# ✨ Constrained Particle Seeking [AAAI 2026] 

Official implementation for: **"Constrained Particle Seeking: Solving Diffusion Inverse Problems with Just Forward Passes"**

<div style="text-align: center;">
  <img src="assets\graph_00.png" width="550" alt="Graph showing Constrained Particle Seeking high-level diagram">
</div>

---

## 📄 Abstract

We introduce **Constrained Particle Seeking (CPS)**, a novel, **gradient-free** approach designed to solve diffusion inverse problems. CPS leverages information from *all* candidate particles of the diffusion model to actively search for the optimal particle.

A key feature of CPS is the efficient incorporation of **constraints** that align the search process with high-density regions of the unconditional prior. This results in a powerful method that achieves high performance while relying solely on **forward passes** of the pre-trained diffusion model and measurement operators, circumventing the need for backpropagation or  gradient estimation.

<div style="text-align: center;">
  <img src="assets\method3_00.png" width="500" alt="Diagram illustrating the Constrained Particle Seeking method workflow">
</div>

---

## 🚀 Getting Started

### 1. Dependencies

Ensure you have the following prerequisites installed:

* Python **(3.8+)**
* PyTorch **(2.1.0)**

### 2. Installation

Clone the repository and install the required packages:

```bash
git clone [YOUR_REPO_LINK_HERE] # Replace with your actual repo link
cd constrained-particle-seeking
pip install -r requirements.txt
````

### 3\. Data and Checkpoints

All necessary data and checkpoints should be placed in the root-level `/data` and `/checkpoints` directories, respectively.

  * **Datasets:**

      * **FFHQ (256x256):** Download the dataset from [Kaggle Link](https://www.kaggle.com/datasets/denislukovnikov/ffhq256-images-only).
      * **InverseBench:** Download the dataset from [Caltech Data Link](https://data.caltech.edu/records/jfdr4-6ws87?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjdiNDk4OGU3LWQ0NTgtNGYwNy04NDc4LWE5YWE3OWIzOTU0MSIsImRhdGEiOnt9LCJyYW5kb20iOiJlYTk1ZjU0YTdmZjcwZTQ1OTYzZTNiZTRkNTBhYmJmMiJ9.NFEYlpOyrepCIFkR6EBrVaQcGGfVam5gileyMjbnrjBCZFemXLsGyGY-qlxlPf9tGE_L1qH3lCpUJz_RTeOfiQ).

  * **Checkpoints:**

      * `ffhq256.pt`: Converted from the [Diffusion Posterior Sampling (DPS) repository](https://github.com/DPS2022/diffusion-posterior-sampling).
      * `blackhole-50k.pt` and `ns-5m.pt`: Converted from the [InverseBench repository releases](https://github.com/devzhk/InverseBench/releases).

-----

## 💡 Usage

To run an experiment, use the `main.py` script and specify the configuration for the problem, algorithm, and pre-trained model checkpoint.

The example below runs the **Constrained Particle Seeking** algorithm with the **FFHQ** inpainting setup:

```bash
python3 main.py problem=ffhq256_inpaint algorithm=aps_r pretrain=ffhq256
```

> **Note:** The configuration files in the `/configs` directory define the exact parameters for each `problem`, `algorithm`, and `pretrain` setting.

-----

## 📁 Code Structure

The repository is organized as follows:

  * `/data`: **Datasets** used for training and evaluation.
  * `/algo`: Implementation of the **Constrained Particle Seeking (CPS)** and baseline algorithms.
  * `/checkpoints`: Pre-trained **model weights** and diffusion checkpoints.
  * `main.py`: The **main execution script** for running experiments.
  * `configs/`: **Configuration files** (using Hydra) defining experiment parameters.


