# 🎓 Uncertainty Quantification in Deep Learning - Master Thesis Repository

Welcome to the official code and resource repository for the master's thesis:

**Title:** *Uncertainty Quantification in Deep Learning*

**Author:** Nathaniel Cogneaux

**Institutions:** University Paris Sciences & Lettres (France) & University of Padua (Italy)

**Research Host:** Hanyang University, Seoul, South Korea

**Year:** 2024

This repository accompanies the thesis, which introduces a **novel post-hoc uncertainty quantification (UQ)** technique that is **efficient**, **model-agnostic**, works in a **single-pass** and applicable to **pre-trained deep neural networks** without retraining.


## Repository Contents

```
Master_Thesis_UQ/
│
├── dataloaders_and_metrics/        ← Data loaders and UQ metrics
│   ├── dataloaders.py              ← All dataset handling logic
│   └── metrics.py                  ← Metric definitions (ECE, NLL, etc.)
│
├── examples/                       ← Interactive usage examples
│   ├── Lenet.ipynb
│   ├── Wideresnet_28_10_cifar10.ipynb
│   └── Wideresnet_28_10_cifar100.ipynb
│
├── models/                         ← Model definitions
│   ├── lenet.py
│   ├── resnet.py
│   ├── vggnet.py
│   └── wide_resnet.py
│
├── multi_output_module/
│   └── multi_output_module.py     ← Our proposed meta-model for UQ
│
├── numerical_experiments/
│   ├── numerical_experiments.py   ← Script to reproduce main experiments
│   └── wandb_module_hyperparameters_tuning.py
│
├── Master_s_thesis_Nathaniel_Cogneaux.pdf   ← Full Thesis Document
└── dissertation_defense.pdf                ← Slides of the Defense Presentation
```


## Project Highlights

✅ **Post-Hoc UQ**
A simple, lightweight wrapper that estimates uncertainty without retraining.

✅ **Model-Agnostic**
Compatible with any pre-trained model (LeNet, VGG, ResNet, WideResNet, etc.).

✅ **Efficient & Scalable**
Achieves near SoTA results with drastically lower computational cost.

✅ **Ready-to-Use Notebooks**
Step-by-step usage on MNIST, CIFAR-10, CIFAR-100.

✅ **Benchmark Results**
Includes standard and corrupted datasets (CIFAR-10-C, CIFAR-100-C).

✅ **Thesis & Defense Slides**
Theory fully explained and experimental insights.


## Thesis Summary

Modern deep neural networks (DNNs) are powerful but often **overconfident**, especially when they’re wrong. In safety-critical systems, this is unacceptable.

> This work proposes a **post-hoc meta-model** that can be added on top of any pre-trained model to provide meaningful uncertainty estimates **in a single forward pass**, without requiring retraining or architectural knowledge.

Results demonstrate robust uncertainty quantification on:

* **MNIST**, **FashionMNIST**
* **CIFAR-10**, **CIFAR-100**
* Corrupted datasets: **CIFAR-10-C**, **CIFAR-100-C**

For a deep dive, please give a look at:

* [`Master_s_thesis_Nathaniel_Cogneaux.pdf`](./Master_s_thesis_Nathaniel_Cogneaux.pdf), Full thesis (theory, method, results)
* [`dissertation_defense.pdf`](./dissertation_defense.pdf), Defense presentation slides


## Quick Start

1. **Clone the repo**

   ```bash
   git clone https://github.com/your_username/Master_Thesis_UQ.git
   cd Master_Thesis_UQ
   ```

2. **Create an environment**

   ```bash
   python -m venv uq_env
   source uq_env/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Try a notebook**
   Launch Jupyter and open any notebook in `examples/`.


## Reproducing Results

Run the main experiment script to regenerate results from the thesis:

```bash
python numerical_experiments/numerical_experiments.py
```

Or tune hyperparameters using:

```bash
python numerical_experiments/wandb_module_hyperparameters_tuning.py
```

All training/evaluation logic is managed through `dataloaders_and_metrics/` and `multi_output_module/`.


## 📘 Citation

If you use this work in your research, please cite:

```
@mastersthesis{cogneaux2024masteruq,
  author       = {Nathaniel Cogneaux},
  title        = {Uncertainty Quantification in Deep Learning},
  school       = {University Paris Sciences & Lettres and University of Padua},
  year         = {2024},
  note         = {\url{https://github.com/NathanielCogneaux/Master_Thesis_UQ}},
}

```


## Contact

For questions or collaborations:

* [Open a GitHub Issue](https://github.com/NathanielCogneaux/Master_Thesis_UQ/issues)
* [LinkedIn Profile](https://www.linkedin.com/in/nathaniel-cogneaux/)
