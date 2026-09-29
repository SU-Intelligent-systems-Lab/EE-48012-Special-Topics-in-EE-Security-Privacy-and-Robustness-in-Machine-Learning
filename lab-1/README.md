# Lab 1 — Training Deep Classifiers and Looking Inside Them

Code for the Lab 1 handout: MLP / CNN / ResNet-18 on MNIST / CIFAR-10 / CIFAR-100
with SGD, SGD+Momentum and Adam(W) under constant, step and poly schedules (Part 1),
then Grad-CAM (Part 2) and t-SNE (Part 3) on the CIFAR-10 CNN.

```text
configs/    one YAML per run: <dataset>_<optimizer>_<scheduler>.yaml (_lr<value>: Exercise 1.4)
models/     mlp.py, cnn.py, resnet_cifar.py
train.py    single training entry point
analysis/   plots.py (Part 1), gradcam.py (Part 2), tsne.py (Part 3)
results/    CSV logs, final metrics, checkpoints and figures
```

## Step 0 — Setup

```bash
pip install -r requirements.txt
```

Run every command from this folder. Datasets are downloaded to `data/`; a GPU is used
when available. Each run writes `results/<run>/`:

- `config.yaml`: every hyperparameter of the run
- `log.csv`: per epoch training loss/accuracy, validation loss/accuracy, learning rate, wall-clock time
- `lr.csv`: the learning rate of every iteration
- `metrics.json`: final training, validation, top-1 and top-5 test accuracy
- `epoch_<e>.pth`: checkpoints at epochs 1, 0.25E, 0.5E and E

Figures and tables go to `results/figures/`.

## Part 1 — Training Experiments

**Step 1 — Exercise 1.1: pipeline and Task A (MLP / MNIST).**
`train.py` is the single configurable script (1.1a). Check the poly schedule
against `PolynomialLR` (1.1b, `poly_check.png`) and run the 9-configuration grid (1.1c):

```bash
python -m analysis.plots --dataset mnist
for f in configs/mnist_{sgd,momentum,adam}_{constant,step,poly}.yaml; do python train.py --config $f; done
```

**Step 2 — Exercise 1.2: Task B (CNN / CIFAR-10).**

```bash
for f in configs/cifar10_{sgd,momentum,adam}_{constant,step,poly}.yaml; do python train.py --config $f; done
```

**Step 3 — Exercise 1.3: Task C (ResNet-18 / CIFAR-100).** Top-1 and top-5 test
accuracy are in `metrics.json`. ResNet-18 is written from scratch in
`models/resnet_cifar.py` (3x3 stride-1 stem, no max-pool); `torchvision.models.resnet18`
is not used. With limited compute you may set `epochs: 60` in all nine YAMLs.

```bash
for f in configs/cifar100_{sgd,momentum,adam}_{constant,step,poly}.yaml; do python train.py --config $f; done
```

**Step 4 — Exercise 1.4: learning-rate sensitivity (Task B, step schedule).**

```bash
for f in configs/cifar10_*_step_lr*.yaml; do python train.py --config $f; done
```

**Step 5 — Figures and tables.** Optionally, for several seeds copy a YAML to
`<run>_s1.yaml`, `<run>_s2.yaml`, ... with a different `seed:`; the tables then show
mean ± std and the curves the mean. For each dataset:

```bash
python -m analysis.plots --dataset mnist      # also cifar10, cifar100
```

| Required output | File |
| --- | --- |
| 3x3 test-accuracy table, best configuration marked (top-5 for CIFAR-100) | `table_<dataset>.md` |
| Learning curves: one panel per optimiser, colour by scheduler | `curves_<dataset>.png` |
| Logged learning rate vs. iteration | `lr_<dataset>.png` |
| Generalisation gap (Task C) | `gap_cifar100.png` |
| Sensitivity plot (Exercise 1.4) | `sensitivity_cifar10.png` |

Then answer Q1–Q6. For Q6, copy a CIFAR-100 YAML under a new name with
`warmup_epochs: 0` and train it.

## Part 2 — Grad-CAM Analysis (CIFAR-10 CNN)

**Step 6 — Exercises 2.1–2.5.** Pass the best CIFAR-10 configuration from
`table_cifar10.md` (all CIFAR-10 grid runs must be finished for Exercise 2.4):

```bash
python -m analysis.gradcam --config configs/cifar10_<optimizer>_<scheduler>.yaml
```

| Exercise | Output |
| --- | --- |
| 2.1 Grad-CAM vs. CAM (must be < 1e-5) | printed |
| 2.2 correct vs. misclassified (true and predicted class) | `gradcam_correct_<run>.png`, `gradcam_misclassified_<run>.png` |
| 2.3 last vs. second block, bilinear vs. nearest | `gradcam_layers_<run>.png` |
| 2.4 optimiser comparison, Pearson correlation | printed, `gradcam_optimizers_cifar10.png` |
| 2.5 cascading randomisation, Spearman correlation (optional) | printed, `gradcam_sanity_<run>.png` |

Then answer Q7–Q10.

## Part 3 — t-SNE Latent Analysis (CIFAR-10 CNN)

**Step 7 — Exercises 3.1–3.3.** Same configuration as Step 6:

```bash
python -m analysis.tsne --config configs/cifar10_<optimizer>_<scheduler>.yaml
```

| Exercise | Output |
| --- | --- |
| 3.1 final model at perplexity 30, 5, 50 | `tsne_final_<run>.png` |
| 3.2 epochs 1, 0.25E, 0.5E, E; raw-pixel and random-network baselines | `tsne_evolution_<run>.png`, `tsne_baselines_<run>.png` |
| 3.3 k-NN accuracy and silhouette on z, best vs. worst configuration (validation accuracy) | `separability_cifar10.md`, `tsne_best_worst_cifar10.png` |

Then answer Q11–Q15.
