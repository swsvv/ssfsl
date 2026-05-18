# Self-Supervised Few-Shot Learning

Experiment repository for an empirical study of unsupervised few-shot learning that utilizes self-supervised representation learning. This framework provides a unified codebase for training and evaluating various SSL methods under the few-shot classification setting, enabling fair and reproducible comparisons across methods, encoders, and datasets.

## Supported Methods

| Method       | Paper                                                    |
| ------------ | -------------------------------------------------------- |
| SimCLR       | [Chen et al., 2020](https://arxiv.org/abs/2002.05709)    |
| MoCo         | [He et al., 2020](https://arxiv.org/abs/1911.05722)      |
| BYOL         | [Grill et al., 2020](https://arxiv.org/abs/2006.07733)   |
| SimSiam      | [Chen & He, 2021](https://arxiv.org/abs/2011.10566)      |
| Barlow-Twins | [Zbontar et al., 2021](https://arxiv.org/abs/2103.03230) |
| SwAV         | [Caron et al., 2020](https://arxiv.org/abs/2006.09882)   |
| VICReg       | [Bardes et al., 2022](https://arxiv.org/abs/2105.04906)  |
| W-MSE        | [Ermolov et al., 2021](https://arxiv.org/abs/2007.06346) |
| NNCLR        | [Dwibedi et al., 2021](https://arxiv.org/abs/2104.14548) |
| DINOv2       | [Oquab et al., 2024](https://arxiv.org/abs/2304.07193)   |

## Supported Encoders

ConvNet4, ConvNet5, ResNet10, ResNet18, ResNet34, ResNet50, ViT-Tiny, ViT-Small

## Datasets

| Dataset                                    | Source                                                                    |
| ------------------------------------------ | ------------------------------------------------------------------------- |
| miniImageNet                               | [Ravi & Larochelle, 2017](https://openreview.net/forum?id=rJY0-Kcll)      |
| tieredImageNet                             | [Ren et al., 2018](https://arxiv.org/abs/1803.00676)                      |
| CIFAR-FS                                   | [Bertinetto et al., 2019](https://arxiv.org/abs/1805.08136)               |
| FC100                                      | [Oreshkin et al., 2018](https://arxiv.org/abs/1805.10123)                 |
| CUB-200-2011                               | [Wah et al., 2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) |
| CDFSL (ChestX, CropDisease, EuroSAT, ISIC) | [Guo et al., 2020](https://arxiv.org/abs/1912.07200)                      |

## Setup

```bash
conda env create -f env.yml
conda activate ssfsl
```

## Usage

### Training

```bash
python train.py config/simclr.yaml --device=cuda:0
```

Override any config field via CLI:

```bash
python train.py config/simclr.yaml --device=cuda:0 --batch_size=128 --lr=0.01 --encoder=ResNet34
```

### Evaluation

Evaluate all checkpoints from a training run with N-way K-shot episodes:

```bash
python test.py <exp_path>/config.yaml \
    --device=cuda:0 \
    --mode=test \
    --classifier=LogisticRegression \
    --num_episode=1000 \
    --n_way=5 \
    --num_spt=5 \
    --num_qry=15
```

Or use the evaluation script to sweep over multiple shot values:

```bash
bash script/run_eval.sh <exp_path>/config.yaml cuda:0
```

## Project Structure

```
ssfsl/
├── train.py              # Training entry point
├── test.py               # Evaluation entry point
├── config.py             # Config classes (expedantic)
├── interface.py          # Base model interfaces (Sup, UnSup)
├── dataloader.py         # DataLoader construction
├── logger.py             # CSV + console logging
├── validation.py         # Validation (loss & accuracy)
├── config/               # YAML configs per method
├── model/                # SSL method implementations
├── encoder/              # Backbone architectures
├── transform/            # Per-method data augmentation pipelines
│   └── augmentations.py  # Shared augmentation primitives
├── dataset/              # Samplers and dataset utilities
├── util/                 # Checkpointing, metrics, wandb logging
└── script/               # Shell scripts for experiments
```

## Configuration

Each method has a YAML config in `config/`. The config system uses [expedantic](https://github.com/rnilva/expedantic) for typed config parsing with CLI overrides.

Key config fields:

```yaml
dataset: miniImageNet          # Training dataset
eval_dataset: miniImageNet     # Evaluation dataset
dataset_path: /path/to/data
encoder: ResNet18              # Backbone architecture
batch_size: 256
num_epoch: 500
lr: 0.03
proj:
    proj_size: 512-512-128     # Projector MLP dimensions
    activate_fn: ReLU
model:
    name: SimCLR               # SSL method name
    temperature: 0.5           # Method-specific hyperparams
```

Experiment names are auto-generated with timestamps and random seeds. Checkpoints, logs, and configs are saved to `save_path/<exp_name>/`.

## Adding a New Method

1. Create `model/<method>.py` inheriting from `UnSup` (or `Sup`)
2. Create `transform/<method>_transform.py` with the augmentation pipeline
3. Add a config class to `config.py` and include it in the `Config.model` union
4. Create `config/<method>.yaml`
5. Register the model, encoder, and transform in their respective `__init__.py` registries

## Citation

```bibtex
@article{seo2026empirical,
  title={An empirical study of unsupervised few-shot learning that utilizes self-supervised representation learning},
  author={Seo, Sungwon and Kim, Jong-Kook},
  journal={Neurocomputing},
  volume={681},
  pages={133357},
  year={2026},
  publisher={Elsevier}
}
```
