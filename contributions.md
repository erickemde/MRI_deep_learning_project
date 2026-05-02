# Team Contributions

Grouped by team member. Summaries derived from git commit history.

---

## Erick Emde

### Core Model / Training
| File | Summary |
|---|---|
| `src/models/vgg19.py` | Core VGG19-BN model; added four attention modules, custom classifier head, freeze/unfreeze logic, and LoRA adapter |
| `src/data/augmentation.py` | Augmentation pipeline; removed ColorJitter (MRI clinical validity), merged RandomRotation/RandomAffine, added finetuning variant |
| `src/experiments/config.py` | Central experiment registry; expanded from 6 to 18+ configs, added LoRA entry |
| `ablation_study.py` | Full ablation runner over all experiment configs; integrated W&B logging, fixed bugs |
| `train_model.py` | Main training entry point; YAML-driven, supports GradCAM generation, HF upload, reproducibility seeding, and LoRA |
| `resnet_train.py` | ResNet baseline training script for comparison against VGG19 results |
| `model_eval.py` | Test-set evaluation; added GradCAM output, confusion matrix, CBAM/Self-Attention and LoRA support |
| `model_val_eval.py` | Validation-set evaluation; added sorted() for reproducibility and LoRA support |
| `visualize_augmentation.py` | Generates side-by-side augmentation comparison figures for the paper |
| `huggingface_upload.py` | Root-level HF upload script; later refactored into `src/utils/` |

### Configs
| File | Summary |
|---|---|
| `configs/lora.yaml` | Hyperparameters for LoRA training (rank=8, lr, epochs) |
| `configs/softmax_attention_finetune_aug.yaml` | Config for Softmax attention with finetuning and augmentation |
| `configs/eval.yaml` | Evaluation config for model_eval.py; updated as experiments expanded |
| `configs/val_eval.yaml` | Validation eval config; updated for LoRA and reproducibility fix |

### Project Setup
| File | Summary |
|---|---|
| `environment.yaml` | Conda environment spec; updated as dependencies were added (loralib, scikit-learn, HF) |
| `.gitignore` | Repo configuration |
| `.vscode/launch.json`, `.vscode/settings.json` | VS Code IDE configuration |
| `.claude/settings.local.json` | Claude Code settings |

---

## Kelton Southard

### Core Model / Training
| File | Summary |
|---|---|
| `src/models/LightningWrapper.py` | PyTorch Lightning training wrapper; evolved from lit_vgg.py, added GradCAM hooks, weight decay, and num_classes support |
| `src/models/lit_vgg_attention.py` | Attention-specific Lightning module; eventually merged into LightningWrapper |
| `src/experiments/config.py` | Contributed ablation study experiment setup |
| `ablation_study.py` | Initial ablation study scaffold |
| `train_model.py` | Contributed YAML-config refactor and experiment setup utilities |
| `resnet_train.py` | Co-developed ResNet comparison trainer |
| `model_eval.py` | Co-developed initial HuggingFace-based evaluation script |

### Configs
| File | Summary |
|---|---|
| `configs/ablation.yaml` | Config for the full ablation sweep; updated epochs and W&B settings |
| `configs/eval.yaml` | Contributed eval config updates |

### Project Setup
| File | Summary |
|---|---|
| `environment.yaml` | Contributed dependency additions |
| `eda.ipynb` | Exploratory data analysis notebook for the brain tumor MRI dataset |
| `eda.yml` | Conda environment for EDA work |
| `template.sbatch` | SLURM batch script for Georgia Tech's ICE-PACE HPC cluster |
| `.gitignore` | Repo configuration |

---

## Man Young Lee

### Core Model / Data
| File | Summary |
|---|---|
| `src/models/vgg19.py` | Co-developed attention module integration and experiment runs |
| `src/models/vgg19_attention.py` | Standalone attention model definitions; later consolidated into vgg19.py |
| `src/models/lit_vgg_attention.py` | Co-developed attention Lightning module |
| `src/models/__init__.py` | Module exports; updated as models were consolidated |
| `src/data/dataset.py` | Dataset class for loading and splitting brain tumor MRI data |
| `src/data/augmentation.py` | Co-developed augmentation pipeline; contributed initial finetuning variant |
| `src/data/__init__.py` | Package init for data module |
| `src/experiments/config.py` | Contributed 18-experiment expansion (CBAM, Self-Attention, SE, Softmax variants) |
| `src/__init__.py` | Top-level package init |

### Training Scripts
| File | Summary |
|---|---|
| `train_model.py` | Co-developed; contributed 18-experiment run support |
| `train_with_augmentation.py` | Standalone augmentation ablation script; showed +2.84% accuracy improvement |
| `data_preprocessing.py` | Preprocesses raw Kaggle dataset into train/val/test splits |
| `model_val_eval.py` | Co-developed validation evaluation script |
| `visualize_augmentation.py` | Co-developed augmentation visualization figures |
| `run.sh` | Shell script to sequentially launch all 18 training runs; added reproducibility fixes |

### Configs
| File | Summary |
|---|---|
| `configs/cbam.yaml` | Frozen CBAM experiment config |
| `configs/cbam_aug.yaml` | CBAM with augmentation config |
| `configs/cbam_finetune.yaml` | CBAM with finetuning config |
| `configs/cbam_finetune_aug.yaml` | CBAM with finetuning and augmentation config |
| `configs/se_attention_finetune.yaml` | SE attention finetuning config |
| `configs/se_attention_finetune_aug.yaml` | SE attention finetuning with augmentation config |
| `configs/self_attention.yaml` | Frozen Self-Attention config |
| `configs/self_attention_aug.yaml` | Self-Attention with augmentation config |
| `configs/self_attention_finetune.yaml` | Self-Attention with finetuning config |
| `configs/self_attention_finetune_aug.yaml` | Self-Attention with finetuning and augmentation config |
| `configs/softmax_attention_aug.yaml` | Softmax attention with augmentation config |
| `configs/softmax_attention_finetune_aug.yaml` | Softmax attention with finetuning and augmentation config |
| `configs/eval.yaml` | Contributed eval config updates |
| `configs/val_eval.yaml` | Contributed validation eval config updates |

### Project Setup / Artifacts
| File | Summary |
|---|---|
| `environment.yaml` | Contributed dependency additions |
| `logs/` | Training logs for all 18 experiment runs |

---

## Alex Gelpi

### Core Model
| File | Summary |
|---|---|
| `src/models/vgg19.py` | Co-developed; contributed initial model scaffolding and file structure simplification |
| `src/models/vgg19_attention.py` | Co-developed; initial Custom Channel Attention implementation |
| `src/models/LightningWrapper.py` | Co-developed; original Lightning module author |
| `src/models/lit_vgg.py` | Original Lightning module for plain VGG19; superseded by LightningWrapper |
| `src/models/lit_vgg_attention.py` | Co-developed; initial CBAM/attention Lightning module |
| `src/models/gradcam.py` | Created and iteratively refined GradCAM implementation; later moved to `src/visualization/` |
| `src/models/__init__.py` | Initial module init |
| `src/experiments/config.py` | Created initial experiment registry and setup utilities |
| `src/experiments/__init__.py` | Package init for experiments module |
| `src/utils/huggingface_upload.py` | HF upload utility refactored from root-level script |
| `src/visualization/gradcam.py` | Final home of GradCAM; fixed saliency map bug, added confusion matrix output |
| `models/vgg19.py` | Early prototype model file before src/ restructure |

### Training / Inference Scripts
| File | Summary |
|---|---|
| `train.py` | Original training script; iteratively improved then renamed to simple_train.py and deleted |
| `simple_train.py` | Renamed from train.py; deleted once train_model.py became the main entry point |
| `train_model.py` | Co-developed; contributed GradCAM integration and file structure simplification |
| `train_with_augmentation.py` | Co-developed augmentation ablation script |
| `ablation_study.py` | Co-developed; contributed file structure simplification pass |
| `gradcam_inference.py` | Standalone script for generating GradCAM saliency maps from saved checkpoints |
| `model_eval.py` | Created initial HuggingFace-based evaluation script |
| `data_preprocessing.py` | Co-developed; contributed initial preprocessing script |
| `download_data.py` | Downloads brain tumor MRI dataset from Kaggle; path aligned with preprocessing script |
| `huggingface_upload.py` | Created initial HF upload script at root level |

### Configs
| File | Summary |
|---|---|
| `configs/baseline.yaml` | Frozen baseline experiment config |
| `configs/baseline_aug.yaml` | Baseline with augmentation config |
| `configs/se_attention.yaml` | Frozen SE attention config |
| `configs/se_attention_aug.yaml` | SE attention with augmentation config |
| `configs/se_attention_finetune.yaml` | SE attention with finetuning config |
| `configs/softmax_attention.yaml` | Frozen Softmax attention config |
| `configs/softmax_attention_finetune.yaml` | Softmax attention with finetuning config |
| `configs/softmax_attention_finetune_aug.yaml` | Softmax attention with finetuning and augmentation config |
| `configs/self_attention_finetune_aug.yaml` | Self-Attention with finetuning and augmentation config |
| `configs/cbam_finetune_aug.yaml` | CBAM with finetuning and augmentation config |
| `configs/eval.yaml` | Co-developed; added generate_gradcam flag and consolidated config structure |
| `configs/gradcam/` | GradCAM-specific inference configs (baseline, SE, Softmax variants) |
| `configs/train/` | Training configs organized by experiment (baseline, SE, Softmax variants) |

### Project Setup / Notebooks
| File | Summary |
|---|---|
| `requirements.txt` | pip requirements; incrementally updated as new packages were needed |
| `eda.yml` | EDA conda environment |
| `notebooks/eda.ipynb` | Initial EDA notebook before src/ restructuring |
| `model_testing.ipynb` | Notebook for interactive model testing and debugging |
| `.gitignore` | Repo configuration |
| `.vscode/launch.json`, `.vscode/settings.json` | VS Code IDE configuration |
| `.claude/settings.local.json` | Claude Code settings |
