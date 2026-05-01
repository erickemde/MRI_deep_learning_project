from src.models.LightningWrapper import LightningWrapper
from src.models.vgg19 import (
    VGG19Baseline, 
    VGG19SEAttention, 
    VGG19SoftmaxAttention,
    VGG19CBAMAttention,
    VGG19SelfAttention,
    VGG19LoRA
)

EXPERIMENTS = {
    # ── Baseline ──────────────────────────────────────────────────────────────
    "baseline": (
        "vgg19_baseline", "VGG19 Baseline", False, VGG19Baseline, {}
    ),
    "baseline_aug": (
        "vgg19_baseline_aug", "VGG19 Baseline + Aug", True, VGG19Baseline, {}
    ),

    # ── Softmax Attention ─────────────────────────────────────────────────────
    "softmax_attention": (
        "vgg19_softmax_attention", "VGG19 + Softmax Attention (Frozen)", False, VGG19SoftmaxAttention, {}
    ),
    "softmax_attention_aug": (
    "vgg19_softmax_attention_aug", "VGG19 + Softmax Attention + Aug", True, VGG19SoftmaxAttention, {}
    ),
    "softmax_attention_finetune": (
        "vgg19_softmax_attention_finetune", "VGG19 + Softmax Attention (Partial Fine-tune)", False, VGG19SoftmaxAttention, {"unfreeze_from_layer": 35}
    ),
    "softmax_attention_finetune_aug": (
        "vgg19_softmax_attention_finetune_aug",
        "VGG19 + Softmax Attention (Partial Fine-tune) + Aug",
        True,
        VGG19SoftmaxAttention,
        {"unfreeze_from_layer": 40}
    ),

    # ── SE Attention ──────────────────────────────────────────────────────────
    "se_attention": (
        "vgg19_se_attention", "VGG19 + SE Attention (Frozen)", False, VGG19SEAttention, {"reduction": 16}
    ),
    "se_attention_aug": (
        "vgg19_se_attention_aug", "VGG19 + SE Attention + Aug", True, VGG19SEAttention, {"reduction": 16}
    ),
    "se_attention_finetune": (
        "vgg19_se_attention_finetune", "VGG19 + SE Attention (Fine-tune)", False, VGG19SEAttention, {"reduction": 16, "unfreeze_from_layer": 40}
    ),
    "se_attention_finetune_aug": (
        "vgg19_se_attention_finetune_aug", "VGG19 + SE Attention (Fine-tune) + Aug", True, VGG19SEAttention, {"reduction": 16, "unfreeze_from_layer": 40}
    ),

    # ── CBAM Attention ────────────────────────────────────────────────────────
    "cbam": (
        "vgg19_cbam", "VGG19 + CBAM Attention (Frozen)", False, VGG19CBAMAttention, {}
    ),
    "cbam_aug": (
        "vgg19_cbam_aug", "VGG19 + CBAM Attention + Aug", True, VGG19CBAMAttention, {}
    ),
    "cbam_finetune": (
        "vgg19_cbam_finetune", "VGG19 + CBAM Attention (Fine-tune)", False, VGG19CBAMAttention, {"unfreeze_from_layer": 40}
    ),
    "cbam_finetune_aug": (
        "vgg19_cbam_finetune_aug", "VGG19 + CBAM Attention (Fine-tune) + Aug", True, VGG19CBAMAttention, {"unfreeze_from_layer": 40}
    ),

    # ── Self Attention ────────────────────────────────────────────────────────
    "self_attention": (
        "vgg19_self_attention", "VGG19 + Self Attention (Frozen)", False, VGG19SelfAttention, {}
    ),
    "self_attention_aug": (
        "vgg19_self_attention_aug", "VGG19 + Self Attention + Aug", True, VGG19SelfAttention, {}
    ),
    "self_attention_finetune": (
        "vgg19_self_attention_finetune", "VGG19 + Self Attention (Fine-tune)", False, VGG19SelfAttention, {"unfreeze_from_layer": 40}
    ),
    "self_attention_finetune_aug": (
        "vgg19_self_attention_finetune_aug", "VGG19 + Self Attention (Fine-tune) + Aug", True, VGG19SelfAttention, {"unfreeze_from_layer": 40}
    ),
    # ── LoRA ────────────────────────────────────────────────────────
    "lora": ("vgg19_lora", "VGG19 + LoRA", False, VGG19LoRA, {"rank": 8})
}


def setup_experiment(config):
    experiment = config["experiment"]

    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {experiment}")

    experiment_name, model_description, use_augmentation, model_class, model_kwargs = EXPERIMENTS[experiment]

    return {
        **config,
        "experiment_name": experiment_name,
        "model_description": model_description,
        "use_augmentation": use_augmentation,
        "model_class": model_class,
        "model_kwargs": model_kwargs,
        "checkpoint_dir": f"checkpoints/{experiment_name}",
        "gradcam_dir": f"gradcam_examples/{experiment_name}",
    }


def build_model(config):
    model = config["model_class"](pretrained=True, **config["model_kwargs"])
    print("\tModel:", config["model_description"])
    return LightningWrapper(model, lr=config["lr"], weight_decay=config.get("weight_decay", 0.0))


def setup_ablation_study(config):
    '''
    Create a dictionary of configs for each experiment in EXPERIMENTS, using shared hyperparameters from the config file
    '''
    experiment_dict = {}
    for experiment, args in EXPERIMENTS.items():
        experiment_name, model_description, use_augmentation, model_class, model_kwargs = args

        experiment_dict[experiment] = {
            **config,
            "experiment_name": experiment_name,
            "model_description": model_description,
            "use_augmentation": use_augmentation,
            "model_class": model_class,
            "model_kwargs": model_kwargs,
            "checkpoint_dir": f"checkpoints/{experiment_name}",
            "gradcam_dir": f"gradcam_examples/{experiment_name}"
        }
    
    return experiment_dict