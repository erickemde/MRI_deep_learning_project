from src.models.lit_vgg_attention import *

# define experiments
# experiment | experiment_name, model_description, use_augmentation, model_class, kwargs
EXPERIMENTS = {
    "baseline": (
        "vgg19_baseline", "VGG19 Baseline", False, VGG19Baseline, {}
    ),
    "baseline_aug": (
        "vgg19_baseline_aug", "VGG19 Baseline", True, VGG19Baseline, {}
    ),
    "softmax_attention": (
        "vgg19_softmax_attention", "VGG19 + Softmax Attention (Frozen)", False, VGG19SoftmaxAttention, {}
    ),
    "softmax_attention_finetune": (
        "vgg19_softmax_attention_finetune", "VGG19 + Softmax Attention (Partial Fine-tune)", False, VGG19SoftmaxAttention, {"unfreeze_from_layer", 35}
    ),
    "se_attention": (
        "vgg19_se_attention", "VGG19 + SE Attention", False, VGG19SEAttention, {"reduction": 16}
    ),
    "se_attention_aug": (
        "vgg19_se_attention_aug", "VGG19 + SE Attention", True, VGG19SEAttention, {"reduction":16}
    )
}

def setup_experiment(config):
    experiment = config["experiment"]
    
    if config["experiment"] not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {experiment}")
    
    experiment_name, model_decription, use_augmentation, model_class, model_kwargs = EXPERIMENTS[experiment]

    return {
        **config,
        "experiment_name": experiment_name,
        "model_description": model_decription,
        "use_augmentation": use_augmentation,
        "model_class": model_class,
        "model_kwargs": model_kwargs,
        "checkpoint_dir": f"checkpoints/{experiment_name}",
        "gradcam_dir": f"gradcam_examples/{experiment_name}"
    }

def build_model(config):
    model = config["model_class"](pretrained=True, **config["model_kwargs"])
    print("\tModel:", config["model_description"])
    return VGGLightningWrapper(model, lr=config["lr"])

def setup_ablation_study(config):
    '''
    Create a dictionary of configs for each experiment in EXPERIMENTS, using shared hyperparameters from the config file
    '''
    experiment_dict = {}
    for experiment, args in EXPERIMENTS.items():
        experiment_name, model_decription, use_augmentation, model_class, model_kwargs = args

        experiment_dict[experiment] = {
            **config,
            "experiment_name": experiment_name,
            "model_description": model_decription,
            "use_augmentation": use_augmentation,
            "model_class": model_class,
            "model_kwargs": model_kwargs,
            "checkpoint_dir": f"checkpoints/{experiment_name}",
            "gradcam_dir": f"gradcam_examples/{experiment_name}"
        }
    
    return experiment_dict