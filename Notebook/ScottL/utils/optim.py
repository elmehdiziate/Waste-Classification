import torch

# Build optimizer with optional staged learning rates for fine‑tuning
def get_optimizer(model, config: dict):
    opt_cfg = config["optimizer"]
    ft_cfg  = config["fine_tuning"]
    lr      = config["training"]["learning_rate"]

    # Staged learning rates protect pretrained backbone features by updating them more gently,
    # while allowing newly‑initialised classification layers to learn quickly.
    if ft_cfg["staged_lr"]:
        new_layer_names = ft_cfg.get("new_layers", ["fc", "classifier", "head"])
        head_params, backbone_params = [], []

        # Parameter grouping enables differential learning rates across the network.
        for name, param in model.named_parameters():
            if any(layer in name for layer in new_layer_names):
                head_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = [
            {"params": backbone_params, "lr": lr * ft_cfg["base_lr_mult"]},
            {"params": head_params,     "lr": lr}
        ]

        print(f"[Optimizer] Staged LR — backbone: {lr * ft_cfg['base_lr_mult']:.2e} | head: {lr:.2e}")

    else:
        # Uniform LR is appropriate when training from scratch or when full fine‑tuning is desired.
        param_groups = model.parameters()
        print(f"[Optimizer] Uniform LR — {lr:.2e}")

    # Optimizer choice shapes convergence behaviour:
    # Adam adapts per‑parameter learning rates and handles noisy gradients well,
    # while SGD with momentum often yields stronger generalisation in vision tasks.
    if opt_cfg["type"] == "adam":
        return torch.optim.Adam(
            param_groups,
            lr=lr,
            betas=(opt_cfg["adam_beta1"], opt_cfg["adam_beta2"]),
            weight_decay=opt_cfg["weight_decay"]
        )

    elif opt_cfg["type"] == "sgd":
        return torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=opt_cfg["momentum"],
            dampening=opt_cfg["sgd_dampening"],
            nesterov=opt_cfg["sgd_nesterov"],
            weight_decay=opt_cfg["weight_decay"]
        )

    else:
        raise ValueError(f"Unknown optimizer: {opt_cfg['type']}")


# Build LR scheduler based on config
def get_scheduler(optimizer, config: dict):
    sched_type = config["scheduler"]["type"]
    ft_cfg     = config["fine_tuning"]

    # Cosine annealing provides a smooth, non‑monotonic decay that helps the model
    # escape shallow minima and stabilises late‑stage training.
    if sched_type == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=ft_cfg["T_max"]
        )

    # Multi‑step decay applies abrupt LR drops at predefined epochs,
    # which can help the optimiser settle into a better minimum after plateaus.
    elif sched_type == "multi_step":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=ft_cfg["stepsize"],
            gamma=ft_cfg["gamma"]
        )

    else:
        raise ValueError(f"Unknown scheduler: {sched_type}")