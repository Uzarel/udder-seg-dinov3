import torch
import torch_pruning as tp

from pytorch_lightning.loggers import WandbLogger


def log_macs_params(model, example_inputs, logger=None):
    """
    Logs:
      {"macs_g": ..., "params_m": ...}
    to either a Lightning WandbLogger (`logger`) or the global `wandb` run.
    """
    was_training = model.training
    device = next(model.parameters()).device
    ex = example_inputs.to(device)

    with torch.no_grad():
        model.eval()
        macs, params = tp.utils.count_ops_and_params(model, ex)

    if was_training:
        model.train()

    record = {"macs_g": macs / 1e9, "params_m": params / 1e6}
    
    if logger is not None and hasattr(logger, "experiment"):
        logger.experiment.log(record)   # for WandbLogger
    else:
        import wandb
        wandb.log(record) # for a plain wandb run

def get_loggers(project_name, model_name, wandb_offline):
    train_logger = WandbLogger(project=project_name, name=model_name, offline=wandb_offline)
    prune_logger = WandbLogger(project=project_name, name=f"{model_name}_pruned", offline=wandb_offline)
    return train_logger, prune_logger
