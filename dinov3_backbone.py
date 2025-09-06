import torch


def get_backbone(dinov3_repo_source, dinov3_repo_location, model_name, model_weights = None):
    """Load the DINOv3 backbone from torch hub."""
    if model_weights:
        backbone = torch.hub.load(
            repo_or_dir=dinov3_repo_location,
            model=model_name,
            source=dinov3_repo_source,
            weights=model_weights
        )
    else:
        backbone = torch.hub.load(
            repo_or_dir=dinov3_repo_location,
            model=model_name,
            source=dinov3_repo_source,
    )
    return backbone
