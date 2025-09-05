import torch
import torch_pruning as tp
import pytorch_lightning as pl
import wandb

from callbacks import get_callbacks
from config import get_config
from dataloaders import get_dataloaders
from dinov3_backbone import get_backbone    
from lit_dino import LitDinoModule
from logger import get_loggers, log_macs_params
from loss import get_segmentation_loss


# Get config
cfg = get_config()
SEED = cfg["GENERAL"]["SEED"]
DINOV3_REPO_SOURCE = cfg["GENERAL"]["DINOV3_REPO_SOURCE"]
DINOV3_REPO_LOCATION = cfg["GENERAL"]["DINOV3_REPO_LOCATION"]
MODEL_ENCODER = cfg["MODEL"]["MODEL_ENCODER"]
MODEL_NAME = cfg["MODEL"]["MODEL_NAME"]
MODEL_WEIGHTS = cfg["MODEL"]["MODEL_WEIGHTS"]
TRAIN_IMAGES = cfg["DATA"]["TRAIN_IMAGES"]
TRAIN_MASKS  = cfg["DATA"]["TRAIN_MASKS"]
VAL_IMAGES   = cfg["DATA"]["VAL_IMAGES"]
VAL_MASKS    = cfg["DATA"]["VAL_MASKS"]
TEST_IMAGES  = cfg["DATA"]["TEST_IMAGES"]
TEST_MASKS   = cfg["DATA"]["TEST_MASKS"]
BATCH_SIZE = cfg["TRAINING"]["BATCH_SIZE"]
MAX_EPOCHS = cfg["TRAINING"]["MAX_EPOCHS"]
LR = cfg["TRAINING"]["LR"]
WEIGHT_DECAY = cfg["TRAINING"]["WEIGHT_DECAY"]
WARMPUP_RATIO = cfg["TRAINING"]["WARMUP_RATIO"]
LOSS_BCE = cfg["LOSS"]["BCE"]
LOSS_DICE = cfg["LOSS"]["DICE"]
LOSS_TVERSKY = cfg["LOSS"]["TVERSKY"]
LOSS_LOVASZ = cfg["LOSS"]["LOVASZ"]
LOSS_FOCAL = cfg["LOSS"]["FOCAL"]
LOSS_IGNORE_INDEX = cfg["LOSS"]["IGNORE_INDEX"]
WANDB_OFFLINE = cfg['WANDB']['OFFLINE']
PROJECT_NAME = cfg['WANDB']['PROJECT_NAME']

# Seed everything rng related
pl.seed_everything(SEED)

train_logger, prune_logger = get_loggers(
    model_name=MODEL_NAME,
    project_name=PROJECT_NAME,
    wandb_offline=WANDB_OFFLINE
    )

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=-1,
    precision=32,
    callbacks=get_callbacks(model_name=MODEL_NAME, max_epochs=MAX_EPOCHS, mode="train"),
    logger=train_logger,
)

train_loader, val_loader, test_loader = get_dataloaders(
    TRAIN_IMAGES, TRAIN_MASKS,
    VAL_IMAGES, VAL_MASKS,
    TEST_IMAGES, TEST_MASKS,
    BATCH_SIZE
)

# Training loop
loss = get_segmentation_loss(LOSS_BCE, LOSS_DICE, LOSS_TVERSKY, LOSS_LOVASZ, LOSS_FOCAL, ignore_index=LOSS_IGNORE_INDEX)
backbone = get_backbone(DINOV3_REPO_SOURCE, DINOV3_REPO_LOCATION, MODEL_NAME, MODEL_WEIGHTS)
lit_train = LitDinoModule(backbone, encoder=MODEL_ENCODER, loss=loss, lr=LR, weight_decay=WEIGHT_DECAY, warmup_ratio=WARMPUP_RATIO, freeze_backbone=True)
trainer.fit(lit_train, train_loader, val_loader)
example_inputs = torch.randn(1, 1, 480, 640)
log_macs_params(lit_train.model, example_inputs, train_logger)

# Testing loop
ckpt_cb = trainer.checkpoint_callback
best_ckpt = ckpt_cb.best_model_path
lit_test = LitDinoModule.load_from_checkpoint(best_ckpt, encoder=MODEL_ENCODER, backbone=backbone, loss=loss)
torch.save(lit_test.model.state_dict(), f"checkpoints/{MODEL_NAME.lower()}/best.pt")
trainer.test(lit_test, test_loader)
wandb.finish()

# Pruning
lit_prune = LitDinoModule.load_from_checkpoint(best_ckpt, encoder=MODEL_ENCODER, backbone=backbone, loss=loss)
model = lit_prune.model
model.eval().cpu()

imp = tp.importance.GroupMagnitudeImportance(p=2)  # L2 over grouped weights
ignored_layers = [
    model.encoder.input_adapter.proj,
    model.seg_head,
]

# Pruner (global + isomorphic) and channels round to 8 for Jetson/TensorRT
pruner = tp.pruner.BasePruner(
    model,
    example_inputs,
    importance=imp,
    global_pruning=True,
    isomorphic=True,
    pruning_ratio=0.30, # approx. 50% param reduction due to bidirectional coupling
    ignored_layers=ignored_layers,
    round_to=8, # Jetson/TensorRT friendly
)

pruner.step() # Prune the graph
log_macs_params(model, example_inputs, prune_logger)

# Put the pruned module back; unfreeze to fine-tune everything
lit_prune.model = model
for p in lit_prune.model.parameters():
    p.requires_grad = True

# Recreate the optimizer (shapes changed!)
trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=-1,
    precision=32,
    strategy="ddp_find_unused_parameters_true",
    callbacks=get_callbacks(model_name=MODEL_NAME, max_epochs=MAX_EPOCHS, mode="prune"),
    logger=prune_logger,
)
trainer.fit(lit_prune, train_loader, val_loader)
pruned_ckpt_cb = trainer.checkpoint_callback
pruned_best_ckpt = pruned_ckpt_cb.best_model_path

# Load the state_dict from the checkpoint into the already-pruned LightningModule
ckpt = torch.load(pruned_best_ckpt, map_location="cpu")
lit_prune.load_state_dict(ckpt["state_dict"], strict=True)

# Save/export weights and test
torch.save(lit_prune.model.state_dict(), f"checkpoints/{MODEL_NAME.lower()}_pruned/best.pt")
trainer.test(lit_prune, test_loader)
wandb.finish()
