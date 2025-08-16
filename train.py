# train.py (Corrected for V8 - No Augmentations)

import os
import torch
import argparse
import random
import numpy as np
from tqdm import tqdm
import torchmetrics

# Project-specific imports
import config
from dataset import IVUSSideBranchDataset
from temporal_attention_model import FPN_Temporal_Attention_FasterRCNN
# THIS IS THE PRIMARY FIX: Import the correctly named functions
from augmentations import get_train_transforms, get_val_transforms
from utils import collate_fn, EarlyStopping, save_plots

def set_seed(seed):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    prog_bar = tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch+1} [Training]")
    total_loss = 0.0
    for i, (images, targets) in enumerate(prog_bar):
        # Dataloader now returns a [B, T, C, H, W] tensor, must be a list for the model
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        if not torch.isfinite(losses):
            print(f"!!! Infinite loss detected. Skipping batch {i}.")
            continue
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
        prog_bar.set_postfix(loss=f'{total_loss / (i + 1):.4f}')
    return total_loss / len(data_loader)

@torch.no_grad()
def evaluate(model, data_loader, device):
    torch.cuda.empty_cache()
    model.eval()
    metric = torchmetrics.detection.MeanAveragePrecision(box_format='xyxy')
    metric.to(device)
    prog_bar = tqdm(data_loader, total=len(data_loader), desc="[Validating]")
    for images, targets in prog_bar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(images)
        metric.update(outputs, targets)
    results = metric.compute()
    return results

def main():
    set_seed(config.SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-augment', action='store_true', help='Disable data augmentations.')
    args = parser.parse_args()
    use_augmentations = not args.no_augment

    # --- Datasets and DataLoaders ---
    train_dir = os.path.join(config.DATA_ROOT, 'train')
    val_dir = os.path.join(config.DATA_ROOT, 'val')

    # THIS IS THE FIX: We explicitly get the transform we want to use.
    # For this run, it will ALWAYS be get_val_transforms() because of the --no-augment flag.
    if use_augmentations:
        transform_to_use = get_train_transforms()
    else:
        print("INFO: Data augmentations are DISABLED for training.")
        transform_to_use = get_val_transforms()
    
    dataset_train = IVUSSideBranchDataset(train_dir, transform=transform_to_use)
    dataset_val = IVUSSideBranchDataset(val_dir, transform=get_val_transforms())
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn
    )
    
    print(f"Device: {config.DEVICE}")
    print(f"Found {len(dataset_train)} images for training.")
    print(f"Found {len(dataset_val)} images for validation.")
    
    # --- Model ---
    model = FPN_Temporal_Attention_FasterRCNN(
        num_classes=config.NUM_CLASSES,
        num_frames=(2 * config.TEMPORAL_FRAMES_K) + 1
    )
    print(f"INFO: Created the Temporal Attention Faster R-CNN model with {(2 * config.TEMPORAL_FRAMES_K) + 1} input channels")
    model.to(config.DEVICE)
    
    # --- Optimizer, Scheduler, Early Stopping ---
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config.LEARNING_RATE, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    early_stopping = EarlyStopping(patience=30, verbose=True, path=os.path.join(config.OUTPUT_DIR, 'best_model.pth'))
    
    # --- History Lists for Plotting ---
    train_loss_history = []
    val_metrics_history = []

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, optimizer, data_loader_train, config.DEVICE, epoch)
        val_metrics = evaluate(model, data_loader_val, device=config.DEVICE)

        # --- Store metrics for plotting ---
        train_loss_history.append(train_loss)
        val_metrics_history.append(val_metrics)

        current_map50 = val_metrics['map_50'].item()
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Training Loss: {train_loss:.4f}")
        print(f"  Validation mAP@50: {current_map50:.4f}")
        
        early_stopping(current_map50, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
            
        lr_scheduler.step()
        print("-" * 50)
        
    print("--- Training Finished ---")

    # --- Generate and Save Plots ---
    if train_loss_history and val_metrics_history:
        print("Generating and saving training plots...")
        save_plots(train_loss_history, val_metrics_history, config.OUTPUT_DIR)
    else:
        print("Skipping plot generation due to insufficient training data.")

if __name__ == '__main__':
    main()