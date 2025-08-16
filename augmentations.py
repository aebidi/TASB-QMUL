# augmentations.py

import albumentations as A
from albumentations.pytorch import ToTensorV2
import config

def get_train_transforms():
    """Returns the FULL augmentation pipeline for training."""
    # We are NOT using this function in the current experiment.
    # It is here for when you decide to re-enable augmentations.
    print("INFO: Using data augmentations for training.")
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        # Add other random augmentations here in the future...
        A.Normalize(),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def get_val_transforms():
    """Returns the SIMPLE pipeline for validation, testing, or training without random augmentations."""
    return A.Compose([
        A.Resize(height=config.RESOLUTION, width=config.RESOLUTION),
        A.Normalize(),
        ToTensorV2(),
    ])