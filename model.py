# model.py

import torch
from torch import nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# importing new attention layer
from attention import SpatialAttention

def get_model(num_classes, in_channels=3):
    """
    Loads a Faster R-CNN model and modifies it for:
    1. Custom input channels (for temporal stacking).
    2. An injected Spatial Attention layer.
    """
    # loading a pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # --- Part 1: Modify for custom input channels ---
    if in_channels != 3:
        # modify the first convolutional layer
        original_conv1 = model.backbone.body.conv1
        new_conv1 = nn.Conv2d(
            in_channels, 
            original_conv1.out_channels, 
            kernel_size=original_conv1.kernel_size, 
            stride=original_conv1.stride, 
            padding=original_conv1.padding, 
            bias=(original_conv1.bias is not None)
        )
        with torch.no_grad():
            new_conv1.weight.copy_(original_conv1.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1))
        model.backbone.body.conv1 = new_conv1
        
        # this part for handling the internal transform is no longer needed
        # if we handle normalization fully in the dataset/augmentations
        # but keeping it can be safer depending on the torchvision version
        grayscale_mean = [0.485] * in_channels
        grayscale_std = [0.229] * in_channels
        model.transform.image_mean = grayscale_mean
        model.transform.image_std = grayscale_std
        
    # --- Part 2: Inject the Spatial Attention Layer ---
    # the output of ResNet50's layer4 has 2048 channels
    attention_channels = 2048
    
    # we create a new Sequential module that contains the original layer4
    # followed by our new attention layer.
    original_layer4 = model.backbone.body.layer4
    attention_block = SpatialAttention(in_channels=attention_channels)
    
    # replacing the original layer4 with our new sequential block
    model.backbone.body.layer4 = nn.Sequential(original_layer4, attention_block)
    
    print(f"INFO: Injected SpatialAttention block after ResNet's layer4.")

    # --- Part 3: Modify the classifier head (as before) ---
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model