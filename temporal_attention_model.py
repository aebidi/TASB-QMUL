import torch
from torch import nn
from collections import OrderedDict
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from attention import TemporalFeatureMapAttention

class FPN_Temporal_Attention_FasterRCNN(nn.Module):
    def __init__(self, num_classes, num_frames):
        super().__init__()
        self.num_frames = num_frames
        
        # loading a standard Faster R-CNN model to steal its components
        base_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        
        # --- 1. Steal the pre-trained components ---
        self.transform = base_model.transform
        self.backbone = base_model.backbone # This is the ResNet+FPN
        self.rpn = base_model.rpn
        self.roi_heads = base_model.roi_heads
        
        # --- 2. Create an Attention Module for EACH FPN level ---
        # the FPN for ResNet50 outputs 256 channels for all its feature maps
        fpn_out_channels = self.backbone.out_channels
        
        # creating a dictionary of attention modules, one for each FPN output
        self.temporal_attentions = nn.ModuleDict({
            name: TemporalFeatureMapAttention(in_channels=fpn_out_channels) 
            for name in ['0', '1', '2', '3', 'pool'] # Standard FPN output names
        })
        
        # --- 3. Replace the classification head ---
        in_features = self.roi_heads.box_predictor.cls_score.in_features
        self.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)


    def forward(self, images, targets=None):
        # setup
        is_training = self.training and targets is not None
        images = torch.stack(images, dim=0)
        B, T, C, H, W = images.shape
        center_idx = T // 2

        # --- Backbone Processing ---
        # reshaping to [B*T, C, H, W] to process all frames at once
        images_reshaped = images.view(B * T, C, H, W)
    
        # we need a list of the central images for the final stages
        central_images_list = list(images[:, center_idx].unbind(0))
    
        # we transform the whole stack, this is simpler to implement
        dummy_targets = None
        if is_training:
            dummy_targets = [t for t in targets for _ in range(T)]
        transformed_images, _ = self.transform(images_reshaped, dummy_targets)
    
        # get the dictionary of 5 feature maps for the whole stack
        all_features = self.backbone(transformed_images.tensors)
    
        # --- Temporal Attention on EACH FPN level ---
        refined_features = OrderedDict()
    
        for name, feature_map in all_features.items():
            # reshaping the feature map into a sequence
            C_feat, H_feat, W_feat = feature_map.shape[1:]
            features_sequence = feature_map.view(B, T, C_feat, H_feat, W_feat)
            features_sequence_TBC = features_sequence.permute(1, 0, 2, 3, 4)
        
            # applying the corresponding attention module (now with gamma)
            refined_feature_map = self.temporal_attentions[name](features_sequence_TBC)
        
            refined_features[name] = refined_feature_map
            
        # --- Detection Heads ---
        # we need the transformed central images and their valid targets
        transformed_central_images, valid_targets = self.transform(central_images_list, targets)
    
        proposals, proposal_losses = self.rpn(transformed_central_images, refined_features, valid_targets)
        detections, detector_losses = self.roi_heads(refined_features, proposals, transformed_central_images.image_sizes, valid_targets)

        if is_training:
            return {**proposal_losses, **detector_losses}
        else:
            # during eval, transform the detections back to original image sizes
            original_image_sizes = [img.shape[-2:] for img in central_images_list]
            detections = self.transform.postprocess(detections, transformed_central_images.image_sizes, original_image_sizes)
            return detections