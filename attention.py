import torch
from torch import nn

class TemporalFeatureMapAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention_scorer = nn.Conv2d(in_channels * 2, 1, kernel_size=1)
        
        # a learnable parameter, initialized to 0.
        self.gamma = nn.Parameter(torch.zeros(1))
    

    def forward(self, feature_maps_sequence):
        center_idx = feature_maps_sequence.size(0) // 2
        center_feature_map = feature_maps_sequence[center_idx]
        
        scores = []
        for frame_idx in range(feature_maps_sequence.size(0)):
            other_feature_map = feature_maps_sequence[frame_idx]
            combined = torch.cat([center_feature_map, other_feature_map], dim=1)
            score = self.attention_scorer(combined)
            scores.append(score)
            
        scores_tensor = torch.cat(scores, dim=1)
        attention_weights = torch.softmax(scores_tensor, dim=1).unsqueeze(2)
        
        sequence_reshaped = feature_maps_sequence.permute(1, 0, 2, 3, 4)
        
        # calculating the attention-weighted sum of all features
        attended_features = (attention_weights * sequence_reshaped).sum(dim=1)
        
        # final output is the original center feature plus the attended features,
        # scaled by our learnable gamma
        refined_feature_map = center_feature_map + self.gamma * attended_features
        
        return refined_feature_map