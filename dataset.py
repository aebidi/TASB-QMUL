# dataset.py (Corrected for V8 - No Augmentations)

import torch
import cv2
import numpy as np
import os
import config

# THIS IS THE PRIMARY FIX: Import the correctly named function.
from augmentations import get_val_transforms

class IVUSSideBranchDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform # This will be get_val_transforms
        self.image_files = []
        self.annotation_files = []

        # ... (The file loading part is correct, no changes needed) ...
        for series_folder in sorted(os.listdir(self.root_dir)):
            series_path = os.path.join(self.root_dir, series_folder)
            if not os.path.isdir(series_path): continue
            for file_name in sorted(os.listdir(series_path)):
                if file_name.endswith('_bboxes.npy'):
                    base_name = file_name.replace('_bboxes.npy', '')
                    img_path = os.path.join(series_path, f"{base_name}.jpg")
                    if os.path.exists(img_path):
                        self.image_files.append(img_path)
                        self.annotation_files.append(os.path.join(series_path, file_name))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # --- 1. Load the sequence of raw frames ---
        center_img_path = self.image_files[idx]
        k = config.TEMPORAL_FRAMES_K
        raw_frames_to_load = []
        path_parts = center_img_path.rsplit('_', 1)
        base_path = path_parts[0]
        center_frame_num = int(path_parts[1].replace('.jpg', ''))
        center_color_frame = cv2.cvtColor(cv2.imread(center_img_path), cv2.COLOR_BGR2RGB)
        for i in range(center_frame_num - k, center_frame_num + k + 1):
            frame_path = f"{base_path}_{i:04d}.jpg"
            if os.path.exists(frame_path):
                frame = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
            else:
                frame = center_color_frame
            raw_frames_to_load.append(frame)

        # --- 2. Load annotations ---
        annot_path = self.annotation_files[idx]
        boxes_xywh = np.load(annot_path, allow_pickle=True)
        boxes = []
        for box in boxes_xywh:
            x_min, y_min, x_max, y_max = box['x'], box['y'], box['x'] + box['width'], box['y'] + box['height']
            boxes.append([x_min, y_min, x_max, y_max])
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        # --- 3. Apply a consistent transform to each frame ---
        transformed_frames = []
        if self.transform is not None:
            # The transform passed from train.py is get_val_transforms.
            # It has NO bbox_params, so it can be safely called on each image.
            for frame in raw_frames_to_load:
                transformed_frames.append(self.transform(image=frame)['image'])
        
        # Stack the list of processed [C, H, W] tensors into a single [T, C, H, W] tensor
        image_stack = torch.stack(transformed_frames, dim=0)

        # --- 4. Manually Scale Bounding Boxes ---
        orig_h, orig_w, _ = center_color_frame.shape
        scale_h = config.RESOLUTION / orig_h
        scale_w = config.RESOLUTION / orig_w
        scaled_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            scaled_boxes.append([x1 * scale_w, y1 * scale_h, x2 * scale_w, y2 * scale_h])

        # --- 5. Prepare the final target dictionary ---
        final_target = {}
        if scaled_boxes:
            final_target['boxes'] = torch.as_tensor(scaled_boxes, dtype=torch.float32)
        else:
            final_target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
        final_target['labels'] = labels
        final_target['image_id'] = torch.tensor([idx])
            
        return image_stack, final_target