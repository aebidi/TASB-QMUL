import torch
import cv2
import numpy as np
import os
import random

# Project-specific imports
import config
from dataset import IVUSSideBranchDataset
from augmentations import get_val_transforms

# --- THIS IS THE PRIMARY FIX: Import the correct V9 model ---
from temporal_attention_model import FPN_Temporal_Attention_FasterRCNN

def main():
    # --- Configuration ---
    MODEL_PATH = os.path.join(config.OUTPUT_DIR, 'best_model.pth')
    TEST_DATA_DIR = os.path.join(config.DATA_ROOT, 'test')
    OUTPUT_VIS_DIR = os.path.join(config.OUTPUT_DIR, 'visualisations_v8')
    
    DEVICE = config.DEVICE
    CONFIDENCE_THRESHOLD = config.CONFIDENCE_THRESHOLD
    NUM_IMAGES_TO_VISUALIZE = 20

    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)
    
    # --- 1. BUILD THE CORRECT V9 MODEL ---
    print(f"INFO: Building the FPN Temporal Attention model for visualisation.")
    model = FPN_Temporal_Attention_FasterRCNN(
        num_classes=config.NUM_CLASSES,
        num_frames=(2 * config.TEMPORAL_FRAMES_K) + 1
    )
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # --- Load Dataset (used as a file index) ---
    dataset = IVUSSideBranchDataset(root_dir=TEST_DATA_DIR, transform=None)
    
    image_indices = random.sample(range(len(dataset)), NUM_IMAGES_TO_VISUALIZE)
    print(f"Visualising {len(image_indices)} random images...")

    val_transform = get_val_transforms()

    for idx in image_indices:
        # --- 2. LOAD THE TEMPORAL RGB DATA STACK ---
        center_img_path = dataset.image_files[idx]
        k = config.TEMPORAL_FRAMES_K
        raw_frames_to_load = []
        path_parts = center_img_path.rsplit('_', 1)
        base_path = path_parts[0]
        center_frame_num = int(path_parts[1].replace('.jpg', ''))
        center_color_frame_raw = cv2.cvtColor(cv2.imread(center_img_path), cv2.COLOR_BGR2RGB)
        for i in range(center_frame_num - k, center_frame_num + k + 1):
            frame_path = f"{base_path}_{i:04d}.jpg"
            if os.path.exists(frame_path):
                frame = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
            else:
                frame = center_color_frame_raw
            raw_frames_to_load.append(frame)

        # --- 3. PREPARE INPUT TENSOR FOR THE MODEL ---
        transformed_frames = []
        for frame in raw_frames_to_load:
            transformed_frames.append(val_transform(image=frame)['image'])
        image_stack = torch.stack(transformed_frames, dim=0)
        
        with torch.no_grad():
            outputs = model([image_stack.to(DEVICE)])
        
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        
        # --- 4. DRAW BOXES ON THE *RESIZED* CENTRAL IMAGE ---
        image_to_draw_on = cv2.resize(center_color_frame_raw, (config.RESOLUTION, config.RESOLUTION))
        image_to_draw_on = cv2.cvtColor(image_to_draw_on, cv2.COLOR_RGB2BGR)

        # --- THIS IS THE FIX: Load and scale ground-truth boxes directly ---
        annot_path = dataset.annotation_files[idx]
        boxes_xywh = np.load(annot_path, allow_pickle=True)
        gt_boxes = []
        for box in boxes_xywh:
            x_min, y_min, x_max, y_max = box['x'], box['y'], box['x'] + box['width'], box['y'] + box['height']
            gt_boxes.append([x_min, y_min, x_max, y_max])

        orig_h, orig_w, _ = center_color_frame_raw.shape
        scale_h = config.RESOLUTION / orig_h
        scale_w = config.RESOLUTION / orig_w
        
        for box in gt_boxes:
            x1, y1, x2, y2 = box
            # Scale the box coordinates
            scaled_x1, scaled_y1, scaled_x2, scaled_y2 = int(x1 * scale_w), int(y1 * scale_h), int(x2 * scale_w), int(y2 * scale_h)
            # Draw the scaled box in GREEN
            cv2.rectangle(image_to_draw_on, (scaled_x1, scaled_y1), (scaled_x2, scaled_y2), (0, 255, 0), 2)
        # -----------------------------------------------------------------

        # draw predicted boxes in RED
        for box, score in zip(outputs[0]['boxes'], outputs[0]['scores']):
            if score > CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image_to_draw_on, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"{score:.2f}"
                cv2.putText(image_to_draw_on, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # saving the result
        original_filename = os.path.basename(center_img_path)
        output_path = os.path.join(OUTPUT_VIS_DIR, original_filename)
        cv2.imwrite(output_path, image_to_draw_on)

    print(f"\nVisualisations saved to: {OUTPUT_VIS_DIR}")

if __name__ == '__main__':
    main()