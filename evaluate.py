import torch
import os
import config
from datetime import datetime

from temporal_attention_model import FPN_Temporal_Attention_FasterRCNN 

from dataset import IVUSSideBranchDataset
from augmentations import get_val_transforms
from train import evaluate # We can reuse the evaluate function from train.py
from utils import collate_fn

def main():
    # --- 1. Load Test Data ---
    test_dir = os.path.join(config.DATA_ROOT, 'test')
    
    # Use the simple validation transform
    dataset_test = IVUSSideBranchDataset(test_dir, transform=get_val_transforms())
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=config.BATCH_SIZE, # Use batch size from config
        shuffle=False,
        num_workers=config.NUM_WORKERS, 
        collate_fn=collate_fn
    )
    print(f"Found {len(dataset_test)} images in the test set.")
    
    # --- 2. Build and Load the CORRECT Model ---
    # --- THIS IS THE PRIMARY FIX ---
    print(f"INFO: Building the FPN Temporal Attention model for evaluation.")
    model = FPN_Temporal_Attention_FasterRCNN(
        num_classes=config.NUM_CLASSES,
        num_frames=(2 * config.TEMPORAL_FRAMES_K) + 1
    )
    # ------------------------------

    # Now, load the saved weights into the correct model structure
    model_path = os.path.join(config.OUTPUT_DIR, 'best_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.to(config.DEVICE)
    
    print("\n--- Starting Evaluation on Test Set ---")
    
    # --- 3. Run Evaluation ---
    test_metrics = evaluate(model, data_loader_test, device=config.DEVICE)
    
    print("\n--- Test Set Evaluation Finished ---")
    print(f"Final Test mAP@50: {test_metrics['map_50'].item():.4f}")

    # --- 4. SAVE THE CLASSIFICATION REPORT ---
    report_path = os.path.join(config.OUTPUT_DIR, 'classification_report.txt')
    print(f"\nSaving classification report to: {report_path}")
    with open(report_path, 'w') as f:
        f.write("--- FPN Temporal Attention Model (V9) Evaluation Report ---\n")
        f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model checkpoint: {model_path}\n")
        f.write(f"Temporal Frames (k): {config.TEMPORAL_FRAMES_K}\n")
        f.write("-" * 50 + "\n\n")
        
        for key, value in test_metrics.items():
            f.write(f"{key}: {value.item():.4f}\n")
            
    print("Report saved successfully.")

if __name__ == '__main__':
    main()