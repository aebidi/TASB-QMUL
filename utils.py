# for helper fucntions
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

def collate_fn(batch):
    """
    Custom collate function for object detection.
    """
    return tuple(zip(*batch))


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=30, verbose=False, delta=0, path='best_model.pth', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation metric improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation metric improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
            path (str): Path for the checkpoint to be saved to.
                        Default: 'best_model.pth'
            trace_func (function): trace print function.
                                   Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metric_max = -np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
    def __call__(self, val_metric, model):
        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, model):
        """Saves model when validation metric improves."""
        if self.verbose:
            self.trace_func(f'Validation metric improved ({self.val_metric_max:.6f} --> {val_metric:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_metric_max = val_metric


def save_plots(train_losses, val_metrics_history, output_dir):
    """
    Generates and saves plots for training loss and validation metrics.
    """
    # Plot 1: Training Loss vs. Validation mAP@50
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    plt.title("Training Loss and Validation mAP@50 vs. Epochs")
    
    # Plot Training Loss on the primary y-axis
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color='tab:red')
    ax1.plot(train_losses, color='tab:red', label='Training Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Create a secondary y-axis for the mAP@50
    ax2 = ax1.twinx()
    val_map50s = [d['map_50'].item() for d in val_metrics_history]
    ax2.set_ylabel('Validation mAP@50', color='tab:blue')
    ax2.plot(val_map50s, color='tab:blue', label='Validation mAP@50')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    fig.tight_layout()
    # Add a single legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    
    plot_path1 = os.path.join(output_dir, 'loss_vs_map50.png')
    plt.savefig(plot_path1)
    print(f"Saved Loss vs. mAP@50 plot to {plot_path1}")
    plt.close()

    # Plot 2: Advanced Validation Metrics
    plt.figure(figsize=(12, 8))
    plt.title("Advanced Validation Metrics vs. Epochs")
    val_maps = [d['map'].item() for d in val_metrics_history]
    val_mars = [d['mar_100'].item() for d in val_metrics_history]
    
    plt.plot(val_maps, label='mAP @ 0.50:0.95 (COCO)', color='purple')
    plt.plot(val_mars, label='MAR @ 100 Dets', color='green')
    
    plt.xlabel('Epochs')
    plt.ylabel('Metric Score')
    plt.legend()
    plt.grid(True)
    
    plot_path2 = os.path.join(output_dir, 'advanced_metrics.png')
    plt.savefig(plot_path2)
    print(f"Saved advanced metrics plot to {plot_path2}")
    plt.close()