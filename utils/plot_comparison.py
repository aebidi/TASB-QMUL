# plot_comparison.py

import os
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_map_from_log(log_file_path):
    """
    Parses a training log file to extract the validation mAP@50 for each epoch.
    """
    map_scores = []
    # this regular expression looks for the specific line format in your logs
    # it captures the floating-point number after "Validation mAP@50:"
    regex = r"Validation mAP@50: (\d+\.\d+)"
    
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                match = re.search(regex, line)
                if match:
                    # convert the captured string to a float and add to our list
                    map_scores.append(float(match.group(1)))
    except FileNotFoundError:
        print(f"Warning: Log file not found at {log_file_path}. Skipping.")
        return []
        
    return map_scores

def main():
    # --- 1. Define your experiments ---
    # update this dictionary with the correct names and paths to your log files
    # the key is the label that will appear in the plot legend
    # the value is the path to the log file
    experiments_with_logs = {
        'V2: Baseline (Aggressive Augmentations)': 'results/training.log',
        'V3: Baseline (Mild Augmentations+LR Scheduler)': 'results/training_log_v3.log',
        'V4: Baseline (No Augmentations+LR Scheduler)': 'results/training_v4_no_augment.log',
        'V5: Temporal (Stacked, k=5)': 'results/training_v5_no_augment.log',
        'V6: Temporal (Stacked+Attention, k=5)': 'results/training_v6_no_augment.log',
        'V7: FPN Temporal Attention (k=1)': 'results/training_v7_no_augment.log',
        'V8: FPN Temporal Attention (k=3)': 'results/training_v8_no_augment.log'
    }
    
    # Models WITHOUT training logs (we only have their peak performance)
    # Format: 'Model Name': (peak_validation_map50, total_epochs_trained)
    # FROM OUR PREVIOUS RESULTS, V1's peak validation mAP was 0.7901.
    experiments_without_logs = {
        'V1: Baseline (Single-Frame)': (0.7901, 20)
    }


    # --- 2. Parse log files ---
    results_from_logs = {}
    for name, path in experiments_with_logs.items():
        print(f"Parsing {name} from {path}...")
        scores = parse_map_from_log(path)
        if scores:
            results_from_logs[name] = scores
            
    # --- 3. Generate the comparison plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(18, 10))
    plt.title("Model Performance Comparison: Validation mAP@50 vs. Epoch", fontsize=20)
    
    # defining a list of colours and line styles to cycle through
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#1f77b4', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    
    max_epochs = 0


    # plot models with full logs first (the curves)
    for i, (name, scores) in enumerate(results_from_logs.items()):
        epochs = range(1, len(scores) + 1)
        if len(epochs) > max_epochs:
            max_epochs = len(epochs)
        
        plt.plot(
            epochs, 
            scores, 
            label=f"{name} (Peak: {max(scores):.4f})", 
            color=colors[i],
            linestyle=linestyles[i],
            linewidth=2.5,
            zorder=10 # Draw curves on top
        )
        peak_epoch = np.argmax(scores)
        plt.plot(
            peak_epoch + 1, 
            scores[peak_epoch], 
            'o', 
            color=colors[i],
            markersize=10,
            markeredgecolor='black',
            zorder=10
        )

    # plot models without logs (horizontal lines)
    for i, (name, (peak_score, total_epochs)) in enumerate(experiments_without_logs.items()):
        color_index = len(results_from_logs) + i
        
        # plot a horizontal dashed line representing the peak performance
        plt.axhline(
            y=peak_score, 
            color=colors[color_index % len(colors)], 
            linestyle=':', 
            linewidth=2.5,
            label=f"{name} (Peak: {peak_score:.4f})"
        )
        # Plot a star at the end to indicate where its training stopped
        plt.plot(
            total_epochs, 
            peak_score, 
            '*', 
            color=colors[color_index % len(colors)],
            markersize=18,
            markeredgecolor='black',
            label=f'{name} Endpoint'
        )

    # --- formatting ---
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Validation mAP@50", fontsize=16)
    
    # creating a smart legend that avoids duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=14)
    
    if max_epochs == 0 and experiments_without_logs:
        max_epochs = max(v[1] for v in experiments_without_logs.values())

    plt.xticks(np.arange(0, max_epochs + 2, 2), fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    # --- 4. Save the plot ---
    output_path = 'results/plots/model_training_comparison.png'
    plt.savefig(output_path, dpi=300)
    print(f"\nComparison plot saved successfully to: {output_path}")
    plt.show()


if __name__ == '__main__':
    main()