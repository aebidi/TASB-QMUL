# plot_testset_comparison.py

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # We need this for custom legend items
import numpy as np
import os

def plot_final_scores(model_data, output_dir):
    """
    Generates and saves a bar chart of final model test scores.
    """
    # unpack model names and scores from the dictionary
    names = list(model_data.keys())
    scores = list(model_data.values())
    
    # find the index of the best performing model to highlight it
    best_model_index = np.argmax(scores)
    
    # --- Create the plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # creating a list of colors, setting the best model to green
    colors = ['#1f77b4'] * len(scores) # Default blue color
    colors[best_model_index] = '#2ca02c' # Highlight best model in green
    
    # creating the bar chart
    bars = ax.bar(names, scores, color=colors)
    
    # --- Formatting and Labels ---
    plt.title("Model Performance Comparison on Test Set", fontsize=22, pad=20)
    ax.set_xlabel("(Model Version)", fontsize=16, labelpad=15)
    ax.set_ylabel("(Test mAP@50 Score)", fontsize=16, labelpad=15)
    
    # set the y-axis limit to create some space above the bars
    ax.set_ylim(0, max(scores) * 1.15)
    
    # add the score value text on top of each bar for clarity
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.4f}', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
                
    # create a manual legend item (a "patch") for the green bar
    best_model_patch = mpatches.Patch(color='#2ca02c', label='best performing model')

    # explicitly tell the legend function to use this item via the 'handles' argument
    # This directly solves the "called with no argument" warning
    ax.legend(handles=[best_model_patch], fontsize=14)
    # ------------------------------------
               
    plt.xticks(rotation=0, ha='center', fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.grid(axis='y', linestyle='--', linewidth=0.7)
    plt.tight_layout() # Adjust layout to make sure everything fits
    
    # --- Save the Plot ---
    output_path = os.path.join(output_dir, 'model_test_comparison_bar.png')
    plt.savefig(output_path, dpi=300)
    print(f"Final comparison bar chart saved successfully to: {output_path}")
    plt.show()

if __name__ == '__main__':
    # The data for your final models
    final_scores = {
        'V1\nBaseline': 0.7981,
        'V2\nBaseline\n(Aggressive Aug.)': 0.6786,
        'V3\nBaseline\n(Mild Aug.+LR Scheduler)': 0.7114,
        'V4\nBaseline\n(No Aug.+LR Scheduler)': 0.8184,
        'V5\nTemporal\n(Stacked)': 0.7837,
        'V6\nTemporal\n(Stacked+Attention)': 0.7861,
        'V7\nFPN Temporal\nAttention (k=1)': 0.8365,
        'V8\nFPN Temporal\nAttention (k=3)': 0.8314
    }
        
    # defining bar graph directory
    output_directory = 'results/plots' 
    os.makedirs(output_directory, exist_ok=True)
    
    # generate and save the plot
    plot_final_scores(final_scores, output_directory)