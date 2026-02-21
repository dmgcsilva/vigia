import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os 


model_files = {
    "VIGiA": "path/to/vigia_results.json",
}


# --- Helper Function ---
def load_and_process_data(file_path):
    """Loads data from JSON, extracts, and sorts relevant points."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred reading {file_path}: {e}")
        return None

    plot_data_points = []
    for key, metrics in data.items():
        if 'cvmr_sim' not in key:
            continue
        try:
            threshold = float(key.split('@')[1])
            # Ensure the required keys exist in the nested dictionary
            if 'R@1_IoU@0.5' in metrics and 'R@1_IoU@0.7' in metrics:
                 recall_iou_05 = metrics['R@1_IoU@0.5']
                 recall_iou_07 = metrics['R@1_IoU@0.7']
                 plot_data_points.append((threshold, recall_iou_05, recall_iou_07))
            else:
                print(f"Warning: Missing R@1 keys in '{key}' for file {file_path}")

        except (ValueError, IndexError, TypeError):
            print(f"Warning: Could not parse key '{key}' or data in file {file_path}")
            continue

    if not plot_data_points:
        print(f"Warning: No valid data points extracted from {file_path}")
        return None

    # Sort by threshold
    plot_data_points.sort(key=lambda x: x[0])

    # Unzip
    thresholds = [point[0] for point in plot_data_points]
    recalls_iou_05 = [point[1] for point in plot_data_points]
    recalls_iou_07 = [point[2] for point in plot_data_points]

    return thresholds, recalls_iou_05, recalls_iou_07


# --- Plotting ---

# 1. Set Seaborn Style (at the beginning)
sns.set_theme(style="whitegrid") # Options: "whitegrid", "darkgrid", "ticks", "white", "dark"

# 2. Create the figure
plt.figure(figsize=(20, 10)) # Adjust size as needed

# 3. Get a color palette from Seaborn
#    Adjust 'n_colors' if you have more models than the default palette size
palette = sns.color_palette(n_colors=len(model_files))

# 4. Loop through models, load data, and plot
for i, (model_name, file_path) in enumerate(model_files.items()):
    print(f"Processing: {model_name} from {file_path}")

    processed_data = load_and_process_data(file_path)

    if processed_data:
        thresholds, recalls_05, recalls_07 = processed_data
        model_color = palette[i % len(palette)] # Cycle through colors

        # Plot R@1 IoU@0.5 (Solid Line)
        plt.plot(thresholds, recalls_05,
                 marker='o',
                 linestyle='-',
                 color=model_color,
                 label=f'{model_name} (IoU@0.5)',
                 linewidth=5,
                 markersize=15)

        # Plot R@1 IoU@0.7 (Dashed Line)
        plt.plot(thresholds, recalls_07,
                 marker='x',
                 linestyle='--',
                 color=model_color,
                 label=f'{model_name} (IoU@0.7)',
                 linewidth=5,
                 markersize=15)
    else:
        print(f"Skipping plot for {model_name} due to data loading issues.")


# 5. Customize the final plot
fontsize=50
plt.xlabel("Similarity Threshold", fontsize=fontsize)
plt.ylabel("Recall@1", fontsize=fontsize)
plt.title("Validation set Recall@1 vs. Similarity Threshold", fontsize=fontsize)

# Set x-axis ticks at 0.1 intervals from 0.0 to 1.0
x_ticks = np.arange(0.0, 1.01, 0.1)
plt.xticks(x_ticks, fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlim(min(x_ticks), max(x_ticks)) # Optional: Adjust x-axis limits slightly

# Add legend (adjust placement if needed)
plt.legend(title="", fontsize=fontsize)
# Example for placing legend outside:
# plt.legend(title="Model & IoU Threshold", bbox_to_anchor=(1.02, 1), loc='upper left')

plt.tight_layout() # Adjust layout to prevent labels overlapping

# 6. Show the plot
plt.savefig("recall_vs_threshold_8bonly.png") # Save the figure
