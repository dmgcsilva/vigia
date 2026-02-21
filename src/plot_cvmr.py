import json
import random
import os
import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_random_samples(json_file_path, num_plots=10):
    """
    Reads video moment retrieval data from a JSON file, randomly selects a
    specified number of samples, and generates a plot for each.

    Args:
        json_file_path (str): The path to the input JSON file.
        num_plots (int): The number of random plots to generate.
    """
    # --- 1. Load the Data ---
    try:
        data = torch.load(json_file_path, map_location='cpu', weights_only=False)
        # Convert tensors to lists for easier handling in plotting
        data = {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
        # Create a dummy file for demonstration purposes if it doesn't exist
        print("Creating a dummy 'data.json' file for you to test the script.")
        dummy_data = {
            "start_sim_matrixes": [list(np.exp(-((np.arange(100) - random.randint(10, 30))**2) / (2 * 5**2))) for _ in range(50)],
            "end_sim_matrixes": [list(np.exp(-((np.arange(100) - random.randint(40, 80))**2) / (2 * 5**2))) for _ in range(50)],
            "target_ids": [[random.randint(10, 30), random.randint(40, 80)] for _ in range(50)]
        }
        with open("data.json", 'w') as f:
            json.dump(dummy_data, f)
        print("Dummy 'data.json' created. Please run the script again.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{json_file_path}'. Please check its format.")
        return

    start_sim_matrixes = data.get("start_sim_matrixes")
    end_sim_matrixes = data.get("end_sim_matrixes")
    target_ids = data.get("target_ids", [(0,0) for _ in range(len(start_sim_matrixes))])  # Default to (0,0) if not provided
    predicted_ids = data.get("predictions", [(0,0) for _ in range(len(start_sim_matrixes))])  # Default to (0,0) if not provided

    # Validate data
    if not all([start_sim_matrixes, end_sim_matrixes, target_ids]):
        print("Error: JSON file must contain 'start_sim_matrixes', 'end_sim_matrixes', and 'target_ids' fields.")
        return

    num_samples = len(start_sim_matrixes)
    if num_samples < num_plots:
        print(f"Warning: Number of samples ({num_samples}) is less than the number of plots requested ({num_plots}).")
        print(f"Plotting all {num_samples} samples.")
        num_plots = num_samples

    # --- 2. Select Random Samples ---
    sample_indices = random.sample(range(num_samples), num_plots)
    sample_indices = [17, 63, 81, 88, 98, 119, 174]
    print(f"Randomly selected indices to plot: {sample_indices}")

    # --- 3. Create Output Directory ---
    output_dir = 'random_plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: '{output_dir}'")
    
    fontsize = 15
    linewidth = 3

    # --- 4. Generate and Save Plots ---
    for i in sample_indices:
        start_sim = start_sim_matrixes[i]
        end_sim = end_sim_matrixes[i]
        target_start, target_end = target_ids[i]
        pred_start, pred_end = predicted_ids[i]

        # Create the x-axis based on the length of the similarity vector
        x_axis = np.arange(len(start_sim))

        plt.figure(figsize=(10, 6))

        # Plot the similarity scores
        plt.plot(x_axis, start_sim, label='Start Similarity', color='blue', alpha=0.8, linewidth=linewidth)
        plt.plot(x_axis, end_sim, label='End Similarity', color='red', alpha=0.8, linewidth=linewidth)

        # Plot the ground truth vertical lines
        plt.axvline(x=target_start, color='green', linestyle='--', label=f'Ground Truth Start ({target_start})', linewidth=linewidth)
        plt.axvline(x=target_end, color='purple', linestyle='--', label=f'Ground Truth End ({target_end})', linewidth=linewidth)

        # Plot the predicted vertical lines
        # plt.axvline(x=pred_start, color='orange', linestyle=':', label=f'Predicted Start ({pred_start})', linewidth=linewidth)
        # plt.axvline(x=pred_end, color='brown', linestyle=':', label=f'Predicted End ({pred_end})', linewidth=linewidth)

        # Add labels, title, and legend
        plt.xlabel('Frame Position', fontsize=fontsize)
        plt.ylabel('Similarity Score', fontsize=fontsize)
        plt.title(f'Sample {i} - Similarity across Frames', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xlim(0, len(x_axis) - 1) # Ensure x-axis starts at 0
        plt.ylim(0, max(0.5, max(max(start_sim), max(end_sim)) * 1.05)) # Dynamic y-axis limit

        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        # Save the plot to the output directory
        plot_filename = os.path.join(output_dir, f'sample_plot_{i}.png')
        plt.savefig(plot_filename)
        plt.close() # Close the figure to free up memory

    print(f"\nSuccessfully generated and saved {num_plots} plots in the '{output_dir}' directory.")


if __name__ == '__main__':
    # --- IMPORTANT ---
    # Change this to the name of your JSON file
    json_file_path = 'path_to_the_sim_matrixes_file.pth'
    plot_random_samples(json_file_path, num_plots=100)