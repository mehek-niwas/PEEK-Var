# import os
# import numpy as np
# import matplotlib.pyplot as plt

# entropy_dir = "entropies"
# output_dir = "entropy_histograms"
# os.makedirs(output_dir, exist_ok=True)

# for filename in os.listdir(entropy_dir):
#     if filename.endswith(".npy"):
#         layer_name = filename.replace(".npy", "")
#         path = os.path.join(entropy_dir, filename)
#         entropy = np.load(path)

#         # Plot histogram
#         plt.figure(figsize=(8, 6))
#         plt.hist(entropy, bins=20, color='skyblue', edgecolor='black')
#         plt.title(f"Entropy Histogram - {layer_name}")
#         plt.xlabel("Entropy Value")
#         plt.ylabel("Frequency")
#         plt.grid(True)

#         # Save the histogram
#         save_path = os.path.join(output_dir, f"{layer_name}_hist.png")
#         plt.savefig(save_path)
#         plt.close()
#         print(f"Saved histogram for {layer_name} at {save_path}")


import os
import numpy as np
import matplotlib.pyplot as plt

entropy_dir = "entropies"
output_dir = "entropy_histograms"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(entropy_dir):
    if filename.endswith(".npy"):
        layer_name = filename.replace(".npy", "")
        path = os.path.join(entropy_dir, filename)
        entropy = np.load(path)

        # Compute stats
        avg = entropy.mean()
        var = entropy.var()
        min_val = entropy.min()
        max_val = entropy.max()

        # Plot histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(entropy, bins=20, color='skyblue', edgecolor='black')
        ax.set_title(f"Entropy Histogram - {layer_name}")
        ax.set_xlabel("Entropy Value")
        ax.set_ylabel("Frequency")

        # Remove grid
        # ax.grid(True)  ← removed this line

        # Add stats text box
        stats_text = f"Mean: {avg:.4f}\nVar: {var:.4f}\nMin: {min_val:.4f}\nMax: {max_val:.4f}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.9)
        ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=props)

        # Save the histogram
        save_path = os.path.join(output_dir, f"{layer_name}_hist.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved histogram for {layer_name} at {save_path}")
