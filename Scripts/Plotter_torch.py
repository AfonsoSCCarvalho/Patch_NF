import numpy as np
import torch
from matplotlib import pyplot as plt

def plot_imgT(image, title):
    # Define the bands of interest for visualization
    num_bands = image.shape[2]
    band_set = [min(45, num_bands-1), min(25, num_bands-1), min(8, num_bands-1)]

    # Define a normalization function for display purposes
    normColor = lambda R: torch.clip((R - torch.mean(R)) / torch.std(R), -2, 2) / 3 + 0.5

    # Plot the image
    fig, ax = plt.subplots()
    temp_show = image[:, :, band_set]
    temp_show = normColor(temp_show)
    ax.imshow(temp_show)
    ax.set_title(title)
    ax.axis('off')
    plt.show()

# Example usage
# HS0 = torch.randn(256, 256, 93)  # Example HS0 tensor
# plot_img(HS0, "HS image")
