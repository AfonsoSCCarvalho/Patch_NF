import numpy as np


from Quality_indices_pytorch import quality_indices
import matplotlib.pyplot as plt
from AuxFunctions import *

def FinalResults(HS, HS_np, MS, MS_np, Img_reconstructed, X0, X0_np, target_shape):

    quality_indices(Img_reconstructed, X0)



    Img_reconstructed_np = Img_reconstructed.detach().cpu().numpy()

    # plot_img(Img_reconstructed_np, "Img_reconstructed")


    # Define the bands of interest for visualization
    num_bands = HS.shape[2]
    band_set_HS = [min(45, num_bands-1), min(25, num_bands-1), min(8, num_bands-1)]

    # Define a normalization function for display purposes
    normColor = lambda R: np.clip((R - np.mean(R)) / np.std(R), -2, 2) / 3 + 0.5

    # Create a figure with two rows and two columns
    fig, axs = plt.subplots(2, 3, figsize=(10, 10))

    # Plot HS in the top left subplot
    temp_show_HS = HS_np[:, :, band_set_HS]
    temp_show_HS = normColor(temp_show_HS)
    axs[0, 0].imshow(temp_show_HS)
    axs[0, 0].set_title("Hyper spectral image")
    axs[0, 0].axis('off')

    num_bands = MS.shape[2]
    band_set_HS = [min(45, num_bands-1), min(25, num_bands-1), min(8, num_bands-1)]

    # Plot MS in the top right subplot
    temp_show_MS = MS_np[:, :, band_set_HS]
    temp_show_MS = normColor(temp_show_MS)
    axs[0, 1].imshow(temp_show_MS)
    axs[0, 1].set_title("Multi Spectral image")
    axs[0, 1].axis('off')

    upscaled_hs_image_torch = upscale_hs_image(HS, target_shape)

    upscale_hs_image_np = upscaled_hs_image_torch.cpu().numpy()
    # Define the bands of interest for Z_3d and X0
    num_bands_upscaled_hs_image = upscale_hs_image_np.shape[2]
    band_set_upscaled_hs_image = [min(45, num_bands_upscaled_hs_image-1), min(25, num_bands_upscaled_hs_image-1), min(8, num_bands_upscaled_hs_image-1)]

    # Plot X0 in the bottom right subplot
    temp_show_upscale_hs_image_np = upscale_hs_image_np[:, :, band_set_upscaled_hs_image]
    temp_show_upscale_hs_image_np = normColor(temp_show_upscale_hs_image_np)
    axs[0, 2].imshow(temp_show_upscale_hs_image_np)
    axs[0, 2].set_title("Simple Cubic interpolation of HS")
    axs[0, 2].axis('off')


    # Define the bands of interest for Z_3d and X0
    num_bands_ZX0 = Img_reconstructed.shape[2]
    band_set_ZX0 = [min(45, num_bands_ZX0-1), min(25, num_bands_ZX0-1), min(8, num_bands_ZX0-1)]

    # Plot Z_3d in the bottom left subplot
    temp_show_Z = Img_reconstructed_np[:, :, band_set_ZX0]
    temp_show_Z = normColor(temp_show_Z)
    axs[1, 0].imshow(temp_show_Z)
    axs[1, 0].set_title("Img reconstructed with gradient descent")
    axs[1, 0].axis('off')

    # Define the bands of interest for Z_3d and X0
    num_bands_X0 = X0_np.shape[2]
    band_set_X0 = [min(45, num_bands_X0-1), min(25, num_bands_X0-1), min(8, num_bands_X0-1)]

    # Plot Z_3d in the bottom left subplot
    temp_show_X0 = X0_np[:, :, band_set_X0]
    temp_show_X0 = normColor(temp_show_X0)
    axs[1, 2].imshow(temp_show_X0)
    axs[1, 2].set_title("Original Image")
    axs[1, 2].axis('off')

    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


    # Create a figure with two rows and two columns
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))

    # Define the bands of interest for Z_3d and X0
    num_bands_ZX0 = Img_reconstructed.shape[2]
    band_set_ZX0 = [min(45, num_bands_ZX0-1), min(25, num_bands_ZX0-1), min(8, num_bands_ZX0-1)]

    # Plot Z_3d in the bottom left subplot
    temp_show_Z = Img_reconstructed_np[:, :, band_set_ZX0]
    temp_show_Z = normColor(temp_show_Z)
    axs[ 0].imshow(temp_show_Z)
    axs[ 0].set_title("Img reconstructed with gradient descent")
    axs[0].axis('off')

    # Define the bands of interest for Z_3d and X0
    num_bands_X0 = X0_np.shape[2]
    band_set_X0 = [min(45, num_bands_X0-1), min(25, num_bands_X0-1), min(8, num_bands_X0-1)]

    # Plot Z_3d in the bottom left subplot
    temp_show_X0 = X0_np[:, :, band_set_X0]
    temp_show_X0 = normColor(temp_show_X0)
    axs[1].imshow(temp_show_X0)
    axs[1].set_title("Original Image")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

