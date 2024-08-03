import numpy as np
import sys
# caution: path[0] is reserved for script path (or '' in REPL)

from Quality_indices_pytorch import quality_indices_torch
import matplotlib.pyplot as plt
import torch



def plot_img(image, title):
    # Define the bands of interest for visualization
    num_bands = image.shape[2]
    band_set = [min(45, num_bands - 1), min(25, num_bands - 1), min(8, num_bands - 1)]

    # Define a normalization function for display purposes
    normColor = lambda R: np.clip((R - np.mean(R)) / np.std(R), -2, 2) / 3 + 0.5

    # Plot the image
    fig, ax = plt.subplots()
    temp_show = image[:, :, band_set]
    temp_show = normColor(temp_show)
    ax.imshow(temp_show)
    # ax.set_title(title)
    ax.axis('off')
    plt.show()

def upscale_hs_image(hs_image, target_shape):
    # Get the target dimensions
    target_height, target_width = target_shape[:2]
    # print(hs_image.shape)
    # Get the current dimensions of the HS image
    hs_height, hs_width = hs_image.shape[:2]

    # Compute the scaling factors for width and height
    scale_x = target_width / hs_width
    scale_y = target_height / hs_height

    # Reshape tensor to match PyTorch's expected input shape
    hs_image = hs_image.permute(2, 0, 1).unsqueeze(0).float()
    # print(hs_image.shape)

    # Perform interpolation using bilinear interpolation
    upscaled_hs_tensor = torch.nn.functional.interpolate(hs_image, size=(target_height, target_width), mode='bicubic', align_corners=False)

    # Convert tensor back to numpy array
    upscaled_hs_image = upscaled_hs_tensor.squeeze(0).permute(1, 2, 0)
    # print(upscaled_hs_image.shape)
    return upscaled_hs_image

def apply_subsampling_inverse(Y, subsamp):
    rows, cols, channels = Y.shape
    output_rows = rows * subsamp
    output_cols = cols * subsamp

    # Create an output tensor with increased dimensions
    output = torch.zeros((output_rows, output_cols, channels), dtype=Y.dtype)

    # Assign values from Y to the output tensor
    output[::subsamp, ::subsamp, :] = Y

    return output



def plot_results(reg_terms, reg_values, error_list, special_name=''):
    # Determine the number of rows and columns for the subplots grid
    num_rows = 3
    num_cols = 3

    # Create a grid of subplots for regularization terms
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12))

    # Plot each regularization term in a separate subplot
    for i in range(len(reg_terms)):
        row = i // num_cols
        col = i % num_cols
        axs[row, col].plot(range(len(reg_terms[i])), reg_terms[i])
        axs[row, col].set_xlabel('Optimization Steps')
        axs[row, col].set_ylabel('Regularization Term')
        axs[row, col].set_title(f'Regularization Term {i+1}')

    # Adjust the spacing between subplots
    fig.tight_layout()

    # Show the plot
    plt.show()

    # Plot the evolution of reg
    plt.plot(range(0, len(reg_values), 1), reg_values)
    plt.xlabel('Iteration')
    plt.ylabel('reg')
    plt.title('Evolution of reg')
    output_file = "Graphs_Kernels/plots_REG_" + special_name + ".png"  # Specify the file path relative to the output folder
    plt.savefig(output_file, dpi=300)
    plt.show()

    # Plot the error values
    plt.plot(range(0, len(error_list) * 10, 10), error_list)
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.title('Error Progress')
    output_file = "Graphs_Kernels/plots_MSE_" + special_name + ".png"  # Specify the file path relative to the output folder
    plt.savefig(output_file, dpi=300)
    plt.show()

# Call the function in the main block after the optimization loop
# ...
# plot_results(reg_terms, Z_3d, elapsed_time, reg_values, error_list, special_name='')




def main():
    Z_3d = torch.load("Z_3d_bestsofar.pt")
    Z_true = torch.load('Tensors/Z_true.pt')
    Z_true = torch.reshape(Z_true, (256, 256, Z_3d.shape[2])).detach()
    num_spectra = Z_3d.shape[2]
    num_rows = 3
    num_cols = 3

    fig, axs = plt.subplots(num_rows, 2 * num_cols, figsize=(12, 12))

    # Define a normalization function for display purposes
    normColor = lambda R: np.clip((R - np.mean(R)) / np.std(R), -2, 2) / 3 + 0.5

    Z_3d_plot = Z_3d.cpu().detach().numpy()
    Z_true_plot = Z_true.cpu().detach().numpy()

    for i in range(num_rows):
        for j in range(num_cols):
            if i * num_cols + j < num_spectra:
                Z_3d_plot_single = Z_3d_plot[:, :, i * num_cols + j]
                Z_true_plot_single = Z_true_plot[:, :, i * num_cols + j]

                axs[i, j*2].imshow(normColor(Z_true_plot_single), cmap='gray')
                axs[i, j*2].axis('off')

                axs[i, j*2+1].imshow(normColor(Z_3d_plot_single), cmap='gray')
                axs[i, j*2+1].axis('off')
                
    # Add titles to the subplots
    for i in range(num_rows):
        for j in range(num_cols):
            if i * num_cols + j < num_spectra:
                axs[i, j*2].set_title(f'Spectrum {i * num_cols + j + 1} (Z_true)')
                axs[i,  j*2+1].set_title(f'Spectrum {i * num_cols + j + 1} (Z_3d)')

    plt.tight_layout()
    plt.show()






if __name__ == "__main__":
    main()
