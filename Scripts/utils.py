# This code was inspired on the paper
#
# F. Altekrüger, A. Denker, P. Hagemann, J. Hertrich, P. Maass and G. Steidl (2023).
# PatchNR: Learning from Very Few Images by Patch Normalizing Flow Regularization.
# Inverse Problems, vol. 39, no. 6.
#
# 
# and 
# 
# A. Houdard, A. Leclaire, N. Papadakis and J. Rabin. 
# Wasserstein Generative Models for Patch-based Texture Synthesis. 
# ArXiv Preprint#2007.03408
# (https://github.com/ahoudard/wgenpatex)
import scipy.io as sio
import torch
from torch import nn
import skimage.io as io
import numpy as np
import math
import torch.nn.functional as F

from Plotter_torch import *


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_mat_variable(file_path, var_name):
    """
    Loads a specified variable from a .mat file.

    Parameters:
    file_path (str): The path to the .mat file.
    var_name (str): The name of the variable in the .mat file to be loaded.

    Returns:
    numpy.ndarray: The loaded variable as a NumPy array.
    """
    mat_data = sio.loadmat(file_path)
    if var_name in mat_data:
        return mat_data[var_name]
    else:
        raise ValueError(f"Variable '{var_name}' not found in the file {file_path}")
    


def imread(img_name):
    """
    Loads an image file as a torch tensor on the selected device.
    
    Args:
        img_name (str): The name or path of the image file.
        
    Returns:
        torch.Tensor: The image as a torch tensor.
    """
    # Read the image using scikit-image's imread
    np_img = io.imread(img_name)
    
    # Convert the NumPy array to a torch tensor
    tens_img = torch.tensor(np_img, dtype=torch.float, device=DEVICE)
    
    # Normalize pixel values if necessary
    if torch.max(tens_img) > 1:
        tens_img /= 255
    
    # Add an additional dimension for a single color channel if needed
    if len(tens_img.shape) < 3:
        tens_img = tens_img.unsqueeze(2)
    
    # Select the first three channels if more than three channels are present
    if tens_img.shape[2] > 3:
        tens_img = tens_img[:, :, :3]
    
    # Permute the dimensions to match the expected format (channels, height, width)
    tens_img = tens_img.permute(2, 0, 1)
    
    # Add an additional dimension to represent a batch of size 1
    return tens_img.unsqueeze(0)



def save_img(tensor_img, name):
    """
    Saves an image in tensor form to a file with the specified name.
    
    Args:
        tensor_img (torch.Tensor): The image as a tensor.
        name (str): The name of the saved image file (without extension).
    """
    # Convert the tensor image to a NumPy array and clip pixel values between 0 and 1
    img = np.clip(tensor_img.squeeze().detach().cpu().numpy(), 0, 1)
    
    # Save the image as a PNG file
    io.imsave(str(name) + '.png', img)
       

class gaussian_downsample(nn.Module):
    """
    Downsampling module with Gaussian filtering.
    """
    def __init__(self, kernel_size, sigma, stride, pad=False):
        """
        Initializes the gaussian_downsample module.

        Args:
            kernel_size (int): Size of the Gaussian kernel.
            sigma (float): Standard deviation of the Gaussian distribution.
            stride (int): Stride value for the convolution operation.
            pad (bool, optional): Flag indicating whether padding should be applied. Defaults to False.
        """
        super(gaussian_downsample, self).__init__()

        # Create a 2D convolutional layer with 1 input channel, 1 output channel, and specified kernel size and stride
        self.gauss = nn.Conv2d(1, 1, kernel_size, stride=stride, groups=1, bias=False)

        # Initialize the Gaussian kernel weights
        gaussian_weights = self.init_weights(kernel_size, sigma)

        # Assign the Gaussian kernel weights to the convolutional layer and make them non-trainable
        self.gauss.weight.data = gaussian_weights.to(DEVICE)
        self.gauss.weight.requires_grad_(False)

        self.pad = pad
        self.padsize = kernel_size - 1

    def forward(self, x):
        """
        Performs forward pass through the gaussian_downsample module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after Gaussian downsampling.
        """
        if self.pad:
            # Apply padding to the input tensor
            x = torch.cat((x, x[:, :, :self.padsize, :]), 2)
            x = torch.cat((x, x[:, :, :, :self.padsize]), 3)

        # Apply Gaussian downsampling using the convolutional layer
        return self.gauss(x)

    def init_weights(self, kernel_size, sigma):
        """
        Initializes the weights of the Gaussian kernel.

        Args:
            kernel_size (int): Size of the Gaussian kernel.
            sigma (float): Standard deviation of the Gaussian distribution.

        Returns:
            torch.Tensor: Initialized Gaussian kernel weights.
        """
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1) / 2.
        variance = sigma**2.
        gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2 * variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        return gaussian_kernel.view(1, 1, kernel_size, kernel_size)




class patch_extractor(nn.Module):
    """
    Module for creating a custom patch extractor.
    """
    def __init__(self, patch_size, stride_wanted= 1, pad=False, center=False):
        """
        Initializes the patch_extractor module.

        Args:
            patch_size (int): Size of the patches to be extracted.
            pad (bool, optional): Flag indicating whether padding should be applied. Defaults to False.
            center (bool, optional): Flag indicating whether the patches should be centered. Defaults to False.
        """
        super(patch_extractor, self).__init__()

        # Create an Unfold operation with the specified patch size
        self.im2pat = nn.Unfold(kernel_size=patch_size, stride=stride_wanted)

        self.pad = pad
        self.padsize = patch_size - 1
        self.center = center
        self.patch_size = patch_size

    def forward(self, input, batch_size=0, random_interval=True, interval_start=0, interval_end=1, frequency_selection=False, freq_range = 0 ,  gradient_selection=False, gradient_threshold=0.5):
        """
        Performs forward pass through the patch_extractor module.

        Args:
            input (torch.Tensor): Input tensor.
            batch_size (int, optional): Size of the batch to select random patches. Defaults to 0 (no batch selection).
            random_interval (bool, optional): Flag indicating whether to select patches within a random interval. Defaults to False.
            interval_start (float, optional): Start of the random interval for patch selection. Value should be between 0 and 1. Defaults to 0.
            interval_end (float, optional): End of the random interval for patch selection. Value should be between 0 and 1. Defaults to 1.

        Returns:
            torch.Tensor: Extracted patches.
        """
        if self.pad:
            # Apply padding to the input tensor
            input = torch.cat((input, input[:, :, :self.padsize, :]), 2)
            input = torch.cat((input, input[:, :, :, :self.padsize]), 3)

        # Unfold the input image into patches
        patches = self.im2pat(input).squeeze(0).transpose(1, 0)
        # print(patches.shape)

        if random_interval == False:
            # Select patches within a random interval
            interval_start_idx = int(patches.size(0) * interval_start)
            interval_end_idx = int(patches.size(0) * interval_end)
            patches = patches[interval_start_idx:interval_end_idx, :]
        # print(patches.shape)

        if frequency_selection:
            contour_threshold = 2
            # Calculate the high frequency measure for each patch
            high_freq_measure = torch.sum(torch.abs(patches[:, 1:] - patches[:, :-1]), dim=1)

            if freq_range == 0:  
                # Select patches with high frequency (contours)
                patches = patches[high_freq_measure > contour_threshold, :]
            if freq_range == 1:  
                # Select patches with low frequency 
                patches = patches[high_freq_measure < contour_threshold, :]


            # # Calculate the mean frequency value
            # mean_freq = torch.mean(high_freq_measure)
            # # print(mean_freq_1)
            # # print(mean_freq_2)

            if 0:
                # Plot the frequency measure
                plt.figure()
                plt.plot(high_freq_measure.cpu().numpy(), label='Patch Frequency 1')
                plt.axhline(contour_threshold, color='r', linestyle='--', label='Contour Threshold')
                plt.xlabel('Patch Index')
                plt.ylabel('Frequency Measure')
                plt.legend()
                plt.title('Frequency Measure of Patches')
                plt.show()

        if gradient_selection:
            # Apply the Sobel operator to calculate the gradient magnitude for each patch
            sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
            gradient_x = F.conv2d(input, sobel_x)
            gradient_y = F.conv2d(input, sobel_y)
            gradient_magnitude = torch.sqrt(gradient_x.pow(2) + gradient_y.pow(2))

            # Calculate the average gradient magnitude for each patch
            average_gradient_magnitude = torch.mean(gradient_magnitude.view(gradient_magnitude.size(0), -1), dim=1)

            # Select patches with average gradient magnitudes above the threshold
            selected_patches = patches[average_gradient_magnitude > gradient_threshold, :]
            patches = selected_patches

        if batch_size > 0:
            # Select a random subset of patches if batch_size is specified
            idx = torch.randperm(patches.size(0))[:batch_size]
            patches = patches[idx, :]

        if self.center:
            # Center the patches by subtracting their mean
            patches = patches - torch.mean(patches, -1).unsqueeze(-1)
            
        plot_it = False
        if plot_it:

            num_patches_to_plot = 2

            # Randomly select some patches to plot
            random_patch_indices = torch.randint(high=patches.size(0), size=(num_patches_to_plot,))

            print(random_patch_indices.shape)
            # Plot the selected patches
            for idx in random_patch_indices:
                patch = patches[idx].reshape((input.size(1), self.patch_size, self.patch_size))
                patch = patch.permute(1, 2, 0)  # Reshape to (patch_size, patch_size, num_channels)
                plot_imgT(patch.cpu(),"Patch number " + str(idx))

        return patches


class patch_extractor_2(nn.Module):
    """
    Module for creating a custom patch extractor.
    """
    def __init__(self, patch_size, stride_wanted= 1, pad=False, center=False):
        """
        Initializes the patch_extractor module.

        Args:
            patch_size (int): Size of the patches to be extracted.
            pad (bool, optional): Flag indicating whether padding should be applied. Defaults to False.
            center (bool, optional): Flag indicating whether the patches should be centered. Defaults to False.
        """
        super(patch_extractor_2, self).__init__()

        # Create an Unfold operation with the specified patch size
        self.im2pat = nn.Unfold(kernel_size=patch_size, stride=stride_wanted)

        self.pad = pad
        self.padsize = patch_size - 1
        self.center = center
        self.patch_size = patch_size

    def forward(self, input_1,input_2,  batch_size=0, high_frequency_selection=False):
        """
        Performs forward pass through the patch_extractor module.

        Args:
            input (torch.Tensor): Input tensor.
            batch_size (int, optional): Size of the batch to select random patches. Defaults to 0 (no batch selection).

        Returns:
            torch.Tensor: Extracted patches.
        """
        if self.pad:
            # Apply padding to the input tensor
            input_1 = torch.cat((input_1, input_1[:, :, :self.padsize, :]), 2)
            input_1 = torch.cat((input_1, input_1[:, :, :, :self.padsize]), 3)

            input_2 = torch.cat((input_2, input_2[:, :, :self.padsize, :]), 2)
            input_2 = torch.cat((input_2, input_2[:, :, :, :self.padsize]), 3)

        # Unfold the input image into patches
        patches_1 = self.im2pat(input_1).squeeze(0).transpose(1, 0)
        patches_2 = self.im2pat(input_2).squeeze(0).transpose(1, 0)

        if high_frequency_selection:
            contour_threshold = 3
            # Calculate the high frequency measure for each patch
            high_freq_measure_1 = torch.sum(torch.abs(patches_1[:, 1:] - patches_1[:, :-1]), dim=1)
            high_freq_measure_2 = torch.sum(torch.abs(patches_2[:, 1:] - patches_2[:, :-1]), dim=1)

            # Select patches with high frequency (contours)
            patches_1 = patches_1[high_freq_measure_1 > contour_threshold, :]
            patches_2 = patches_2[high_freq_measure_2 > contour_threshold, :]
            
            # Calculate the mean frequency value
            mean_freq_1 = torch.mean(high_freq_measure_1)
            mean_freq_2 = torch.mean(high_freq_measure_2)
            # print(mean_freq_1)
            # print(mean_freq_2)

            if 1:
                # Plot the frequency measure
                plt.figure()
                plt.plot(high_freq_measure_1.cpu().numpy(), label='Patch Frequency 1')
                plt.plot(high_freq_measure_2.cpu().numpy(), label='Patch Frequency 2')
                plt.axhline(contour_threshold, color='r', linestyle='--', label='Contour Threshold')
                plt.xlabel('Patch Index')
                plt.ylabel('Frequency Measure')
                plt.legend()
                plt.title('Frequency Measure of Patches')
                plt.show()

        if batch_size > 0:
            # Select a random subset of patches if batch_size is specified
            idx = torch.randperm(patches_1.size(0))[:batch_size]
            patches_1 = patches_1[idx, :]
            patches_2 = patches_2[idx, :]


        if self.center:
            # Center the patches by subtracting their mean
            patches_1 = patches_1 - torch.mean(patches_1, -1).unsqueeze(-1)
            patches_2 = patches_2 - torch.mean(patches_2, -1).unsqueeze(-1)


        plot_it = False
        if plot_it:

            num_patches_to_plot = 5

            # Randomly select some patches to plot
            idx_s = torch.randperm(patches_1.size(0))[:batch_size]

            # print(random_patch_indices.shape)
            # Plot the selected patches
            for idx in idx_s:
                patch = patches_1[idx].reshape((input_1.size(1), self.patch_size, self.patch_size))
                patch = patch.permute(1, 2, 0)  # Reshape to (patch_size, patch_size, num_channels)
                plot_imgT(patch.cpu(),"Patch number " + str(idx))

        return patches_1, patches_2

