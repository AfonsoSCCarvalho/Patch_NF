from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import random

from utils import *


def plot_spectrograms(patches, inversions):
    # Create subplots for original and inverted histograms
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plot the original histogram
    axes[0].hist(patches, bins=300, density=True, alpha=0.5)
    axes[0].set_title('Original Patch')

    # Plot the inverted histogram
    axes[1].hist(inversions, bins=300, density=True, alpha=0.5)
    axes[1].set_title('Inverted Patch')

    # Add labels and title
    fig.suptitle('Histogram Comparison')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()



import cv2
import numpy as np
from scipy.signal import periodogram


def calculate_blurriness(image):
    # Convert the image to grayscale
    gray = image

    # Perform Fourier transform on the grayscale image
    f = np.fft.fft2(gray)

    # Shift the zero-frequency component to the center
    fshift = np.fft.fftshift(f)

    # Calculate the magnitude spectrum of the shifted Fourier transform
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # Compute the periodogram of the flattened magnitude spectrum
    freqs, power = periodogram(magnitude_spectrum.flatten(), fs=1.0)

    # Calculate blurriness metric, e.g., mean power in low-frequency region
    low_freq_power = np.mean(power[:10])  # Adjust the frequency range as per your requirement

    return low_freq_power



def train_patchNR(patchNR, img, patch_size, steps, batch_size, center, special_name, plot_loss_train = False):
    """
    Train the patchNR for the given img (low resolution)
    """
    # Initialize an empty list to store the loss values
    loss_values = []
    # Set the batch size and optimizer steps
    batch_size = batch_size
    optimizer_steps = steps
    # Set the center and create an Adam optimizer for the PatchNR model
    center = center 
    # optimizer = torch.optim.Adam(patchNR.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam(patchNR.parameters(), lr=1e-4)
    # Create a patch extractor
    im2patch = patch_extractor(patch_size=patch_size, pad = False, center=center)
    # Initialize an empty tensor to store the training patches
    patches = torch.empty(0, device=DEVICE)


    patches = im2patch(img) #no transformation
    
    # Enlarge training patches by rotation and mirroring # commented case it did not make sense in our case
    # for j in range(2): 
    #     if j == 0:
    #         tmp = img
    #     elif j == 1:
    #         tmp = torch.flip(img, [1])
    #     for i in range(4):
    #         # Extract patches from the image and concatenate them to the patches tensor
    #         print(patches.shape)
    #         print(tmp.shape)
    #         # print(torch.rot90(tmp, i, [2, 3]).shape)
    #         # print("Shape of tmp:", tmp.shape)
    #         patches = torch.cat([patches, im2patch(torch.rot90(tmp, i, [2, 3]), frequency_selection=False)])

    print(patches.shape)
    # Optimization loop
    for k in tqdm(range(optimizer_steps)):
        # Randomly sample a batch of patches
        idx = torch.tensor(random.sample(range(patches.shape[0]), batch_size))
        patch_example = patches[idx, :]
        # print(patch_example.shape)
        # Compute the loss
        loss = 0
        invs, jac_inv = patchNR(patch_example, rev=True)
        # print(invs.shape) # 1000 batch and, 36 for a 6x6, 

        loss += torch.mean(0.5 * torch.sum(invs**2, dim=1) - jac_inv)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()# Append the current loss value to the list
        loss_values.append(loss.item())

    # plot_it = True  # Set plot_it to True
    # special_name = "different_specter"
    # plot_spectrograms(patch_example.detach().cpu().numpy(), invs.detach().cpu().numpy())

    torch.save(patch_example, 'patch_example'+ str(patch_size) + '_'+  special_name + '.pt')
    # Plot the loss curve
    torch.save(invs, 'invs'+ str(patch_size) + '_'+  special_name +'.pt')


    torch.save(loss, 'Kernels/loss_values_patchsize'+ str(patch_size) + '_'+  special_name +'.pt')
    # Plot the loss curve
    if plot_loss_train:
        plt.plot(loss_values)
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.show()
        
    # Save the trained model weights and training configuration
    weights = dict()
    weights['batch_size'] = batch_size
    weights['optimizer_steps'] = optimizer_steps
    weights['patch_size'] = patch_size
    weights['net_state_dict'] = patchNR.state_dict()
    torch.save(weights, 'Kernels/MS_weights_patchsize'+ str(patch_size) + '_'+  special_name +'.pt')    


    
def train_patchNR_1_specter(patchNR, img, patch_size, steps, batch_size, center):
    """
    Train the patchNR for the given img (low resolution)
    """
    # Initialize an empty list to store the loss values
    loss_values = []
    # Set the batch size and optimizer steps
    batch_size = batch_size
    optimizer_steps = steps
    # Set the center and create an Adam optimizer for the PatchNR model
    center = center 
    optimizer = torch.optim.Adam(patchNR.parameters(), lr=1e-4)
    # Create a patch extractor
    im2patch = patch_extractor(patch_size=patch_size, pad = False, center=center)
    # Initialize an empty tensor to store the training patches
    patches = torch.empty(0, device=DEVICE)
    
    # Enlarge training patches by rotation and mirroring
    for j in range(2): 
        if j == 0:
            tmp = img
        elif j == 1:
            tmp = torch.flip(img, [1])
        for i in range(4):
            # Extract patches from the image and concatenate them to the patches tensor
            print(patches.shape)
            print(tmp.shape)
            print(torch.rot90(tmp, i, [1, 2]).shape)
            patches = torch.cat([patches, im2patch(torch.rot90(tmp, i, [1, 2]))])


    # Optimization loop
    for k in tqdm(range(optimizer_steps)):
        # Randomly sample a batch of patches
        idx = torch.tensor(random.sample(range(patches.shape[0]), batch_size))
        patch_example = patches[idx, :]
        
        # Compute the loss
        loss = 0
        invs, jac_inv = patchNR(patch_example, rev=True)
        loss += torch.mean(0.5 * torch.sum(invs**2, dim=1) - jac_inv)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()# Append the current loss value to the list
        loss_values.append(loss.item())

    # Plot the loss curve
    torch.save(loss_values, 'Kernels/loss_values_patchsize'+ str(patch_size) +'.pt')
        
    # Save the trained model weights and training configuration
    weights = dict()
    weights['batch_size'] = batch_size
    weights['optimizer_steps'] = optimizer_steps
    weights['patch_size'] = patch_size
    weights['net_state_dict'] = patchNR.state_dict()
    torch.save(weights, 'Kernels/MS_weights_patchsize'+ str(patch_size) +'.pt')

# def patchNR(img, lam, patch_size, n_patches_out, flow, n_iter_max, center):
#     """
#     Defines the reconstruction using patchNR as a regularizer
#     """
    
#     center = center 
#     # Upsample the input image using bicubic interpolation
#     init = torch.nn.functional.interpolate(img, scale_factor=2, mode='bicubic')
#     #save_img(init,'bicubic')
    
#     # create patch extractors
#     input_im2pat = patch_extractor(patch_size, pad=False, center=center)

#     # intialize optimizer for image
#     fake_img = init.clone().detach().requires_grad_(True).to(DEVICE)
#     optim_img = torch.optim.Adam([fake_img], lr=0.001)

#     # Main optimization loop
#     for it in tqdm(range(n_iter_max)):
#         optim_img.zero_grad()
        
#         # Pad the fake image with reflection mode
#         tmp = torch.nn.functional.pad(fake_img, pad=(7, 7, 7, 7), mode='reflect')
#         # Extract patches from the padded image
#         fake_data = input_im2pat(tmp, n_patches_out)
        
#         # PatchNR regularization
#         pred_inv, log_det_inv = flow(fake_data, rev=True)
#         reg = torch.mean(torch.sum(pred_inv**2, dim=1) / 2) - torch.mean(log_det_inv)
        
#         # Data fidelity term
#         data_fid = torch.sum((spatial_blur(tmp, B, subsamp) - img)**2)

#         # Compute the total loss
#         loss = data_fid + lam * reg
#         loss.backward()
#         optim_img.step()
    
#     return fake_img
