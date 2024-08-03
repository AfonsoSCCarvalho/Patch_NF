"""
This script pre_process the data, for the Pavia Dataset, returning 
X0, MS, HS, sigmaM_inv, sigmaH_inv, Y_H, UR_UR_T, Y_M_UR

"""
import sys


import numpy as np
import torch
import scipy.io as sio
from tqdm import tqdm
import torch.linalg as LA
from skimage.io import imread

from gen_B_tensor import gen_B
from Spectral_blur_torch import spectral_blur
from Spatial_blur_torch import spatial_blur
from AuxFunctions import *

# learning_rates =[0.0005, 5]
# SNR_images = 20
# special_name_run =f'SNR_{SNR_images}_baseline_SUSE'



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
    

def preprocess_data(dataset='PaviaU', subsamp=4, SNR_MS =20, SNR_HS = 20, Principal_components = 9, plot_images = False, save_tensors=False, special_name='Baseline_Final_lamda',matlab_path = "Matlab_files/PaviaU"):
    # Set the device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if dataset == 'PaviaU':
        # Load the PaviaU dataset
        mat_data = sio.loadmat('data_set/PaviaU.mat')
        data_name = 'paviaU'
        # Access the specific variable from the loaded data
        X0 = mat_data[data_name]
        
        print(X0.shape)
        X0 = X0.astype(float)

        # Crop the image to the central 256x256 pixels and remove the first 11 bands
        X0 = X0[50:306, 50:306, 10:103]

    elif dataset == 'Indian_pines':
        # Load the Indian Pines dataset
        mat_data = sio.loadmat('data_set/Indian_pines.mat')
        data_name = 'indian_pines'

        # Access the specific variable from the loaded data
        X0 = mat_data[data_name]
        
        print(X0.shape)
        X0 = X0.astype(float)

        # Crop the image to the central 256x256 pixels and remove the first 11 bands
        X0 = X0[0:140, 0:140, 11:104]


    else:
        raise ValueError("Unsupported dataset. Please choose 'PaviaU' or 'Indian_pines'.")
    

    # Find the global minimum and maximum values across all bands
    global_min = np.min(X0)
    global_max = np.max(X0)

    X0 = (X0 - global_min) / (global_max - global_min)



    # Set the number of pixels and bands
    nr, nc, L = X0.shape

    # Set the SNR levels for the HS and MS images
    snr_HS = np.concatenate([SNR_HS * np.ones(L - 50), (SNR_HS) * np.ones(50)])
    snr_MS = SNR_MS

    # Load the PSF matrix R
    R = sio.loadmat('R.mat')['R']
    # Discard first 10 columns of the PSF matrix
    R = R[:, 10:]
    # Normalize each row of the PSF matrix
    R = R / R.sum(axis=1, keepdims=True)

    # Perform spectral blurring of the original hyperspectral (HS) image X0 using the PSF matrix R
    X0 = torch.from_numpy(X0).float()
    R = torch.from_numpy(R).float()
    MS0 = spectral_blur(X0, R)

    print("################################# SIGMA_M #######################################")

    # Add Gaussian noise to the HS image with noise level specified by snr_MS
    Ps = torch.mean(torch.mean(MS0 ** 2, dim=0), dim=0).squeeze()
    print(Ps.shape)
    sigmaM = Ps * (10 ** (-snr_MS / 10))
    sigmaM_Mean = torch.mean(sigmaM)
    MS = torch.zeros_like(MS0)
    for i in range(MS.shape[2]):
        noise = torch.randn(MS0[:,:,i].shape[0], MS0[:,:,i].shape[1]) * torch.sqrt(sigmaM_Mean)
        MS[:, :, i] = MS0[:, :, i] + noise

    
    sigmaM_other = torch.mean(sigmaM)
    # print(sigmaM_Mean.item())
    # print("Inverted")
    # print(1/sigmaM_Mean.item())
    # print("Matlab")
    # print(0.0007727383063195060)
    
        # Calculate the inverse of sigmaM and sigmaH
    sigmaM_inv = torch.pow(sigmaM,-0.5)  # Square root
    print(sigmaM_inv)
    sigmaM_inv = torch.reciprocal(sigmaM_inv) # Reciprocal
    print(sigmaM_inv)
    sigmaM_inv = torch.diag_embed(sigmaM_inv).to(DEVICE)
    print(sigmaM_inv)
    mean_sigmaM_inv2 =  torch.mean(sigmaM)

    sigmaM_other = torch.diag(torch.full((sigmaM_inv.size(0),), mean_sigmaM_inv2)).to(DEVICE)
    sigmaM_other = sigmaM_inv
    # sigmaM_other = torch.pow(sigmaM_other,-0.5)  # Square root
    # print(sigmaM_other)
    # sigmaM_other = torch.reciprocal(sigmaM_other) # Reciprocal
    # print(sigmaM_other)
    ###########################################################################
    # Spatial degradation (HS)
    torch.manual_seed(2016)

    B = gen_B(X0.shape[0], X0.shape[1])  # Generates the point spread function
    # Subsampling mask
    mask = torch.zeros_like(B)
    mask[0::subsamp, 0::subsamp] = 1  # Define the subsampling mask

    # Spatially blur the ground truth image X0 with the PSF
    Y = spatial_blur(X0, B, subsamp)
    HS0 = Y

    # Add Gaussian noise to the HS image with noise level specified by snr_HS
    Ps = torch.mean(torch.mean(HS0 ** 2, dim=0), dim=0).squeeze()
    print(Ps.shape)
    sigmaH = Ps * (10 ** (-snr_HS / 10))

    sigmaH_mean = torch.mean(sigmaH)
   


    HS = torch.zeros_like(HS0)
    for i in range(HS.shape[2]):
        noise = torch.randn(*HS0[:,:,i].shape) * torch.sqrt(sigmaH_mean)
        HS[:, :, i] = HS0[:, :, i] + noise



    print("################################# SIGMA_H #######################################")
    sigmaH_Mean = (sigmaH_mean).item()
    # print(sigmaH_Mean)
    # print("Inverted")
    # print(1/sigmaH_Mean)
    # print("Matlab")
    # print(0.0008167737172920329)
    #################################### U  #######################################


    HS = HS.unsqueeze(0).unsqueeze(0).to(DEVICE)
    X0 = X0.unsqueeze(0).unsqueeze(0).to(DEVICE)

    # Perform PCA dimensionality reduction on HS
    img = HS.clone().squeeze(0).squeeze(0)
    # Reshape image into 2 dimensions: space x spectrum
    img2 = torch.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    mu = torch.mean(img2, axis=0) # Calculate the mean of each band (spectrum)
    img2 -= mu # Center the data by subtracting the mean from each band
    cov = torch.matmul(img2.T, img2)# Compute the covariance matrix

    # Calculate the eigenvalues (lambdas) and eigenvectors (U)
    lambdas, U = LA.eigh(cov)
    U = U[:, -Principal_components:]  # Select the first 9 eigenvectors (components)
    U = U.T.to(DEVICE)
    # Map the image to the reduced-dimensional space
    y = torch.matmul(img2, U.T)
    # Reshape the mapped image to its original dimensions
    y = torch.reshape(y, (img.shape[0], img.shape[1], y.shape[1]))
    # Reproject data into the original space
    reconstructed_img = torch.matmul(y, U) + mu
    reconstructed_img = reconstructed_img.reshape(img.shape) # Reshape the reconstructed image to its original dimension
    quality_indices_torch(reconstructed_img,img)
    # Move tensors to the specified device
    X0 = X0.squeeze(0).squeeze(0).to(DEVICE)
    MS = MS.to(DEVICE)
    HS = HS.squeeze(0).squeeze(0).to(DEVICE)
    # Reshape MS to have rows as pixels and columns as spectra
    Y_M = torch.reshape(MS, (MS.shape[0] * MS.shape[1], MS.shape[2])).to(DEVICE)
    Y_H = torch.reshape(HS, (HS.shape[0] * HS.shape[1], HS.shape[2])).to(DEVICE)
   
    R = R.T.to(DEVICE)
    print(R.shape)
    print(U.shape)
    # sigmaM_matrix = torch.diag(1 / sigmaM).to(DEVICE)
    # sigmaH_matrix = torch.diag(1 / sigmaH).to(DEVICE)
    print(torch.mean(sigmaM))
    # Calculate the inverse of sigmaM and sigmaH
    # sigmaM_inv = torch.pow(sigmaM,-0.5)  # Square root
    # print(sigmaM_inv)
    sigmaM_inv = torch.reciprocal(sigmaM*10) # Reciprocal
    print(sigmaM_inv)
    sigmaM_inv = torch.diag_embed(sigmaM_inv).to(DEVICE)
    # sigmaM_inv = torch.diag_embed(sigmaM).to(DEVICE)
    print(sigmaM_inv)

    # Reshape MS to have rows as pixels and columns as spectra
    # Y_M = torch.reshape(MS, (MS.shape[0] * MS.shape[1], MS.shape[2])).to(DEVICE)
    mu_yM = torch.mean(Y_M, axis=0) # Calculate the mean of each band (spectrum)
    Y_M -= mu_yM # Center the data by subtracting the mean from each band

    # Y_H = torch.reshape(HS, (HS.shape[0] * HS.shape[1], HS.shape[2])).to(DEVICE)
    mu_yH = torch.mean(Y_H, axis=0) # Calculate the mean of each band (spectrum)
    Y_H -= mu_yH # Center the data by subtracting the mean from each band

    # Precalculations for optimization
    UR = torch.matmul(U, R).to(DEVICE)
    UR_UR_T = torch.matmul(UR, UR.T).to(DEVICE)
    Y_M_UR = torch.matmul(Y_M, UR.T).to(DEVICE)
    Y_H_Ut = torch.matmul(Y_H, U.T).to(DEVICE)


    mean_F_sigmaH = np.mean((sigmaH_Mean) ** 2)
    print(mean_F_sigmaH)
        # Assuming the size of the square matrix matches the number of columns in U
    n = U.shape[1]  # This assumes U is already defined and you want a square matrix of size compatible with U

    # Create a vector with sigmaH_Mean repeated n times
    sigmaH_vector = torch.full((n,), sigmaH_Mean)
    # print(sigmaH_Mean)
    # print(sigmaH_vector)

    # Create a diagonal matrix with sigmaH_Mean as the diagonal values
    sigmaH_matrix = torch.diag(sigmaH_vector)
    # print(sigmaH_matrix)
    mean_U = torch.mean((torch.matmul(sigmaH_matrix.to(DEVICE), U.T)) ** 2)
    print(mean_U)

    print(Principal_components/X0.shape[2])
    print(mean_U/mean_F_sigmaH)

    print(1/(9/93))
    print(1/(mean_U/mean_F_sigmaH))

    random_means = []
    for i in range(1):
        random_noise_vector = torch.rand(n)
        # print(random_noise_vector)
        # Move the vector to the specified DEVICE
        random_noise_vector = random_noise_vector.to(DEVICE)

        # Create a diagonal matrix with the random noise vector
        random_noise_matrix = torch.diag(random_noise_vector).to(DEVICE)

        result_with_noise = torch.matmul(random_noise_matrix, U.T)

        # Calculate and print the mean of the square of the result
        mean_result_with_noise = torch.mean(result_with_noise ** 2)
        random_means.append(( mean_result_with_noise/(torch.mean(random_noise_vector)**2 )).cpu().numpy()) 
        # print(mean_result_with_noise/(torch.mean(random_noise_vector)**2 ))
    print(np.mean(random_means))

    

    if plot_images:
        plot_img(X0.cpu().numpy(), "X0 image")
        plot_img(MS.cpu().numpy(), "MS image")
        plot_img(HS.cpu().numpy(), "HS image")
    if save_tensors:
        torch.save(HS.double().detach(), f'Tensors/Comparison_Baseline/HS_{special_name}.pt')
        torch.save(MS.double().detach(), f'Tensors/Comparison_Baseline/MS_{special_name}.pt')
        torch.save(MS0.double().detach(), f'Tensors/Comparison_Baseline/MS0_{special_name}.pt')
        torch.save(HS0.double().detach(), f'Tensors/Comparison_Baseline/HS0_{special_name}.pt')
        torch.save(U.double().detach(), f'Tensors/Comparison_Baseline/U_{special_name}.pt') 
        torch.save(X0.double().detach(), f'Tensors/Comparison_Baseline/X0_{special_name}.pt') 

    return X0, MS, MS0, HS, HS0, sigmaM_Mean,sigmaM_other, sigmaH_Mean , U, B, Y_M, Y_H, UR, UR_UR_T, Y_M_UR, Y_H_Ut, mu

if __name__ == "__main__":
    X0, MS, MS0, HS, HS0, sigmaM_Mean, sigmaM_other, sigmaH, U, B, Y_M, Y_H, UR, UR_UR_T, Y_M_UR, Y_H_Ut, mu = preprocess_data(dataset="PaviaU", plot_images=True, save_tensors=True)