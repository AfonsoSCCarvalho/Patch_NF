#Path towards Common Scripts

#Import classic libraries
import numpy as np

import scipy.io as sio
from skimage.io import imread

import torch
import torch.optim as optim
import torch.linalg as LA

from tqdm import tqdm
import time
import matplotlib.pyplot as plt

import keyboard

#Personal Scripts
from gen_B_tensor import gen_B
from Spectral_blur_torch import spectral_blur
from Spatial_blur_torch import spatial_blur
from Quality_indices_pytorch import quality_indices_torch
from FinalResults_reg import *
from AuxFunctions import *
from model import create_NF
from TrainingPatchesMS import *
from utils import *


#Cleaned big functions
from Pre_process import preprocess_data
from Init_Z import initialise_Z
from First_optimisation_without_reg import optimize_without_regularisation
from Train_load_PatchNN import train_and_load_PatchNN
from Regularisation_PatchNN import Regul_PatchNN



    

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
subsamp = 4




SNR_images = 30
dataset= "PaviaU"
special_name_run =f'SNR_{SNR_images}_baseline_{dataset}_FUSE_DEEP_SURE'

mat_file_path = f'Matlab_files\{dataset}'


### The tensors are created with parameters we wish
X0, MS, MS0, HS, HS0, sigmaM ,sigmaM_other, sigmaH, U, B, Y_M, Y_H, UR, UR_UR_T, Y_M_UR, Y_H_Ut, mu = preprocess_data(subsamp=subsamp, SNR_MS =SNR_images,
                                                                                SNR_HS = SNR_images, Principal_components = 9,
                                                                                     plot_images = False, save_tensors=False,special_name=special_name_run, matlab_path = mat_file_path)

learning_rates =[5, 0.05]
target_shape = MS.shape[:2]
#Initialise Z with the HS interpolated
Z, U, mu_reconstrunction = initialise_Z(X0, HS, U, target_shape, DEVICE, mu, save_tensors=False,special_name=special_name_run )


# #### First Optimisation
# #Optimise Z without reg
# Z, Img_reconstructed, elapsed_time_1 = optimize_without_regularisation(
#     Z, Y_M, UR, Y_H_Ut, B, 
#     subsamp, 
#     sigmaM, sigmaH, sigmaM_other, 
#     U, X0,
#     learning_rate_main=learning_rates[0], niter=150*150*2,
#     save_tensors=True,
#      mu = mu_reconstrunction,special_name=special_name_run )
# Img_reconstructed_adam = Img_reconstructed


# print(f"Optimization without regularization took {elapsed_time_1:.2f} seconds.")
# quality_indices(Img_reconstructed, X0)
# print("#####################")


import os
import re
def load_sure_tensor(SNR, dataset):
    # Construct the file path pattern
    file_pattern = f'out_SURE_dataset_{dataset}_SNR{SNR}.*\\.pt'

    # List all files in the directory
    tensor_directory = 'Tensors\Comparison_Baseline\Deep_SURE'
    files = os.listdir(tensor_directory)

    # Search for the file that matches the pattern
    for file in files:
        if re.match(file_pattern, file):
            # If a matching file is found, load the tensor
            tensor_path = os.path.join(tensor_directory, file)
            tensor = torch.load(tensor_path)
            print(f'Loaded tensor from {tensor_path}')
            tensor = tensor.squeeze(0).permute(1, 2, 0).clone().detach()
            return tensor
        
    print('Tensor file not found.')
    return None
SURE = load_sure_tensor(SNR_images, dataset).to(DEVICE)
quality_indices_torch(SURE, X0)
patch_size = 7
center = True
# Parameters for training
train_steps = 100*20
batch_size = 1000*2
load_model = False
# Start timing
start_time_total = time.time()



step = 0.05*1

prev_obj_f = float('inf')
threshold = 0.5e-12
# Start timing

input_im2pat = patch_extractor(patch_size, stride_wanted= 3, pad=True, center=center)

SURE2 = torch.reshape(SURE, (SURE.shape[0] * SURE.shape[1], SURE.shape[2]))
mu = torch.mean(SURE2, axis=0) # Calculate the mean of each band (spectrum)
SURE2 -= mu # Center the data by subtracting the mean from each band
Z = torch.matmul(SURE2, U.T).detach().requires_grad_().to(DEVICE)
torch.save(Z.double().detach(), f'Tensors\Comparison_Baseline\Z_FUSE_{special_name_run}.pt')

model = train_and_load_PatchNN(MS, patch_size, train_steps, batch_size, center, load_model, special_train_name=special_name_run, plot_loss = False)

Z, Img_reconstructed, reg_values, error_list, reg_terms = Regul_PatchNN(Z, X0, MS, HS, Y_M, Y_H_Ut, UR, 
                                                                        sigmaM, sigmaH,sigmaM_other,
                                                                        B, subsamp,
                                                                        U, lam=5e-3,
                                                                        learning_rate_reg=0.0001, num_iters=100,  
                                                                        input_im2pat=input_im2pat, model=model, special_name=special_name_run,
                                                                        scaling = 1, save_tensors=False, mu = mu_reconstrunction)


# plot_results(reg_terms, reg_values, error_list, special_name='')
# quality_indices(Img_reconstructed, X0)
end_time_total = time.time()
elapsed_time = end_time_total - start_time_total
minutes, seconds = divmod(elapsed_time, 60)


file_path = "Tensors/Comparison_Baseline/timescode.txt"
# Writing to the file
with open(file_path, 'a') as file:
    file.write(f"\n{special_name_run} took {int(minutes)} minutes and {int(seconds)} seconds")

print(f"Elapsed time written to {file_path}")

# # Call the FinalResults function with inputs
# FinalResults_reg(HS, HS.cpu().numpy(), MS, MS.cpu().numpy(), Img_reconstructed_adam, Img_reconstructed, X0, X0.cpu().numpy(), target_shape)
#departing point
SURE = load_sure_tensor(SNR_images, dataset).to(DEVICE)

print("departing point")
quality_indices_torch(SURE, X0)

print("after regularisation")
#after regularisation
mse, rmse, psnr, average_ssim , angle_SAM = quality_indices_torch(Img_reconstructed, X0)

