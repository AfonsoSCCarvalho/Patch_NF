#Path towards Common Scripts


import torch
import torch.linalg as LA
from AuxFunctions import *


def initialise_Z(X0, HS, U, target_shape, DEVICE, mu, save_tensors=False, Principal_components=9,special_name=" " ):
    # Upscale the HS image
    upscaled_hs_image = upscale_hs_image(HS, target_shape)
    upscaled_hs_image_ori = upscaled_hs_image
    print(upscaled_hs_image.shape)
    if save_tensors:
        torch.save(upscaled_hs_image.double().detach(), 'Tensors/upscaled_hs_image_snr20.pt')
        torch.save(U.double().detach(), 'Tensors/U_snr20.pt') 
        torch.save(X0.double().detach(), 'Tensors/X0.pt')   
    # First initialization
    quality_indices_torch(upscaled_hs_image, X0)

    upscaled_hs_image = torch.reshape(upscaled_hs_image, (upscaled_hs_image.shape[0] * upscaled_hs_image.shape[1], upscaled_hs_image.shape[2]))
    # print(upscaled_hs_image.shape)

    mu = torch.mean(upscaled_hs_image, axis=0) # Calculate the mean of each band (spectrum)
    upscaled_hs_image -= mu # Center the data by subtracting the mean from each band
    # cov = torch.matmul(upscaled_hs_image.T, upscaled_hs_image)# Compute the covariance matrix

    Z = torch.matmul(upscaled_hs_image, U.T).detach().requires_grad_().to(DEVICE)
    if save_tensors:        
        X0_2D = torch.reshape(X0, (X0.shape[0] * X0.shape[1], X0.shape[2]))
        print(X0_2D.shape)
        print(Z.shape)
        print("###########################")
        Z_true = torch.matmul(X0_2D, U.T).detach().requires_grad_().to(DEVICE)
        torch.save(Z_true.double().detach(), f'Tensors\Comparison_Baseline\Z_true_{special_name}.pt')
        torch.save(Z.double().detach(), f'Tensors\Comparison_Baseline\Z_init_{special_name}.pt')


    return Z, U, mu
