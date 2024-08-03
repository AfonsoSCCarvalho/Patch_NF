"""
This script returns the first optimisation without the Patch NN

"""

import numpy as np
import torch
import scipy.io as sio
from tqdm import tqdm
import torch.linalg as LA
from skimage.io import imread
import time

from gen_B_tensor import gen_B
from Spectral_blur_torch import spectral_blur
from Spatial_blur_torch import spatial_blur
from AuxFunctions import *

#Cleaned big functions
from Pre_process import preprocess_data
from Init_Z import initialise_Z

def optimize_without_regularisation(Z, Y_M, UR, Y_H_Ut, B, subsamp, sigmaM, sigmaH, 
                                    sigmaM_other, U, X0, mu,f_scale=7.718903943896294e-05, learning_rate_main=0.5*5, niter=150*150*2,
                                    save_tensors=False):
    # Set the device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize previous error and previous objective function
    prev_mse = float('inf')
    prev_obj_f = float('inf')

    # Initialize the progress bar manually since tqdm doesn't work well with PyTorch
    progress_bar = tqdm(total=niter, desc='Iterations')

    # Define optimizer for Z
    optimizer_main = torch.optim.Adam([Z], lr=learning_rate_main)

    # Measure the time taken for optimizing
    start_time = time.time()
    print("$$$$$$$$$$$$$$$$$$$")
    print(sigmaH)
    print(sigmaM.item())
    error_list = []  # Store the error values
    obj_f_values = []
    # Assuming sigmaM_other is a scalar or 1x1 tensor
    # sigmaM_other_matrix = torch.diag(torch.full((4,), sigmaM_other)).to(DEVICE)
    # function_scale = f_scale
    function_scale = sigmaM.item()/10000
    # lr_halving_count=0
    for iter in range(niter):
        # Minus gradient
        Z_3d = torch.reshape(Z, (X0.shape[0], X0.shape[1], Z.shape[1])).to(DEVICE)
        LZ = spatial_blur(Z_3d, B, subsamp)
        LZ = torch.reshape(LZ, (LZ.shape[0] * LZ.shape[1], LZ.shape[2])).to(DEVICE)

        # Obj_F = torch.mean((torch.matmul(Y_M - torch.matmul(Z, UR), sigmaM_inv)) ** 2) / 2 + torch.mean((Y_H_Ut - LZ) ** 2) * sigmaH / 2
        # Obj_F = torch.mean((torch.matmul(Y_M - torch.matmul(Z, UR), sigmaM_inv)) ** 2) / 2 + torch.mean(((Y_H_Ut - LZ) ** 2))/( 2*sigmaH)
        Obj_F = function_scale*(torch.mean((Y_M - torch.matmul(Z, UR)) ** 2) / (sigmaM*2) + (9/93)/( 2*sigmaH)* torch.mean(((Y_H_Ut - LZ) ** 2)) )
        obj_f_values.append(Obj_F.item())
        optimizer_main.zero_grad()
        Obj_F.backward()
        optimizer_main.step()

        if iter % 10 == 0:
            Img_reconstructed = torch.matmul(Z, U) + mu
            Img_reconstructed = torch.reshape(Img_reconstructed  , [X0.shape[0], X0.shape[1], Img_reconstructed.shape[1]])
            mse = torch.mean((X0 - Img_reconstructed) ** 2)
            error_list.append(mse.item())

            obj_f_change = prev_obj_f - Obj_F.item()

            if (mse.item() - prev_mse)/(mse.item()) > 0:
                # lr_halving_count += 1
                # if lr_halving_count ==10:
                        print("Stopping optimization: Change in the objective function value is below the threshold.")
                        break

            prev_obj_f = Obj_F.item()
            prev_mse = mse.item()

        progress_bar.update(1)  # Update progress bar

    progress_bar.close()

    end_time = time.time()
    elapsed_time = end_time - start_time

    if save_tensors:
        torch.save(Z, 'Tensors/two_steps_influence_specter_Z_without_reg_snr20.pt')
        torch.save(Img_reconstructed, 'Tensors/two_steps_influence_specter_Img_reconstructed_without_reg_snr20.pt')
    if False:
        plt.figure(figsize=(10, 5))
        plt.plot(obj_f_values, label=f'Alpha: ')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Function Value')
        plt.title(f'Objective Function Values for Alpha = ')
        plt.legend()
        plt.grid(True)
        plt.show()
    return Z, Img_reconstructed, elapsed_time

if __name__ == "__main__":
    #Cleaned big functions
    from Pre_process import preprocess_data
    from Init_Z import initialise_Z
    from First_optimisation_without_reg import optimize_without_regularisation

    Alpha = 0.097
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    subsamp = 4
    ### The tensors are created with parameters we wish
    ### The tensors are created with parameters we wish
    X0, MS, MS0, HS, HS0, sigmaM ,sigmaM_other, sigmaH, U, B, Y_M, Y_H, UR, UR_UR_T, Y_M_UR, Y_H_Ut, mu = preprocess_data(subsamp=subsamp, SNR_MS =20,
                                                                                    SNR_HS = 20, Principal_components = 9,
                                                                                        plot_images = False, save_tensors=True)

    learning_rates =[5]
    target_shape = MS.shape[:2]
    #Initialise Z with the HS interpolated
    Z, U, mu_reconstrunction = initialise_Z(X0, HS, U, target_shape, DEVICE, mu, save_tensors=True)
    print(sigmaM.item()/10)
    # f_scale = 1
    case = [1, 0.5*learning_rates[0] *sigmaM.item()/10 ]
    # f_scale = sigmaM.item()/10 
    case = [sigmaM.item()/10 , 5]

    #### First Optimisation
    #Optimise Z without reg
    Z, Img_reconstructed, elapsed_time_1 = optimize_without_regularisation(
        Z, Y_M, UR, Y_H_Ut, B, 
        subsamp, 
        sigmaM, sigmaH, sigmaM_other, 
        U, X0, f_scale=case[0],
        learning_rate_main=case[1], niter=150*150*2,
        save_tensors=True,
        alpha=Alpha, mu = mu_reconstrunction)
    print(f"Optimization without regularization took {elapsed_time_1:.2f} seconds.")
    quality_indices(Img_reconstructed, X0)