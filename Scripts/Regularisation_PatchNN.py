"""
This script makes the Patch NN regularisation

"""
import sys

import numpy as np
import torch
import scipy.io as sio
from tqdm import tqdm
import torch.linalg as LA
from skimage.io import imread
import time
from torch.optim.lr_scheduler import ExponentialLR

import os

from gen_B_tensor import gen_B
from Spectral_blur_torch import spectral_blur
from Spatial_blur_torch import spatial_blur
from AuxFunctions import *
from TrainingPatchesMS import *
from utils import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Regul_PatchNN(Z, X0, MS, HS, Y_M, Y_H_Ut, UR, sigmaM, sigmaH, sigmaM_other, B, subsamp, U, mu, lam=0.000000025,
                                learning_rate_reg=0.05*1, num_iters=150*150*2,  input_im2pat=None, model=None, special_name='',
                                scaling = 1, save_tensors=False, plot_graphs =False):
    """
    We can define the scaling as:
    0 - No scaling of Z
    1 - Scaling of Z between 0 and 1 (recommended)
    2 - Scaling of Z between MS min and MS max
    You must uncomment to see these changes
    """
    niter = num_iters
    progress_bar = tqdm(total=niter, desc='Iterations')
    optimizer_reg = torch.optim.Adam([Z], lr=learning_rate_reg)
    scheduler = ExponentialLR(optimizer_reg, gamma=0.99)  # Apply exponential decay with the specified gamma

    error_list = []  # Store the error values
    reg_values = []
    reg_terms = [[0] for _ in range(Z.shape[1])]
    lamda_evolution = [[] for _ in range(Z.shape[1])]  # Store lambda evolution for each component

    prev_mse = float('inf')
    lr_halving_count = 0

    get_reg_coeff = [0.000001 for _ in range(Z.shape[1])]

    
    get_reg_coeff[0] = 45*lam
    get_reg_coeff[1] = 70*lam
    get_reg_coeff[2] = 65*lam
    get_reg_coeff[3] = 10*lam
    get_reg_coeff[4] = 10*lam
    get_reg_coeff[5] = 10*lam
    get_reg_coeff[6] = 5 *lam # 0.001
    get_reg_coeff[7] = 5 *lam # 0.0001
    get_reg_coeff[8] = 5 *lam # 0.00001

    get_reg_coeff = [lam for _ in range(Z.shape[1])]

    progress_bar = tqdm(total=niter, desc='Iterations')
    interval_start_select = 0
    interval_end_select = 0

    min_value_MS = torch.min(MS)
    max_value_MS = torch.max(MS)
        # function_scale = sigmaM.item()/10000
    function_scale = 1
    # sigmaM_inv =0.000772024504840374
    import keyboard
    Z_0 = Z.clone()
    # Define the hotkey combination
    HOTKEY = 'ctrl + s'
    i = 0
    start_time = time.time()
    Y_H_Ut_error_terms = []
    # Initialize a variable to keep track of the previous value of Obj_F1
    prev_Obj_F1 = float('inf')
    mse_list = []
    Obj_F1_list = []
    fidelity_term_list = []
    scaled_reg_term_list = []
    Y_H_Ut_error_terms_list_9 = [ [] for _ in range(Z.shape[1])]
    function_scale = torch.mean(sigmaM)/10
    rho = 10
    alpha = 1.2 #np.sqrt(0.8)

    # function_scale = sigmaM/2
    for iter in range(niter):
        if keyboard.is_pressed(HOTKEY):
            print("Loop interrupted by hotkey.")
            break


        Z_3d = torch.reshape(Z, (X0.shape[0], X0.shape[1], Z.shape[1])).to(DEVICE)
        LZ = spatial_blur(Z_3d, B, subsamp)
        LZ = torch.reshape(LZ, (LZ.shape[0] * LZ.shape[1], LZ.shape[2])).to(DEVICE)

        reg = 0

        Z_3d_specter_first = Z
        Z_3d_specter_first = torch.reshape(Z_3d_specter_first, (X0.shape[0], X0.shape[1], Z.shape[1]))
        Z_3d_specter_first = Z_3d_specter_first.permute(2, 0, 1).unsqueeze(0)



        Y_H_Ut_error_term = torch.mean((Y_H_Ut - LZ) ** 2)
        Y_H_Ut_error_terms.append((Y_H_Ut_error_term).cpu().detach().numpy())  # Store the updated lambda value

        for j in range(Z.shape[1]):  # Assuming Z.shape[1] is the number of components
            Z_3d_component = Z_3d_specter_first

            min_value_Z = torch.min(Z_3d_specter_first[:, i, :, :])
            max_value_Z = torch.max(Z_3d_specter_first[:, i, :, :])

            Z_3d_normalized = (Z_3d_component - min_value_Z) * (1 - 0) / (max_value_Z - min_value_Z)

            fake_data = input_im2pat(Z_3d_normalized, batch_size=0, random_interval=True,
                                     interval_start=interval_start_select, interval_end=interval_end_select,
                                     frequency_selection=False)
            pred_inv, log_det_inv = model(fake_data, rev=True)
            reg_term = torch.mean(torch.sum(pred_inv**2, dim=1) / 2)  - torch.mean(log_det_inv)
            # print(pred_inv)

            reg_terms[j].append(reg_term.item())
            reg += get_reg_coeff[j] * reg_term  # Accumulate regularization from all components
                        
            # adjust_lamdas_for_component(j, get_reg_coeff, Y_H_Ut, LZ, Z, sigmaH, rho, alpha)
            lamda_evolution[j].append(get_reg_coeff[j])  # Store the updated lambda value
            Y_H_Ut_error_terms_list_9[j].append(torch.mean(((Y_H_Ut - LZ)[j]) ** 2).cpu().detach().numpy())



        reg_values.append(reg.item())  # Store the total regularization value

        Obj_F1 = (torch.sum((Y_M - torch.matmul(Z, UR)) ** 2) / (sigmaM*2) + (9/93)/( 2*sigmaH)* torch.sum(((Y_H_Ut - LZ) ** 2))) +  reg 


        optimizer_reg.zero_grad()
        prev_Obj_F1 = Obj_F1.item()  # Update
        Obj_F1.backward()
        optimizer_reg.step()

        Img_reconstructed = torch.matmul(Z, U) + mu
        Img_reconstructed = torch.reshape(Img_reconstructed, [X0.shape[0], X0.shape[1], Img_reconstructed.shape[1]])
        mse = torch.mean((X0 - Img_reconstructed) ** 2)
        # Get current time in seconds
        current_time = time.time()
        error_list.append((mse.item(), current_time))
        if iter % 10 == 0:
            print(f"Mean Squared Error: {mse.item()}")
            print(f"Objective function: {Obj_F1.item()}")

        prev_mse = mse.item()  # Update the previous error
        Y_H_Ut_error_terms.append((torch.mean((Y_H_Ut - LZ) ** 2)*2).cpu().detach())
        progress_bar.update(1)  # Update progress bar
        # i += 1
        mse_list.append(prev_mse)
        Obj_F1_list.append(prev_Obj_F1)
        # Inside the loop
        fidelity_term =  (torch.sum((Y_M - torch.matmul(Z, UR)) ** 2) / (sigmaM * 2) + (9/93) / (2 * sigmaH) * torch.sum((Y_H_Ut - LZ) ** 2))
        scaled_reg_term =  reg

        fidelity_term_list.append(fidelity_term.item())
        scaled_reg_term_list.append(scaled_reg_term.item())

        # if i == 9:
        #     i = 0
          
    end_time = time.time()
    progress_bar.close()
    elapsed_time_2 = end_time - start_time
    print(f"Regularization optimization took {elapsed_time_2} seconds.")
    
    torch.save(error_list, f'Tensors/Comparison_Baseline/error_list_{special_name}.pt')  # Saving error_list

    mse_list = [mse for mse, _ in error_list]  # Assuming error_list contains tuples of (mse, timestamp)
    Loss_list = [obj for obj in Obj_F1_list]  # You need to ensure Obj_F1_list captures Obj_F1 values in your loop

    if plot_graphs:


            # Create a figure and a set of subplots.
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Plotting Mean Squared Error over Iterations on the first y-axis
        color = 'tab:blue'
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('MSE', color=color)
        ax1.plot(mse_list, label='Mean Squared Error', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True)

        # Instantiate a second y-axis for the same x-axis
        ax2 = ax1.twinx()  
        color = 'tab:red'
        ax2.set_ylabel('Objective Function Value', color=color)
        ax2.plot(Loss_list, label='Objective Function (Obj_F1)', color=color)  # Use the smoothed list here
        ax2.tick_params(axis='y', labelcolor=color)

        # Adding title and legend
        fig.tight_layout()  # to ensure there's no overlap in the layout
        plt.title('MSE and Objective Function over Iterations')
        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

        # Show plot
        # plt.show()


        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Plot the Fidelity Term on the primary y-axis
        color = 'tab:blue'
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Fidelity Term Value', color=color)
        ax1.plot(fidelity_term_list, label='Fidelity Term', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True)

        # Create a second y-axis for the scaled regularization term
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Lambda Scaled Regularization Term', color=color)
        ax2.plot(scaled_reg_term_list, label='Lambda Scaled Regularization Term', color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        # Adding title and combined legend
        fig.suptitle('Fidelity and Regularization Terms over Iterations')
        fig.legend(loc="upper right", bbox_to_anchor=(1, 0.9))
        # plt.show()



        # fig, axs = plt.subplots(3, 3, figsize=(15, 10))
        # axs = axs.flatten()  # Flatten the 2D array of axes into a 1D array for easy iteration

        # for j in range(Z.shape[1]):
        #     axs[j].plot([v.cpu().detach().numpy() for v in lamda_evolution[j]], label=f'Lambda {j+1}')
        #     axs[j].set_xlabel('Iteration')
        #     axs[j].set_ylabel('Lambda Value')
        #     axs[j].set_title(f'Lambda {j+1} Evolution')
        #     axs[j].legend()
        #     axs[j].grid(True)

        # # Adjust layout to prevent overlap
        # plt.tight_layout()
        # plt.show()

        constant_value = sigmaH * alpha
        # Create a grid of subplots
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))  # 3x3 grid for 9 components
        lim_small =constant_value *0.8
        lim_big = constant_value *1.2
            # Plot each component's error term evolution in a separate subplot
        for i in range(Z.shape[1]):
            row = i // 3
            col = i % 3
            ax = axs[row, col]

            ax.plot(Y_H_Ut_error_terms_list_9[i], label=f'Component {i + 1}')

            ax.axhline(y=lim_small, color='r', linestyle='--', label='0.8 * SigmaH Alfa')
            ax.axhline(y=lim_big, color='g', linestyle='--', label='1.2 * SigmaH Alfa')

            ax.set_title(f'Component {i + 1}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Y_H_Ut Error Term')
            
            ax.legend()
        fig.suptitle('Evolution of the YH error for each term')
        plt.tight_layout()
        plt.show()
    # Show plot

     # Path to save the images
    save_path = os.path.expanduser("~/Downloads")  # Adjusts to user's home directory

    # Example plot save function
    def save_plots():
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(mse_list, label='MSE', color='tab:blue')
        ax1.set_title('MSE and Objective Function over Iterations')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('MSE')
        ax1.legend()
        fig.savefig(os.path.join(save_path, f'{special_name}_MSE_and_Objective_Function_rho_{rho}_alpha_{alpha}_.png'))

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(fidelity_term_list, color='tab:blue', label='Fidelity Term')
        ax1.plot(scaled_reg_term_list, color='tab:red', label='Scaled Regularization Term')
        ax1.set_title('Fidelity and Regularization Terms')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Term Value')
        ax1.legend()
        fig.savefig(os.path.join(save_path, f'{special_name}_Fidelity_and_Regularization_Terms_rho_{rho}_alpha_{alpha}_{special_name}.png'))

        # Remember to call plt.close() to free up memory
        plt.close('all')

        # Assuming this function call is appropriate in your workflow
    save_plots()

    if save_tensors:
        torch.save(Z_3d.double(), f'Tensors/Comparison_Baseline/rZ_3d_influence_specter_patch_log__{special_name}.pt')
        torch.save(Y_H_Ut_error_terms, f'Tensors/Comparison_Baseline/Y_H_Ut_error_terms_{special_name}.pt')
        torch.save(Img_reconstructed.double(), f'Tensors/Comparison_Baseline/two_steps_influence_specter_Img_reconstructed_reg_patch_log__{special_name}.pt')
        torch.save(Z.double(), f'Tensors/Comparison_Baseline/two_steps_influence_specter_Z_reg_patch_log__{special_name}.pt')

    return Z, Img_reconstructed, reg_values, error_list, reg_terms


def adjust_lamdas_for_component(component, get_reg_coeff, Y_H_Ut, LZ, Z,  sigmaH, rho, alpha):
    Y_H_Ut_error_term_list = []
    Y_H_Ut_error_term_i = torch.mean(((Y_H_Ut.detach() - LZ.detach())[component])**2)

    current_lamda = get_reg_coeff[component]

    current_lamda = current_lamda*torch.exp(rho*( sigmaH*alpha - (Y_H_Ut_error_term_i )/( np.sqrt(Z.shape[0])) ))

    get_reg_coeff[component] = current_lamda