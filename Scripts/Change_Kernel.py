import numpy as np
import scipy.io as sio
from skimage.io import imread
from gen_B_tensor import gen_B
from Spectral_blur_torch import spectral_blur
from Spatial_blur_torch import spatial_blur
import torch.linalg as LA
from tqdm import tqdm
from Quality_indices_pytorch import quality_indices
import time
import matplotlib.pyplot as plt
import torch
from FinalResults import *

from AuxFunctions import *

from model import create_NF
from tqdm import tqdm

import torch.optim as optim

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(2016)

X0 = imread('original_rosis.tif')
X0 = X0.astype(float)
# print(X0.shape)  # It should be 610, 340 , 103

# Transpose and flip the axes to reshape X0
X0 = np.transpose(X0, (1, 2, 0))

# print(X0.shape)  # Output: (610, 340, 103)

# Crop the image to the central 256x256 pixels and remove the first 11 bands
X0 = X0[49:305, 49:305, 10:]

# Normalize the pixel values to be between 0 and 1
X0 = X0 + np.mean(X0)
X0 = X0 / (np.max(X0) + np.min(X0))

# Set the number of pixels and bands
nr, nc, L = X0.shape

# plot_img(X0, "HS image")

# Set the SNR levels for the HS and MS images
# For the HS image, the SNR varies across the bands
snr_value = 30  # SNR value

# snr_HS = torch.cat([torch.tensor(snr_value).repeat(L - 50), torch.tensor(snr_value).repeat(50)])
# print(snr_HS.shape)
#For the HS image, the SNR varies across the bands
snr_HS = np.concatenate([30*np.ones(L-50), 30*np.ones(50)])
# For the MS image, the SNR is constant
snr_MS = 30

# Set the subsampling rate for the HS image
subsamp = 4

###
# Load point spread function (PSF) matrix from the file R.mat

R = sio.loadmat('R.mat')['R']

# Discard first 10 columns of the PSF matrix
R = R[:, 10:]

# Normalize each row of the PSF matrix
R = R / R.sum(axis=1, keepdims=True)
# plot_img(X0, "Original image")
# Perform spectral blurring of the original hyperspectral (HS) image X0 using the PSF matrix R
X0 = torch.from_numpy(X0).float()
X0_np = X0.numpy()
# print(X0.shape)
R = torch.from_numpy(R).float()
# print(R.shape)
MS0 = spectral_blur(X0, R)
MS0_np= MS0.numpy()
# plot_img(MS0_np, "MS image")

# Add Gaussian noise to the HS image with noise level specified by snr_HS
Ps = torch.mean(torch.mean(MS0 ** 2, dim=0), dim=0).squeeze()
sigmaM = Ps * (10 ** (-snr_MS / 10))

MS = torch.zeros_like(MS0)

for i in range(MS.shape[2]):
    noise = torch.randn(MS0[:,:,i].shape[0], MS0[:,:,i].shape[1]) * torch.sqrt(sigmaM[i])
    MS[:, :, i] = MS0[:, :, i] + noise

##Plot with numpy
MS_np = MS.numpy()
#
# # Display the MS image
# plot_img(MS_np, "MS image")

# Spatial degradation (HS)
torch.manual_seed(2016)

# Generate PSF
B = gen_B(X0.shape[0], X0.shape[1])  # Generates the point spread function

# X0 = torch.from_numpy(X0).unsqueeze(0).float()
# B = torch.from_numpy(B).unsqueeze(0).float()

# B = torch.from_numpy(B).unsqueeze(0).float()
# B = B.unsqueeze(0).float()

# print(B[0,1,0,0])


# Subsampling mask
mask = torch.zeros_like(B)
mask[0::subsamp, 0::subsamp] = 1  # Define the subsampling mask

# Convert X0 to a PyTorch tensor
# X0 = torch.from_numpy(X0).unsqueeze(0).float()

# Spatial blurring
Y = spatial_blur(X0, B,subsamp)  # Spatially blur the ground truth image X0 with the PSF
# Z = spatial_blur(Y, B)

# Subsample
HS0 = Y# Subsample the blurred image Y
print("HS0 mean")
print(torch.mean(HS0))


HS0_plot = HS0.numpy()
# plot_img(HS0_plot, "HS image")

# Add Gaussian noise to the HS image with noise level specified by snr_HS
Ps = torch.mean(torch.mean(HS0 ** 2, dim=0), dim=0).squeeze()
sigmaH = Ps * (10 ** (-snr_HS / 10))

HS = torch.zeros_like(HS0)

for i in range(HS.shape[2]):
    noise = torch.randn(*HS0[:,:,i].shape) * torch.sqrt(sigmaH[ i])
    HS[:, :, i] = HS0[ :, :, i] + noise


HS_np = HS.numpy()



# plot_img(MS_np, "MS image")
#plot_img(HS_np, "MS image")

HS_np = HS.numpy()



print(HS.shape)
print(X0.shape)

# HS = HS[:,:,2]
# X0 = X0[:,:,2]

HS = HS.unsqueeze(0).unsqueeze(0).to(DEVICE)
X0 = X0.unsqueeze(0).unsqueeze(0).to(DEVICE)

print(HS.shape)
print(X0.shape)


# Display the MS image
# plot_img(HS_np, "HS image")
print("fffffff")
img = HS.clone().squeeze(0).squeeze(0)
print(img.shape)

# Reshape image into 2 dimensions: space x spectrum
img2 = torch.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))


# Compute sample covariance matrix
mu = torch.mean(img2, axis=0)  # Calculate the mean of each band (spectrum)
img2 -= mu  # Center the data by subtracting the mean from each band
cov = torch.matmul(img2.T, img2)  # Compute the covariance matrix

# Get eigenvalues and eigenvectors
lambdas, U = LA.eigh(cov)  # Calculate the eigenvalues (lambdas) and eigenvectors (U)
print("U")
print(torch.mean(U))

# Select the first 9 eigenvectors (components)
# U = U[:, 84:93:1]
U = U[:, -9:]
# Map the image to the reduced-dimensional space
y = torch.matmul(img2, U)

# Reshape the mapped image to its original dimensions

y = torch.reshape(y, (img.shape[0] , img.shape[1], y.shape[1]))


# Reproject data into the original space
reconstructed_img = torch.matmul(y, U.t()) + mu

# Reshape the reconstructed image to its original dimensions
reconstructed_img = reconstructed_img.reshape(img.shape)

reconstructed_img_np = reconstructed_img.cpu().numpy()

# Display the MS image
# plot_img(reconstructed_img_np, "Reconstructed_img image")

####################################################
X0 = X0.squeeze(0).squeeze(0).to(DEVICE)
MS = MS.to(DEVICE)
HS = HS.squeeze(0).squeeze(0).to(DEVICE)
# Reshape MS to have rows as pixels and columns as spectra
Y_M = torch.reshape(MS, (MS.shape[0] * MS.shape[1], MS.shape[2])).to(DEVICE)
Y_H = torch.reshape(HS, (HS.shape[0] * HS.shape[1], HS.shape[2])).to(DEVICE)

num_rows = X0.shape[0]
num_cols = U.shape[1]

U = U.T.to(DEVICE)


target_shape = MS.shape[:2]

# Upscale the HS image

# quality_indices(upscaled_hs_image,X0)
# upscaled_hs_image = torch.reshape(upscaled_hs_image, (upscaled_hs_image.shape[0] * upscaled_hs_image.shape[1], upscaled_hs_image.shape[2]))
# Z = torch.matmul(upscaled_hs_image, U.T).detach().requires_grad_().to(DEVICE)

R = R.T.to(DEVICE)
# B = B.T
print("U")
print(torch.mean(U))
print("R")
print(torch.mean(R))
print("Y_M")
print(torch.mean(Y_M))
print("Y_H")
print(torch.mean(Y_H))


sigmaM_matrix = torch.diag(1 / sigmaM).to(DEVICE)
sigmaH_matrix = torch.diag(1 / sigmaH).to(DEVICE)

niter = 150 * 15 # Number of iterations


# Precalculus
UR = torch.matmul(U, R).to(DEVICE)
UR_UR_T = torch.matmul(UR, UR.T).to(DEVICE)
Y_M_UR = torch.matmul(Y_M, UR.T).to(DEVICE)

Y_H_Ut = torch.matmul(Y_H, U.T).to(DEVICE)

error_list = []  # Store the error values

# Initialize previous error to a high value
prev_mse = float('inf')



error_list = []  # Store the error values

# Initialize previous error to a high value
prev_mse = float('inf')

sigmaM_inv = 0.00028823601371312177 #1 / (torch.mean(sigmaM_matrix))

sigmaH_inv =  0.006992257372015654  #   make it right   1 / (torch.mean(sigmaH_matrix))
# print("hey")
# print(sigmaM_inv)
# print(torch.mean(torch.matmul(Z, UR)))



from TrainingPatchesMS import *

from utils2 import *



MS_test = MS.unsqueeze(0)
MS_test = torch.reshape(MS_test, (MS_test.shape[0], MS_test.shape[3], MS_test.shape[1], MS_test.shape[2]))
print(MS_test.shape)
MS_test = MS_test[:,1,:,:].unsqueeze(0)
patch_size = 5
center = True
# Parameters for training
train_steps = 100*15
batch_size = 128*10



# load_model=1

# if load_model:
#         # Load pre-trained weights if specified
#         weights = torch.load('MS_weights.pt')
#         patch_size = weights['patch_size']
#         model.load_state_dict(weights['net_state_dict'])
# else:
#         # Train the PatchNR model
#         train_patchNR(model, MS_test, patch_size, train_steps, batch_size, center=center)

print("here")
# print(Z_3d.shape[2])

step = 0.5

prev_obj_f = float('inf')
threshold = 0.5e-12


reg_values = []
# Measure the time taken for optimizing
start_time = time.time()

n_pat_list = [8000, 8000, 8000, 800, 80, 80, 80]
lam = 0.000000000025

# Define the range of kernel sizes
kernel_sizes = np.arange(6,12, 2)
print(kernel_sizes)
# Create lists to store the results
reg_values = []
mse_values = []
elapsed_times = []

for i, kernel_size in enumerate(kernel_sizes):
    # Initialize the progress bar manually since tqdm doesn't work well with PyTorch
    # progress_bar = tqdm(total=niter, desc='Iterations')
    # Set the current kernel size
    patch_size = kernel_size
    n_pat = n_pat_list[i]
    # Create the PatchNR model
    model = create_NF(num_layers=5, sub_net_size=512, dimension=patch_size**2)
    
    # Rest of your code ...
    upscaled_hs_image = upscale_hs_image(HS, target_shape)
    upscaled_hs_image = torch.reshape(upscaled_hs_image, (upscaled_hs_image.shape[0] * upscaled_hs_image.shape[1], upscaled_hs_image.shape[2]))
    Z = torch.matmul(upscaled_hs_image, U.T).detach().requires_grad_().to(DEVICE)
    Z_3d = torch.reshape(Z, (X0.shape[0], X0.shape[1], Z.shape[1])).to(DEVICE)
    optimizer = optim.Adam([Z], lr=step)
    # Train the PatchNR model
    train_patchNR(model, MS_test, patch_size, train_steps, batch_size, center=center)
    prev_mse = float('inf')

    input_im2pat = patch_extractor(patch_size, pad=False, center=center)
    reg_values_kernel = []
    mse_values_kernel = []

    # Measure the time taken for optimizing
    start_time = time.time()
    for iter in range(niter):
        # Minus gradient
        Z_3d = torch.reshape(Z, (X0.shape[0], X0.shape[1], Z.shape[1])).to(DEVICE)
        LZ = spatial_blur(Z_3d, B, subsamp)
        LZ = torch.reshape(LZ, (LZ.shape[0] * LZ.shape[1], LZ.shape[2])).to(DEVICE)

        reg = 0

        Img_reconstructed = torch.matmul(Z, U)
        Img_reconstructed = torch.reshape(Img_reconstructed, [X0.shape[0], X0.shape[1], Img_reconstructed.shape[1]])

        img_to_use = torch.reshape(Img_reconstructed, [Img_reconstructed.shape[2], Img_reconstructed.shape[0], Img_reconstructed.shape[1]]).unsqueeze(0)
        img_to_use=img_to_use[:,0:90:2,:,:]
        fake_data = input_im2pat(img_to_use, n_pat)
        # PatchNR regularization
        pred_inv, log_det_inv = model(fake_data, rev=True)
        reg += torch.mean(torch.sum(pred_inv**2, dim=1) / 2) #- torch.mean(log_det_inv)
        # reg_values.append(reg.item())  # Store the value of reg in the list    
        # reg= reg/Z_3d.shape[2]
        # print(reg)
        Obj_F = torch.mean((Y_M - torch.matmul(Z, UR)) ** 2) * sigmaM_inv / 2 + torch.mean((Y_H_Ut - LZ) ** 2) * sigmaH_inv / 2  + lam* reg

        optimizer.zero_grad()
        Obj_F.backward()
        optimizer.step()

        # error_list.append(Obj_F.item())  # Add the current error to the list

        if iter % 10 == 0:
            Img_reconstructed = torch.matmul(Z, U)
            Img_reconstructed = torch.reshape(Img_reconstructed, [X0.shape[0], X0.shape[1], Img_reconstructed.shape[1]])
            mse = torch.mean((X0 - Img_reconstructed) ** 2)

             # Append the values to the lists
            reg_values_kernel.append(reg.item())
            mse_values_kernel.append(mse.item())
            error_list.append(mse.item())
            print(f"Mean Squared Error: {mse.item()}")
            print(f"Objective function: {Obj_F.item()}")
            print(f"Regu: {reg.item()}")
            # Check if the change in objective function value is below the threshold
            # obj_f_change = prev_obj_f - Obj_F.item()
            # if obj_f_change < threshold:
            #     print("Stopping optimization: Change in objective function value is below the threshold.")
            #     break
            if mse.item() - prev_mse > 0: #mse.item() - prev_mse > 0:
                print("Stopping optimization: Change in objective function value is below the threshold.")
                break

            # prev_obj_f = Obj_F.item()# Update the previous error
            prev_mse = mse.item()# Update the previous error

        # progress_bar.update(1)  # Update progress bar
    # progress_bar.close()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    # Store the results for the current kernel size
    mse_values.append(mse_values_kernel)
    reg_values.append(reg_values_kernel)
    elapsed_times.append(elapsed_time)
    print("with kernel of " +  str(patch_size))
    quality_indices(Img_reconstructed, X0)

print("was it here")


special_name = "attempt_with_all_patch_number_8000"
# Plot the evolution of loss for each kernel size
for i, kernel_size in enumerate(kernel_sizes):
    plt.plot(range(0, len(mse_values[i]) * 10, 10), mse_values[i], label=f"Kernel Size {kernel_size}")

plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error')
plt.title('Evolution of Loss')
plt.legend()
output_file = "Graphs_Kernels/plots_MSE_"+ special_name + ".png"  # Specify the file path relative to the output folder
plt.savefig(output_file, dpi=300)
plt.show()

for i, kernel_size in enumerate(kernel_sizes):
    last_10_iterations = range(len(mse_values[i]) - 10, len(mse_values[i]))
    last_10_mse_values = mse_values[i][-10:]
    plt.plot(last_10_iterations, last_10_mse_values, label=f"Kernel Size {kernel_size}")

plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error')
plt.title('Evolution of Loss')
plt.legend()
output_file = "Graphs_Kernels/plots_MSE_detail"+ special_name + ".png"  # Specify the file path relative to the output folder
plt.savefig(output_file, dpi=300)
plt.show()


# plt.plot(range(0, len(mse_values[0]) * 10, 10), mse_values[0], label=f"Kernel Size {0}")
# plt.plot(range(0, len(mse_values[1]) * 10, 10), mse_values[1], label=f"Kernel Size {1}")
# plt.xlabel('Iteration')
# plt.ylabel('Mean Squared Error')
# plt.title('Evolution of Loss')
# plt.legend()
# plt.show()

# print(len(reg_values))
# print(len(reg_values[0]))
# print(len(reg_values[1]))
# Plot the evolution of regulation for each kernel size
for i, kernel_size in enumerate(kernel_sizes):
    plt.plot(range(0, len(reg_values[i]) * 10, 10), reg_values[i], label=f"Kernel Size {kernel_size}")

plt.xlabel('Iteration')
plt.ylabel('reg')
plt.title('Evolution of Regulation')
plt.legend()
output_file = "Graphs_Kernels/plots_regulation_"+ special_name + ".png"  # Specify the file path relative to the output folder
plt.savefig(output_file, dpi=300)
plt.show()



# Plot the evolution of MRE for each kernel size
plt.scatter(range(len(kernel_sizes)), elapsed_times)
plt.xticks(range(len(kernel_sizes)), kernel_sizes)  # Setting x-axis labels as kernel sizes
plt.xlabel('Kernel Size')
plt.ylabel('Time')
plt.title('Evolution of MRE')
plt.grid(True)
output_file = "Graphs_Kernels/plots_time_scatter_"+ special_name + ".png"  # Specify the file path relative to the output folder
plt.savefig(output_file, dpi=300)
plt.show()

