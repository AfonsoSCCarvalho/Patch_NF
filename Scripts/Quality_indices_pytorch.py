import torch
from torch import nn

import numpy as np

def quality_indices_torch(I_obt, I_REF, ratio=0, mask=None):

    mse_loss = nn.MSELoss()
    mse = mse_loss(I_REF, I_obt)
    print(f"Mean Squared Error (MSE): {mse.item()}")

    rmse = torch.sqrt(mse)
    print(f"Root Mean Squared Error (RMSE): {rmse.item()}")
    # rmse_value = rmse(I_REF, I_obt)
    # print(f"Root Mean Squared Error (RMSE): {rmse_value.item()}")

    max_pixel_value = torch.max(I_REF)
    # psnr = 20 * torch.log10(max_pixel_value / rmse_value.item())
    psnr = 20 * torch.log10(max_pixel_value / rmse)
    print(f"Peak signal-to-noise ratio (PSNR): {psnr.item()}")

    # psnr_all = calculate_psnr(I_REF, I_obt, mask)
    # print(psnr_all)

    # ssim_score = torch.tensor(0.0)  # Placeholder for SSIM score
    num_channels = I_obt.shape[2]

    ssim_sum = 0.0

    for i in range(num_channels):
        X_hat_channel = I_obt[:, :, i:i + 1]  # Extract individual channel from X_hat
        X_channel = I_REF[:, :, i:i + 1]  # Extract individual channel from X

        ssim_channel = ssim(X_hat_channel, X_channel, C1=0.01, C2=0.03)
        ssim_sum += ssim_channel

    average_ssim = ssim_sum / num_channels
    print(f"Average Structural similarity index measure(SSIM): {average_ssim.item()} º")
    # Calculate angle_SAM using PyTorch operations
    ref = I_REF.permute(2, 0, 1)
    tar = I_obt.permute(2, 0, 1)

    # Compute the dot product between reference and target HS data
    prod_scal = torch.sum(ref * tar, dim=0)

    # Compute the norm of the original and fused HS data
    norm_orig = torch.sum(ref * ref, dim=0)
    norm_fusa = torch.sum(tar * tar, dim=0)

    # Compute the product of the norms
    prod_norm = torch.sqrt(norm_orig * norm_fusa)

    # Avoid division by zero and compute the angle map
    prod_map = torch.where(prod_norm == 0, torch.finfo(torch.float32).eps, prod_norm)
    map = torch.acos(prod_scal / prod_map)

    # Flatten the arrays for further computation
    prod_scal = prod_scal.flatten()
    prod_norm = prod_norm.flatten()

    # Exclude zero norm values and compute the angle
    z = torch.nonzero(prod_norm)
    prod_scal = prod_scal[z]
    prod_norm = prod_norm[z]
    angolo = torch.sum(torch.acos(prod_scal / prod_norm)) / prod_norm.shape[0]
    angle_SAM = angolo.real * 180 / torch.tensor(np.pi)

    print(f"Spectral angle distance (SAD): {angle_SAM.item()} º")
    
    return mse, rmse, psnr, average_ssim , angle_SAM

def ssim(X_hat, X, C1=0.01, C2=0.03):
    # Calculate the means
    mean_X_hat = X_hat.mean()
    mean_X = X.mean()

    # Calculate the variances
    var_X_hat = ((X_hat - mean_X_hat) ** 2).mean()
    var_X = ((X - mean_X) ** 2).mean()

    # Calculate the covariance
    cov_X_hat_X = ((X_hat - mean_X_hat) * (X - mean_X)).mean()

    # Calculate the SSIM
    numerator = (2 * mean_X_hat * mean_X + C1) * (2 * cov_X_hat_X + C2)
    denominator = (mean_X_hat ** 2 + mean_X ** 2 + C1) * (var_X_hat + var_X + C2)
    ssim_val = numerator / denominator

    return ssim_val


# def rmse(ref, tar, mask=None):
#     # Get dimensions of the input arrays
#     rows, cols, bands = ref.shape

#     if mask is None:
#         # Calculate RMSE for the whole array
#         squared_error = torch.sum((tar - ref) ** 2)
#         out = torch.sqrt(squared_error / (rows * cols * bands))
#     else:
#         # Calculate RMSE for each band separately
#         squared_error = torch.sum((tar - ref) ** 2, dim=2)
#         out = {}
#         out['rmse_map'] = torch.sqrt(torch.sum(squared_error, dim=2) / bands)

#         # Flatten the arrays to (num_pixels, bands) for masked calculations
#         ref_flat = ref[mask != 0].reshape(-1, bands)
#         tar_flat = tar[mask != 0].reshape(-1, bands)

#         # Calculate RMSE for masked pixels
#         squared_error_masked = torch.sum((tar_flat - ref_flat) ** 2)
#         out['ave'] = torch.sqrt(squared_error_masked / (torch.sum(mask != 0) * bands))

#     return out

def calculate_psnr(ref, tar, mask = None):
    bands = ref.shape[2]

    # Flatten the ref and tar tensors to (num_pixels, bands) for masked calculations
    ref_flat = ref[mask != 0].reshape(-1, bands)
    tar_flat = tar[mask != 0].reshape(-1, bands)

    # Calculate the mean squared error (msr) for each band
    msr = torch.mean((ref_flat - tar_flat) ** 2, dim=0)

    # Calculate the maximum squared value for each band
    max2 = torch.max(ref, dim=0).values ** 2

    # Calculate the PSNR for each band
    psnr_all = 10 * torch.log10(max2 / msr)

    return torch.mean(psnr_all)