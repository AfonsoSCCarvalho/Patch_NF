import torch
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def spatial_blur(X_tensor, B_tensor, subsamp):
    # X_tensor = X_tensor.permute(2, 0, 1)  # Rearrange dimensions of X_tensor
    size_X_3 = X_tensor.shape[2]
    
    # B_tensor = B_tensor.repeat(size_X_1, 1, 1)
    # B_tensor = B_tensor.unsqueeze(0).repeat(1, 1, 1, 1)

    # Compute FFT of X_tensor along the first two dimensions
    X_fft = torch.fft.fft2(X_tensor, dim=(0, 1)).to(DEVICE)

    # Compute FFT of B_tensor
    B_fft = torch.fft.fft2(B_tensor).to(DEVICE)

    # Tile B_fft along the third dimension
    tiled_B_fft = B_fft.unsqueeze(2).repeat(1, 1, size_X_3).to(DEVICE)


    # print(X_fft.shape)
    # print(tiled_B_fft.shape)
    # Element-wise multiplication of X_fft and tiled_B_fft
    product = X_fft * tiled_B_fft

    # Compute the inverse FFT and take the real part
    Z= torch.fft.ifft2(product, dim=(0, 1)).real

    # Y = Y.permute(1, 2, 0)  # Rearrange dimensions back to (H, W, C)
    Z = Z[::subsamp, ::subsamp, :].to('cpu')  # Subsample the result

    return Z



# def spatial_blur(X_tensor, B_tensor, subsamp):
#     X_tensor = X_tensor.permute(2, 0, 1)  # Rearrange dimensions of X_tensor
#     size_X_1 = X_tensor.shape[0]
#
#     # B_tensor = B_tensor.repeat(size_X_1, 1, 1)
#     B_tensor = B_tensor.unsqueeze(0).repeat(1, 1, 1, 1)
#     # print("HEyyyy yo")
#     print(X_tensor.shape)
#     # print(B_tensor.shape)
#
#
#
#     # kernel_size = B_tensor.shape[2]
#     # padding = (kernel_size - 1) // 2
#     padding = (B_tensor.shape[2] - 1) // 2
#     padded_X = F.pad(X_tensor, (padding, padding, padding, padding), mode='reflect')
#     print("padded_X.shape")
#     print(padded_X.shape)
#     Z_list = []  # List to store individual channel convolutions
#
#     for channel_idx in range(size_X_1):
#         X_channel = padded_X[channel_idx, :, :].unsqueeze(0).unsqueeze(0)  # Extract current channel and add a singleton dimension
#         # B_tensor = B_tensor.unsqueeze(1)
#
#         # # Pad the channel
#         # padding = (B_tensor.shape[1] - 1) // 2
#         # padded_X_channel = F.pad(X_channel, (padding, padding, padding, padding), mode='reflect')
#
#         # Apply convolution on the current channel
#         Z_channel = F.conv2d(X_channel, B_tensor,stride=1, groups=1)
#
#         Z_list.append(Z_channel)
#
#     # Concatenate the convolutions along the channel dimension
#     Z = torch.cat(Z_list, dim=0)
#
#
#     # Z = F.conv2d(padded_X, B_tensor, groups=size_X_1)
#     # Z = torch.nn.functional.conv2d(X_tensor, B_tensor, padding=1, padding_mode='reflect', groups=93)
#
#     # Z = F.conv2d(X_tensor, B_tensor, padding='same', groups=93)
#     # Z = torch.nn.Conv2d(X_tensor, B_tensor, padding='same', padding_mode= ,groups=93)
#     # print("hye")
#     # print(Z.shape)  # Output shape: (1, 93, 256, 256)
#     # Z  = Z.permute(1, 2, 0)
#     Z = Z.permute(1,2,3,0).squeeze(0)
#     print(Z.shape)
#     Z = Z[::subsamp, ::subsamp, :]
#     print(Z.shape)
#
#     # Z = Z[:, ::subsamp, ::subsamp, :]
#     # print(Z.shape)  # Output shape: (1, 93, 256, 256)
#     return Z
