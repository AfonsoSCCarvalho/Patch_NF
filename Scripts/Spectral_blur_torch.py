import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def spectral_blur(X_tensor, R_tensor):
    # Reshape X to a 2D tensor
    X_vec = torch.reshape(X_tensor, (X_tensor.shape[0]* X_tensor.shape[1], X_tensor.shape[2])).to(DEVICE)

    # Apply PSF to X by matrix multiplication
    Z_vec = torch.matmul(R_tensor.to(DEVICE), X_vec.permute(1,0)).to(DEVICE)

    # Reshape the result back to a 3D hyperspectral image
    Z = torch.reshape(Z_vec.permute(1,0), (X_tensor.shape[0], X_tensor.shape[1], R_tensor.shape[0])).to('cpu')

    return Z


