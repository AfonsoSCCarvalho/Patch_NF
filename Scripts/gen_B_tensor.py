import torch



def gen_B(nr, nc):

    # Set the parameters
    lx = 2  # spatial size along x direction
    ly = 2  # spatial size along y direction
    sx = 2.5  # standard deviation along x direction
    sy = 2.5  # standard deviation along y direction

    # Define the size of the kernel
    # nr = ny = 5
    # nc = nx = 5

    # Define the center of the kernel
    mid_col = (nc + 1) // 2  # middle column
    mid_row = (nr + 1) // 2  # middle row

    # Initialize the kernel with zeros
    B = torch.zeros(nr, nc)

    # If lx is greater than 0, we generate a 2D Gaussian kernel with the given
    # standard deviation values sx and sy. Otherwise, if lx is 0, we generate
    # a single point kernel with a value of 1.
    if lx > 0:
        for i in range(-ly, ly + 1):
            for j in range(-lx, lx + 1):
                # Calculate the value of the 2D Gaussian at the current (i,j) location
                B[mid_row + i, mid_col + j] = torch.exp(torch.tensor(-((i / sy) ** 2 + (j / sx) ** 2) / 2))
    else:
        B[mid_row, mid_col] = 1

    # Circularly center B_tensor
    B = torch.fft.ifftshift(B)

    # Normalize B_tensor
    B = B / torch.sum(B)
    return B