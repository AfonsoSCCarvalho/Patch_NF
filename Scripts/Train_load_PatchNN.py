#Path towards Common Scripts
import sys

import torch
import torch.optim as optim

from TrainingPatchesMS import *
from utils import *
from model import create_NF

def train_and_load_PatchNN(MS, patch_size, train_steps, batch_size, center=True, load_model=True, special_train_name='', plot_loss = False):
    MS_test = MS[:,:,3].unsqueeze(0)

    MS_test = MS_test.unsqueeze(0).to(DEVICE)
    print(MS_test.shape)
    print("Above")
    # Create the PatchNR model
    model = create_NF(num_layers=5, sub_net_size=512, dimension=patch_size**2)

    if load_model:
        # Load pre-trained weights if specified
        weights = torch.load('Kernels/MS_weights_patchsize' + str(patch_size) + '_.pt')
        patch_size = weights['patch_size']
        model.load_state_dict(weights['net_state_dict'])

        if plot_loss:
            # Plot the loss curve (optional)
            loss_values = torch.load('Kernels/loss_values_patchsize' + str(patch_size) + '_.pt')
            plt.plot(loss_values)
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.title('Loss Curve')
            plt.show()
    else:
        # Train the PatchNR model
        train_patchNR(model, MS_test, patch_size, train_steps, batch_size, center=center, special_name=special_train_name, plot_loss_train = plot_loss)
    return model

