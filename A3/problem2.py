from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as tf
import torch.optim as optim

from utils import flow2rgb
from utils import rgb2gray
from utils import read_flo
from utils import read_image

np.random.seed(seed=2022)


def numpy2torch(array):
    """ Converts 3D numpy (H,W,C) ndarray to 3D PyTorch (C,H,W) tensor.

    Args:
        array: numpy array of shape (H, W, C)
    
    Returns:
        tensor: torch tensor of shape (C, H, W)
    """
    array = np.transpose(array,(2,0,1))
    tensor = torch.from_numpy(array)

    return tensor


def torch2numpy(tensor):
    """ Converts 3D PyTorch (C,H,W) tensor to 3D numpy (H,W,C) ndarray.

    Args:
        tensor: torch tensor of shape (C, H, W)
    
    Returns:
        array: numpy array of shape (H, W, C)
    """
    array = tensor.numpy()
    array = np.transpose(array,(1,2,0))
    
    return array


def load_data(im1_filename, im2_filename, flo_filename):
    """Loading the data. Returns 4D tensors. You may want to use the provided helper functions.

    Args:
        im1_filename: path to image 1
        im2_filename: path to image 2
        flo_filename: path to the ground truth flow
    
    Returns:
        tensor1: torch tensor of shape (B, C, H, W)
        tensor2: torch tensor of shape (B, C, H, W)
        flow_gt: torch tensor of shape (B, C, H, W)
    """
    
    img1 = rgb2gray(read_image(im1_filename))
    img2 = rgb2gray(read_image(im2_filename))
    flo = read_flo(flo_filename)
    
    tensor1 = numpy2torch(img1)
    tensor2 = numpy2torch(img2)
    
    tensor1 = tensor1.unsqueeze(0)
    tensor2 = tensor2.unsqueeze(0)
    
    flow_gt = numpy2torch(flo)
    flow_gt = flow_gt.unsqueeze(0)
    
    return tensor1, tensor2, flow_gt


def evaluate_flow(flow, flow_gt):
    """Evaluate the average endpoint error w.r.t the ground truth flow_gt.
    Excludes pixels, where u or v components of flow_gt have values > 1e9.

    Args:
        flow: torch tensor of shape (B, C, H, W)
        flow_gt: torch tensor of shape (B, C, H, W)
    
    Returns:
        aepe: torch tensor scalar 
    """
    flow = flow.detach()
    epe = np.sqrt((flow[:,0,:,:]-flow_gt[:,0,:,:])**2 + (flow[:,1,:,:]-flow_gt[:,1,:,:])**2)
    valid = (flow_gt[:, 0, :, :] <= 1e9) & (flow_gt[:, 1, :, :] <= 1e9)
    epe = epe * valid

    aepe = torch.sum(epe) / torch.sum(valid.float())
    
    return aepe

def visualize_warping_practice(im1, im2, flow_gt):
    """ Visualizes the result of warping the second image by ground truth.

    Args:
        im1: torch tensor of shape (B, C, H, W)
        im2: torch tensor of shape (B, C, H, W)
        flow_gt: torch tensor of shape (B, C, H, W)
    
    Returns:

    """
    im2w = warp_image(im2,flow_gt)
    
    im1_np2d = torch2numpy(im1[0,:,:,:])[:,:,0]
    im2w_np2d = torch2numpy(im2w[0,:,:,:])[:,:,0]
    diff_im = im1_np2d-im2w_np2d

    fig=plt.figure()
    fig.add_subplot(311)
    plt.imshow(im1_np2d, cmap="gray")
    fig.add_subplot(312)
    plt.imshow(im2w_np2d, cmap="gray")
    fig.add_subplot(313)
    plt.imshow(diff_im, cmap="gray")
    
    plt.show()
    return


def warp_image(im, flow):
    """ Warps given image according to the given optical flow.

    Args:
        im: torch tensor of shape (B, C, H, W)
        flow: torch tensor of shape (B, C, H, W)
    
    Returns:
        x_warp: torch tensor of shape (B, C, H, W)
    """
    (B,C,H,W) = im.shape
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    grid_x = grid_x.to(im.device)
    grid_y = grid_y.to(im.device)
    
    # Stack and repeat grid
    grid = torch.stack((grid_x, grid_y), dim=0).float()  # Shape (2, H, W)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # Shape (B, 2, H, W)
    
    # Add flow to grid
    vgrid = grid + flow
    
    # Normalize grid
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    
    # Permute grid to (B, H, W, 2)
    vgrid = vgrid.permute(0, 2, 3, 1)
    
    # Warp image using grid
    x_warp = tf.grid_sample(im, vgrid, align_corners=True)
    
    return x_warp


def energy_hs(im1, im2, flow, lambda_hs):
    """ Evalutes Horn-Schunck energy function.

    Args:
        im1: torch tensor of shape (B, C, H, W)
        im2: torch tensor of shape (B, C, H, W)
        flow: torch tensor of shape (B, C, H, W)
        lambda_hs: float
    
    Returns:
        energy: torch tensor scalar
    """
    im2w = warp_image(im2,flow)
    E1 = (im2w-im1)**2
    E1 = E1.sum()

    u = flow[:, 0,:,:]
    v = flow[:,1,:,:]
    u_x = tf.pad(u[:, :, 1:] - u[:, :, :-1], (0, 1, 0, 0))
    u_y = tf.pad(u[:, 1:, :] - u[:, :-1, :], (0, 0, 0, 1))
    v_x = tf.pad(v[:, :, 1:] - v[:, :, :-1], (0, 1, 0, 0))
    v_y = tf.pad(v[:, 1:, :] - v[:, :-1, :], (0, 0, 0, 1))
    smoothness_term = (u_x ** 2 + u_y ** 2 + v_x ** 2 + v_y ** 2).sum()
    energy = E1 + lambda_hs * smoothness_term
    return energy


def estimate_flow(im1, im2, flow_gt, lambda_hs, learning_rate, num_iter):
    """
    Estimate flow using HS with Gradient Descent.
    Displays average endpoint error.
    Visualizes flow field.

    Args:
        im1: torch tensor of shape (B, C, H, W)
        im2: torch tensor of shape (B, C, H, W)
        flow_gt: torch tensor of shape (B, C, H, W)
        lambda_hs: float
        learning_rate: float
        num_iter: int
    
    Returns:
        aepe: torch tensor scalar
    """
    B, C, H, W = im1.size()
    flow = torch.zeros(B, 2, H, W, requires_grad=True).to(im1.device)
    optimizer = torch.optim.SGD([flow], lr=learning_rate)
    initial_aepe = evaluate_flow(flow, flow_gt)
    print(f"Initial AEPE: {initial_aepe.item()}")
    for _ in range(num_iter):
        optimizer.zero_grad()
        loss = energy_hs(im1, im2, flow, lambda_hs)
        loss.backward()
        optimizer.step()
    aepe = evaluate_flow(flow, flow_gt)
    print(f"Final AEPE: {aepe.item()}")
    print(flow.permute(0,2,3,1).detach().cpu().numpy()[0,:,:,:].shape)
    flow_rgb = flow2rgb(flow.permute(0,2,3,1).detach().cpu().numpy()[0,:,:,:])
    plt.figure(figsize=(10, 5))
    for i in range(B):
        plt.subplot(1, B, i + 1)
        print(flow_rgb.shape)
        plt.imshow(flow_rgb)
        plt.title(f"Flow Visualization {i+1}")
    plt.show()
    return aepe

# Example usage in main()
# Feel free to experiment with your code in this function
# but make sure your final submission can execute this code
def main():

    # Loading data
    im1, im2, flow_gt = load_data("data/frame10.png", "data/frame11.png", "data/flow10.flo")

    # Parameters
    lambda_hs = 0.002
    num_iter = 500

    # Warping_practice
    visualize_warping_practice(im1, im2, flow_gt)
    
    # Gradient descent
    #
    learning_rate = 18
    estimate_flow(im1, im2, flow_gt, lambda_hs, learning_rate, num_iter)


if __name__ == "__main__":
    main()
