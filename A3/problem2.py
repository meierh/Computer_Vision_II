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
    array = np.tranpose(array,(2,0,1))
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
    array = np.tranpose(array,(1,2,0))
    
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
    flo = read_flo(flow_filename)
    
    tensor1 = numpy2torch(img1)
    tensor2 = numpy2torch(img2)
    
    tensor1 = tf.expand_dims(tensor1,axis=0)
    tensor2 = tf.expand_dims(tensor2,axis=0)
    
    flow_gt = numpy2torch(flo)
    flow_gt = tf.expand_dims(flow_gt,axis=0)
    
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
    epe = np.sqrt((flow[:,0,:,:]-flow_gt[:,0,:,:])**2 + (flow[:,1,:,:]-flow_gt[:,1,:,:])**2)
    epe = tf.where(epe<1e9,x,0)
    aepe = tf.mean(epe)
    
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
    plt.imshow(im1_np2d)
    fig.add_subplot(312)
    plt.imshow(im2w_np2d)
    fig.add_subplot(313)
    plt.imshow(diff_im)
    
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
    
    ww = torch.arange(0, W).view(1 ,-1).repeat(H ,1)
    hh = torch.arange(0, H).view(-1 ,1).repeat(1 ,W)
    ww = ww.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
    hh = hh.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
    grid = torch.cat((ww ,hh) ,1).float()

    vgrid = Variable(grid) + flo

    vgrid[: ,0 ,: ,:] = 2.0 *vgrid[: ,0 ,: ,:].clone() / max( W -1 ,1 ) -1.0
    vgrid[: ,1 ,: ,:] = 2.0 *vgrid[: ,1 ,: ,:].clone() / max( H -1 ,1 ) -1.0

    vgrid = vgrid.permute(0 ,2 ,3 ,1)
    flo = flo.permute(0 ,2 ,3 ,1)
    output = tf.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size()))
    mask = tf.grid_sample(mask, vgrid)

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
    E1 = tf.sum(E1)

    #E2 = ....

    energy = E1
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
    learning_rate = 18
    estimate_flow(im1, im2, flow_gt, lambda_hs, learning_rate, num_iter)


if __name__ == "__main__":
    main()
