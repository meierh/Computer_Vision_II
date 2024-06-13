import numpy as np

from scipy import interpolate   # Use this for interpolation
from scipy import signal        # Feel free to use convolutions, if needed
from scipy import optimize      # For gradient-based optimisation
from PIL import Image           # For loading images
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from skimage.transform import pyramid_gaussian, resize

# for experiments with different initialisation
from problem1 import random_disparity, mrf_log_prior, constant_disparity


def rgb2gray(rgb):
    """Converting RGB image to greyscale.
    The same as in Assignment 1 (no graded here).

    Args:
        rgb: numpy array of shape (H, W, 3)

    Returns:
        gray: numpy array of shape (H, W)

    """
    weights = np.array([0.2126, 0.7152, 0.0722])
    gray = np.dot(rgb[...,:3], weights)
    return gray


def load_data(i0_path, i1_path, gt_path):
    """Loading the data.
    The same as in Assignment 1 (not graded here).

    Args:
        i0_path: path to the first image
        i1_path: path to the second image
        gt_path: path to the disparity image
    
    Returns:
        i_0: numpy array of shape (H, W)
        i_1: numpy array of shape (H, W)
        g_t: numpy array of shape (H, W)
    """
    i_0 = np.array(Image.open(i0_path)).astype(np.float64) / 255.0
    i_1 = np.array(Image.open(i1_path)).astype(np.float64) / 255.0
    g_t = (np.array(Image.open(gt_path)).astype(np.float64) / 255.0) * 16
    return i_0, i_1, g_t

def log_gaussian(x,  mu, sigma):
    """Calcuate the value and the gradient w.r.t. x of the Gaussian log-density

    Args:
        x: numpy.float 2d-array
        mu and sigma: scalar parameters of the Gaussian

    Returns:
        value: value of the log-density
        grad: gradient of the log-density w.r.t. x
    """
    # return the value and the gradient
    value = np.log(np.sqrt(2*np.pi*sigma**2)) + -0.5*((x-mu)/sigma)**2
    #grad = ((x-mu)*mu)/(sigma**2)
    grad = -(x - mu) / sigma**2
    return value, grad

def stereo_log_prior(x, mu, sigma):
    """Evaluate gradient of pairwise MRF log prior with Gaussian distribution

    Args:
        x: numpy.float 2d-array (disparity)

    Returns:
        value: value of the log-prior
        grad: gradient of the log-prior w.r.t. x
    """
    x=x.astype(np.float64)
    value = mrf_log_prior(x, mu, sigma)  # log of the unnormalized MRF prior density
    # gradient of the log-density
    x_i_diff = np.diff(x, n=1, axis=1)
    y_i_diff = np.diff(x, n=1, axis=0)

    _, grad_x = log_gaussian(x_i_diff, mu, sigma)
    _, grad_y = log_gaussian(y_i_diff, mu, sigma)
    # total gradient
    grad = np.zeros_like(x)
    grad[:, :-1] -= grad_x # subtracts the gradient contribution from the left pixel to the right pixel
    grad[:, 1:] += grad_x # adds the gradient contribution from the left pixel to the right pixel to the right pixel
    grad[:-1, :] -= grad_y
    grad[1:, :] += grad_y
    return value, grad

def shift_interpolated_disparity(im1, d):
    """Shift image im1 by the disparity value d.
    Since disparity can now be continuous, use interpolation.

    Args:
        im1: numpy.float 2d-array  input image
        d: numpy.float 2d-array  disparity

    Returns:
        im1_shifted: Shifted version of im1 by the disparity value.
    """
    height, width = im1.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))  # create a grid of pixel indices
    # subtract the disparity map from the x indices since we are dealing with horizontal shifts between the left and right images
    x = x - d
    # ensure we won't try to access pixels that are outside the boundaries of the image
    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)
    interpolator = RegularGridInterpolator((np.arange(height), np.arange(width)), im1, method='linear', fill_value=0)
    coords = np.stack((y, x), axis=-1)
    im1_shifted = interpolator(coords)
    return im1_shifted

def stereo_log_likelihood(x, im0, im1, mu, sigma):
    """Evaluate gradient of the log likelihood.

    Args:
        x: numpy.float 2d-array of the disparity
        im0: numpy.float 2d-array of image #0
        im1: numpy.float 2d-array of image #1

    Returns:
        value: value of the log-likelihood
        grad: gradient of the log-likelihood w.r.t. x

    Hint: Make use of shift_interpolated_disparity and log_gaussian
    """

    # Shift the second image according to the disparity map
    im1_shifted = shift_interpolated_disparity(im1, x)
    # Calculate the difference between the first image and the shifted second image
    diff = im0 - im1_shifted
    # Calculate the log-density (log-likelihood) and its gradient
    log_likelihood, grad_log_likelihood = log_gaussian(diff, mu, sigma)

    # Sum the log-likelihood values for the total log-likelihood
    value = np.sum(log_likelihood)
    grad = grad_log_likelihood * (im1_shifted - np.roll(im1_shifted, 1, axis=1))

    return value, grad



def stereo_log_posterior(d, im0, im1, mu, sigma, alpha):
    """Computes the value and the gradient of the log-posterior

    Args:
        d: numpy.float 2d-array of the disparity
        im0: numpy.float 2d-array of image #0
        im1: numpy.float 2d-array of image #1

    Returns:
        value: value of the log-posterior
        grad: gradient of the log-posterior w.r.t. x
    """
    d = d.reshape(im0.shape)
    likelihood, likelihood_grad = stereo_log_likelihood(d,im0,im1,mu,sigma)
    prior, prior_grad = stereo_log_prior(d,mu,sigma)
    log_posterior = likelihood+alpha*prior
    value = -log_posterior.sum()
    log_posterior_grad = likelihood_grad+alpha*prior_grad
    grad = -log_posterior_grad.flatten()
    return value, grad


def optim_method():
    """Simply returns the name (string) of the method 
    accepted by scipy.optimize.minimize, that you found
    to work well.
    This is graded with 1 point unless the choice is arbitrary/poor.
    """
    return "L-BFGS-B"

def stereo(d0, im0, im1, mu, sigma, alpha, method=optim_method()):
    """Estimating the disparity map

    Args:
        d0: numpy.float 2d-array initialisation of the disparity
        im0: numpy.float 2d-array of image #0
        im1: numpy.float 2d-array of image #1

    Returns:
        d: numpy.float 2d-array estimated value of the disparity
    """
    # Run the optimization

    result = minimize(stereo_log_posterior, d0.flatten(), args=(im0, im1, mu, sigma, alpha), method=method, jac=True)
    d = result.x.reshape(im0.shape)
    return d

def coarse2fine(d0, im0, im1, mu, sigma, alpha, num_levels):
    """Coarse-to-fine estimation strategy. Basic idea:
        1. create an image pyramid (of size num_levels)
        2. starting with the lowest resolution, estimate disparity
        3. proceed to the next resolution using the estimated 
        disparity from the previous level as initialisation

    Args:
        d0: numpy.float 2d-array initialisation of the disparity
        im0: numpy.float 2d-array of image #0
        im1: numpy.float 2d-array of image #1

    Returns:
        pyramid: a list of size num_levels containing the estimated
        disparities at each level (from finest to coarsest)
        Sanity check: pyramid[0] contains the finest level (highest resolution)
                      pyramid[-1] contains the coarsest level
    """
    pyramid0 = list(pyramid_gaussian(im0, max_layer=num_levels-1, downscale=2.5))
    pyramid1 = list(pyramid_gaussian(im1, max_layer=num_levels-1, downscale=2.5))
    pyramid = [None] * num_levels
    pyramid[-1] = resize(d0, pyramid0[-1].shape, mode='reflect', anti_aliasing=True)

    for level in range(num_levels-1, -1, -1):
        im0_level = pyramid0[level]
        im1_level = pyramid1[level]
        d_init = pyramid[level]
        d_optimized = stereo(d_init, im0_level, im1_level, mu, sigma, alpha)
        pyramid[level] = d_optimized

        if level > 0:
            d_upscale = resize(d_optimized, pyramid0[level-1].shape, mode='reflect', anti_aliasing=True) * 2
            pyramid[level-1] = d_upscale

    return pyramid

# Example usage in main()
# Feel free to experiment with your code in this function
# but make sure your final submission can execute this code
def main():

    # these are the same functions from Assignment 1
    # (no graded in this assignment)
    im0, im1, gt = load_data('./data/i0.png', './data/i1.png', './data/gt.png')
    im0, im1 = rgb2gray(im0), rgb2gray(im1)

    mu = 0.0
    sigma = 1.0

    # experiment with other values of alpha
    alpha = 1.0

    # initial disparity map
    # experiment with constant/random values
    d0 = gt
    #d0 = random_disparity(gt.shape)
    #d0 = constant_disparity(gt.shape, 6)

    # Display stereo: Initialized with noise
    disparity = stereo(d0, im0, im1, mu, sigma, alpha)

    # Pyramid
    num_levels = 3
    pyramid = coarse2fine(d0, im0, im1, mu, sigma, alpha, num_levels)

if __name__ == "__main__":
    main()
