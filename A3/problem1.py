#Group 23: Gustavo Willner 2708177, Pedro Campana 2461919, Helge Meier 2465180

import math
import gco
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
np.random.seed(seed=2022)

def mrf_denoising_nllh(x, y, sigma_noise):
    """Elementwise negative log likelihood.

      Args:
        x: candidate denoised image
        y: noisy image
        sigma_noise: noise level for Gaussian noise

      Returns:
        A `nd.array` with dtype `float32/float64`.
    """
    
    # Missing summation for equation (1) but only this way is is still an array
    nllh = -(1/(2*sigma_noise))*(x-y)**2 
    
    assert (nllh.dtype in [np.float32, np.float64])
    return nllh

def edges4connected(height, width):
    """Construct edges for 4-connected neighborhood MRF.
    The output representation is such that output[i] specifies two indices
    of connected nodes in an MRF stored with row-major ordering.

      Args:
        height, width: size of the MRF.

      Returns:
        A `nd.array` with dtype `int32/int64` of size |E| x 2.
    """
    E = 2 * (height * width) - (height + width)
    edges = np.zeros((E, 2), dtype=np.int32)

    # rmo: Convert 2D coordinates (h, w) to a 1D index using row-major order
    rmo = lambda h, w: h * width + w

    index = 0
    for h in range(height):
        for w in range(width):
            if w < width - 1:
                # Connect node (h, w) to (h, w+1)
                edges[index][0] = rmo(h, w)
                edges[index][1] = rmo(h, w + 1)
                index += 1
            if h < height - 1:
                # Connect node (h, w) to (h+1, w)
                edges[index][0] = rmo(h, w)
                edges[index][1] = rmo(h + 1, w)
                index += 1
    assert (edges.shape[0] == 2 * (height*width) - (height+width) and edges.shape[1] == 2)
    assert (edges.dtype in [np.int32, np.int64])
    return edges

def my_sigma():
    return 2

def my_lmbda():
    return 40

def alpha_expansion(noisy, init, edges, candidate_pixel_values, s, lmbda):
    """ Run alpha-expansion algorithm.

      Args:
        noisy: Given noisy grayscale image.
        init: Image for denoising initilisation
        edges: Given neighboor of MRF.
        candidate_pixel_values: Set of labels to consider
        s: sigma for likelihood estimation
        lmbda: Regularization parameter for Potts model.

      Runs through the set of candidates and iteratively expands a label.
      If there have been recorded changes, re-run through the complete set of candidates.
      Stops, if there are no changes in the labelling.

      Returns:
        A `nd.array` of type `int32`. Assigned labels minimizing the costs.
    """
    def unary_potential(x, y, sigma_noise):
        """ Compute the unary potential using negative log likelihood. """
        return mrf_denoising_nllh(x, y, sigma_noise)

    def pairwise_potential(label1, label2, lmbda):
        """ Compute the pairwise potential for a pair of labels. """
        return 0 if label1 == label2 else lmbda

    height, width = noisy.shape
    labels = init.copy()
    curentPNSR = 0
    converged = False

    while True:
        change_made = False
        # Loop from 1 to 225 possible pixel values
        for alpha in candidate_pixel_values:

            ## Make Unary matrix
            unary = np.zeros((height, width, 2))
            for i in range(height):
                for j in range(width):
                    unary[i, j, 0] = unary_potential(labels[i, j], noisy[i, j], s)
                    unary[i, j, 1] = unary_potential(alpha, noisy[i, j], s)
            unary = np.transpose(unary.reshape(-1, 2))

            ## Make Pairwise Matrix
            pairwise_row = []
            pairwise_col = []
            pairwise_data = []
            for edge in edges:
                i, j = edge
                pairwise_row.append(i)
                pairwise_col.append(j)
                pairwise_data.append(pairwise_potential(labels.flat[i], labels.flat[j], lmbda))
            pairwise = csr_matrix((pairwise_data, (pairwise_row, pairwise_col)), shape=(height * width, height * width))

            ## Get the new labels (0 or 1) and update our current labels (0 to 225)
            new_labels = gco.graphcut(unary, pairwise)
            old_img = labels.copy()
            for i in range(height):
                for j in range(width):
                    idx = i * width + j
                    if new_labels[idx] == 0:
                        labels[i, j] = alpha

            if np.any(labels != old_img):
                change_made = True
                newPNSR = compute_psnr(labels, gt)
                print("New PSNR: ")
                print(newPNSR)
                if curentPNSR >= newPNSR:
                    converged = True
                    # Stop the algorithm when it converges
                    break
                curentPNSR = newPNSR
        # Besides convergence check if changes were made (this logic is probably not working properly)
        if not change_made or converged:
            break

    denoised = labels.astype(init.dtype)
    assert (np.equal(denoised.shape, init.shape).all())
    assert (denoised.dtype == init.dtype)
    return denoised


def compute_psnr(img1, img2):
    """Computes PSNR between img1 and img2"""
    assert img1.shape == img2.shape, "Input images must have the same dimensions."

    # Calculate Mean Squared Error
    mse = np.mean((img1 - img2) ** 2)

    # Maximum possible pixel value of the image
    v_max = max(img1.max(), img2.max())

    # If MSE is zero (images are identical), return infinity
    if mse == 0:
        return float('inf')

    # Compute PSNR
    psnr = 10 * np.log10((v_max ** 2) / mse)

    return psnr


def show_images(i0, i1):
    """
    Visualize estimate and ground truth in one Figure.
    Only show the area for valid gt values (>0).
    """

    # Crop images to valid ground truth area
    row, col = np.nonzero(i0)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(i0, "gray", interpolation='nearest')
    plt.subplot(1,2,2)
    plt.imshow(i1, "gray", interpolation='nearest')
    plt.show()

# Example usage in main()
# Feel free to experiment with your code in this function
# but make sure your final submission can execute this code
if __name__ == '__main__':
    # Read images
    noisy = ((255 * plt.imread('data/la-noisy.png')).squeeze().astype(np.int32)).astype(np.float32)
    gt = (255 * plt.imread('data/la.png')).astype(np.int32)

    lmbda = my_lmbda()
    s = my_sigma()

    # Create 4 connected edge neighborhood
    edges = edges4connected(noisy.shape[0], noisy.shape[1])

    # Candidate search range
    labels = np.arange(0, 255)

    # Graph cuts with random initialization
    random_init = np.random.randint(low=0, high=255, size=noisy.shape)
    estimated = alpha_expansion(noisy, random_init, edges, labels, s, lmbda)
    show_images(noisy, estimated)
    psnr_before = compute_psnr(noisy, gt)
    psnr_after = compute_psnr(estimated, gt)
    print(psnr_before, psnr_after)