from PIL import Image
import numpy as np
np.random.seed(seed=2024)

def load_data(gt_path):
    """Loading the data.
    The same as loading the disparity from Assignment 1 
    (not graded here).

    Args:
        gt_path: path to the disparity image
    
    Returns:
        g_t: numpy array of shape (H, W)
    """
    g_t = (np.array(Image.open(gt_path)).astype(np.float64) / 255.0) * 16
    return g_t

def random_disparity(disparity_size):
    """Compute a random disparity map.

    Args:
        disparity_size: tuple containg height and width (H, W)

    Returns:
        disparity_map: numpy array of shape (H, W)

    """
    disparity_map = np.random.randint(0, 15, disparity_size)
    return disparity_map

def constant_disparity(disparity_size, a):
    """Compute a constant disparity map.

    Args:
        disparity_size: tuple containg height and width (H, W)
        a: the value to initialize with

    Returns:
        disparity_map: numpy array of shape (H, W)

    """
    disparity_map = np.full(disparity_size,a)
    return disparity_map


def log_gaussian(x, mu, sigma):
    """Compute the log gaussian of x.

    Args:
        x: numpy array of shape (H, W) (np.float64)
        mu: float
        sigma: float

    Returns:
        result: numpy array of shape (H, W) (np.float64)

    """
    result = - ((x-mu)**2)/(2*sigma**2) # here we are not including the constant -0.5 * np.log(2 * np.pi * sigma**2) because we are following a definition from the lecture slides where this constant is not given
    return result


def mrf_log_prior(x, mu, sigma):
    """Compute the log of the unnormalized MRF prior density.

    Args:
        x: numpy array of shape (H, W) (np.float64)
        mu: float
        sigma: float

    Returns:
        logp: float

    """
    x_i_diff = np.diff(x, n=1, axis=1)
    y_i_diff = np.diff(x, n=1, axis=0)
    
    # gaussian potentials
    x_i_potential = np.sum(log_gaussian(x_i_diff, mu, sigma))
    y_i_potential = np.sum(log_gaussian(y_i_diff, mu, sigma))
    
    logp = x_i_potential + y_i_potential
    return logp

# Example usage in main()
# Feel free to experiment with your code in this function
# but make sure your final submission can execute this code
def main():

    gt = load_data('./data/gt.png')

    # Display log prior of GT disparity map
    logp = mrf_log_prior(gt, mu=0, sigma=3.1)
    print("Log Prior of GT disparity map:", logp)

    # Display log prior of random disparity ma
    random_disp = random_disparity(gt.shape)
    logp = mrf_log_prior(random_disp, mu=0, sigma=3.1)
    print("Log-prior of noisy disparity map:",logp)

    # Display log prior of constant disparity map
    constant_disp = constant_disparity(gt.shape, 6)
    logp = mrf_log_prior(constant_disp, mu=0, sigma=3.1)
    print("Log-prior of constant disparity map:", logp)

if __name__ == "__main__":
	main()

'''
The log prior of the constant disparity is zero. This is expected because there is no difference between disparity in different pixels and the prior is based on a pixel wise substraction of these disparities

The log prior of the random disparity is smaller than for the ground truth. This can be explained by the fact that the differences between neighbouring pixels are larger due to their random values. The ground truth values have more similiarity in their neighbourhood.

Varying the variance of the gaussian leads to smaller log prior for smaller values of sigma and vice versa. This is expected due to the fact that values that are more spread out have a higher probability given a higher variance and a smaller probability given a lower variance

sigma = 0.1
Log Prior of GT disparity map: -24144.90426758938
Log-prior of noisy disparity map: -411079799.99999994
Log-prior of constant disparity map: 0.0

sigma = 1.1
Log Prior of GT disparity map: -199.54466336850732
Log-prior of noisy disparity map: -3397353.719008263
Log-prior of constant disparity map: 0.0

sigma = 2.1
Log Prior of GT disparity map: -54.750349813127855
Log-prior of noisy disparity map: -932153.7414965986
Log-prior of constant disparity map: 0.0

sigma = 3.1
Log Prior of GT disparity map: -25.124770309666374
Log-prior of noisy disparity map: -427762.5390218522
Log-prior of constant disparity map: 0.0
'''

