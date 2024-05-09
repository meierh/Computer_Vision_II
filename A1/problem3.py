
from PIL import Image
import numpy as np
np.random.seed(seed=2022)

# convert a RGB image to grayscale
# input (rgb): numpy array of shape (H, W, 3)
# output (gray): numpy array of shape (H, W)
def rgb2gray(rgb):
	weights = np.array([0.2126, 0.7152, 0.0722])
	gray = np.dot(rgb[...,:3], weights)
	return gray

# load the data
# input (i0_path): path to the first image
# input (i1_path): path to the second image
# input (gt_path): path to the disparity image
# output (i_0): numpy array of shape (H, W, 3)
# output (i_1): numpy array of shape (H, W, 3)
# output (g_t): numpy array of shape (H, W)
def load_data(i0_path, i1_path, gt_path):
	i_0 = np.array(Image.open(i0_path)).astype(np.float64) / 255.0
	i_1 = np.array(Image.open(i1_path)).astype(np.float64) / 255.0
	g_t = (np.array(Image.open(gt_path)).astype(np.float64) / 255.0) * 16
	return i_0, i_1, g_t

# image to the size of the non-zero elements of disparity map
# input (img): numpy array of shape (H, W)
# input (d): numpy array of shape (H, W)
# output (img_crop): numpy array of shape (H', W')
def crop_image(img, d):
	rows, cols = np.where(d > 0) # identify the region where the disparity is defined
	row_min, row_max = np.min(rows), np.max(rows)
	col_min, col_max = np.min(cols), np.max(cols)
	img_crop = img[row_min:row_max+1, col_min:col_max+1]
	return img_crop

# shift all pixels of i1 by the value of the disparity map
# input (i_1): numpy array of shape (H, W)
# input (d): numpy array of shape (H, W)
# output (i_d): numpy array of shape (H, W)
def shift_disparity(i_1,d):
	height, width = i_1.shape
	x, y = np.meshgrid(np.arange(width), np.arange(height)) # create a grid of pixel indices 
	# subtract the disparity map from the x indices since we are dealing with horizontal shifts between the left and right images
	x = x - d.astype(int) 
	# ensure we won't try to access pixels that are outside the boundaries of the image
	x = np.clip(x, 0, width - 1)
	y = np.clip(y, 0, height - 1)

	i_d = i_1[y, x]

	return i_d

# compute the negative log of the Gaussian likelihood
# input (i_0): numpy array of shape (H, W)
# input (i_1_d): numpy array of shape (H, W)
# input (mu): float
# input (sigma): float
# output (nll): numpy scalar of shape ()
def gaussian_nllh(i_0, i_1_d, mu, sigma):
	# difference between images
	diff = i_0 - i_1_d 
	# gaussian probability density function
	nll = 0.5 * np.sum(((diff - mu) / sigma) ** 2) + np.prod(diff.shape) * np.log(sigma * np.sqrt(2 * np.pi))
	return nll

# compute the negative log of the Laplacian likelihood
# input (i_0): numpy array of shape (H, W)
# input (i_1_d): numpy array of shape (H, W)
# input (mu): float
# input (s): float
# output (nll): numpy scalar of shape ()
def laplacian_nllh(i_0, i_1_d, mu,s):
	# difference between images
	diff = np.abs(i_0 - i_1_d - mu)
	# laplacian probability density function
	nll = np.sum(np.log(2 * s) + diff / s)
	return nll

# replace p% of the image pixels with values from a normal distribution
# input (img): numpy array of shape (H, W)
# input (p): float
# output (img_noise): numpy array of shape (H, W)
def make_noise(img, p):
	img_noise = img.copy()
	num_pixels = int(p * img.size) # number of pixels to modify
	num_pixels = min(num_pixels, img.size) # ensure they do not exceed img.size
	indices = np.random.choice(np.arange(img.size), replace=False, size=num_pixels) # choose pixel indices randomly
	img_noise.flat[indices] = np.random.normal(0.45, 0.14, size=num_pixels) # replace the chosen ones with values from a normal distribution
	return img_noise

# apply noise to i1_sh and return the values of the negative lok-likelihood for both likelihood models with mu, sigma, and s
# input (i0): numpy array of shape (H, W)
# input (i1_sh): numpy array of shape (H, W)
# input (noise): float
# input (mu): float
# input (sigma): float
# input (s): float
# output (gnllh) - gaussian negative log-likelihood: numpy scalar of shape ()
# output (lnllh) - laplacian negative log-likelihood: numpy scalar of shape ()
def get_nllh_for_corrupted(i_0, i_1_d, noise, mu, sigma, s):
    i_1_d_noise = make_noise(i_1_d, noise)

    # gaussian and laplacian negative log-likelihoods
    gnllh = gaussian_nllh(i_0, i_1_d_noise, mu, sigma)
    lnllh = laplacian_nllh(i_0, i_1_d_noise, mu, s)
	
    return gnllh, lnllh

# DO NOT CHANGE
def main():
	# load images
	i0, i1, gt = load_data('./data/i0.png', './data/i1.png', './data/gt.png')
	i0, i1 = rgb2gray(i0), rgb2gray(i1)

	# shift i1
	i1_sh = shift_disparity(i1, gt)

	# crop images
	i0 = crop_image(i0, gt)
	i1_sh = crop_image(i1_sh, gt)

	mu = 0.0
	sigma = 1.4
	s = 1.4
	for noise in [0.0, 14.0, 27.0]:

		gnllh, lnllh = get_nllh_for_corrupted(i0, i1_sh, noise, mu, sigma, s)

if __name__ == "__main__":
	main()
