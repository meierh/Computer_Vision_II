from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from matplotlib import pyplot as plt
from skim import color

from utils import VOC_LABEL2COLOR
from utils import VOC_STATISTICS
from utils import numpy2torch
from utils import torch2numpy

from torchvision import models

class VOC2007Dataset(Dataset):
    """
    Class to create a dataset for VOC 2007
    Refer to https://pytorch.org/tutorials/beginner/basics/data_tutorial.html for an instruction on PyTorch datasets.
    """

    def __init__(self, root, train, num_examples):
        super().__init__()
        """
        Initialize the dataset by setting the required attributes.

        Args:
            root: root folder of the dataset (string)
            train: indicates if we want to load training (train=True) or validation data (train=False)
            num_examples: size of the dataset (int)

        Returns (just set the required attributes):
            input_filenames: list of paths to individual ims
            target_filenames: list of paths to individual segmentations (corresponding to input_filenames)
            rgb2label: lookup table that maps RGB values to class labels using the constants in VOC_LABEL2COLOR.
        """

        self.input_filenames = []
        self.target_filenames = []

        if train:
            filename = os.path.join(root, 'ImageSets/Segmentation/train.txt')
        else:
            filename = os.path.join(root, 'ImageSets/Segmentation/val.txt')

        with open(filename, 'r') as file:
            list = file.readlines()

        list = list[:num_examples] # restricts the size of the dataset 

        for filename in list:
            filename = filename.strip() # removes whitespace characters from the filename string
            self.input_filenames.append(os.path.join(root, 'JPEGImages', filename + '.jpg'))
            self.target_filenames.append(os.path.join(root, 'SegmentationClass', filename + '.png'))

        self.rgb2label = VOC_LABEL2COLOR

    def __getitem__(self, index):
        """
        Return an item from the datset.

        Args:
            index: index of the item (Int)

        Returns:
            item: dictionary of the form {'im': the_im, 'gt': the_label}
            with the_im being a torch tensor (3, H, W) (float) and 
            the_label being a torch tensor (1, H, W) (long) and 
        """
        item = dict()

         # image reading
        img_path = self.input_filenames[index]
        img = plt.imread(img_path).astype(np.float32)
        img = numpy2torch(img)

        # segmentation reading
        segm_path = self.target_filenames[index]
        segm = plt.imread(segm_path)
        segm = (segm[:, :, 0:3] * 255).astype(np.int64)

        # segmentation conversion
        conv = np.full(segm.shape[:2], -1, dtype=np.int64) # ambiguous labels represented by label -1
        for label, rgb in enumerate(self.rgb2label):
            mask = np.all(segm == rgb, axis=2)
            conv[mask] = label
        conv = numpy2torch(conv)

        item['im'] = img
        item['gt'] = conv

        assert (isinstance(item, dict))
        assert ('im' in item.keys())
        assert ('gt' in item.keys())

        return item

    def __len__(self):
        """
        Return the length of the datset.

        Args:

        Returns:
            length: length of the dataset (int)
        """
        return len(self.input_filenames)



def create_loader(dataset, batch_size, shuffle, num_workers=1):
    """
    Return loader object.

    Args:
        dataset: PyTorch Dataset
        batch_size: int
        shuffle: bool
        num_workers: int

    Returns:
        loader: PyTorch DataLoader
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    assert (isinstance(loader, DataLoader))
    return loader


def voc_label2color(np_im, np_label):
    """
    Super-impose labels on a given im using the colors defined in VOC_LABEL2COLOR.

    Args:
        np_im: numpy array (H,W,3) (float)
        np_label: numpy array  (H,W) (int)

    Returns:
        colored: numpy array (H,W,3) (float)
    """
    assert (isinstance(np_im, np.ndarray))
    assert (isinstance(np_label, np.ndarray))

    hsv_im = color.rgb2hsv(np_im) # grayscale to hsv

    for label_id, rgb in enumerate(VOC_LABEL2COLOR):
        mask = (np_label == label_id).reshape((np_label.shape[0], np_label.shape[1]))
        color = color.rgb2hsv(np.array(rgb).reshape((1, 1, 3)))

        # hue channel
        hsv_im[..., 0][mask] = color[0, 0, 0]
        hsv_im[..., 1][mask] = color[0, 0, 1]

    colored = color.hsv2rgb(hsv_im) # hsv to rgb

    assert (np.equal(colored.shape, np_im.shape).all())
    assert (np_im.dtype == colored.dtype)
    return colored


def show_dataset_examples(loader, grid_height, grid_width, title):
    """
    Visualize samples from the dataset.

    Args:
        loader: PyTorch DataLoader
        grid_height: int
        grid_width: int
        title: string
    """
    pass

def normalize_input(input_tensor):
    """
    Normalize a tensor using statistics in VOC_STATISTICS.

    Args:
        input_tensor: torch tensor (B,3,H,W) (float32)
        
    Returns:
        normalized: torch tensor (B,3,H,W) (float32)
    """
    normalized = []

    assert (type(input_tensor) == type(normalized))
    assert (input_tensor.size() == normalized.size())
    return normalized

def run_forward_pass(normalized, model):
    """
    Run forward pass.

    Args:
        normalized: torch tensor (B,3,H,W) (float32)
        model: PyTorch model
        
    Returns:
        prediction: class prediction of the model (B,1,H,W) (int64)
        acts: activations of the model (B,21,H,W) (float 32)
    """
    prediction = []
    acts = []

    assert (isinstance(prediction, torch.Tensor))
    assert (isinstance(acts, torch.Tensor))
    return prediction, acts

def show_inference_examples(loader, model, grid_height, grid_width, title):
    """
    Perform inference and visualize results.

    Args:
        loader: PyTorch DataLoader
        model: PyTorch model
        grid_height: int
        grid_width: int
        title: string
    """
    pass

def average_precision(prediction, gt):
    """
    Compute percentage of correctly labeled pixels.

    Args:
        prediction: torch tensor (B,1,H,W) (int)
        gt: torch tensor (B,1,H,W) (int)
       
    Returns:
        avg_prec: torch scalar (float32)
    """
    return None



# Example usage in main()
# Feel free to experiment with your code in this function
# but make sure your final submission can execute this code
def main():
    # Please set an environment variables 'VOC2007_HOME' pointing to your '../VOCdevkit/VOC2007' folder
    root = os.environ["VOC2007_HOME"]

    # create datasets for training and validation
    train_dataset = VOC2007Dataset(root, train=True, num_examples=128)
    valid_dataset = VOC2007Dataset(root, train=False, num_examples=128)

    # create data loaders for training and validation
    train_loader = create_loader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    valid_loader = create_loader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

    # show some ims for the training and validation set
    show_dataset_examples(train_loader, grid_height=2, grid_width=3, title='training examples')
    show_dataset_examples(valid_loader, grid_height=2, grid_width=3, title='validation examples')

    # Load FCN network
    model = models.segmentation.fcn_resnet101(pretrained=True, num_classes=21)

    # Apply fcn. Switch to training loader if you want more variety.
    show_inference_examples(valid_loader, model, grid_height=2, grid_width=3, title='inference examples')


if __name__ == '__main__':
    main()
