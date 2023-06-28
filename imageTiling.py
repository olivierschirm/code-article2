# ---------------------------------------------------------------------------------
# File: imageTiling.py
# 
# Author: SCHIRM Olivier
# 
# Copyright (c) 2023 Visorando
# 
# This file is part of the supplementary material for the paper titled:
# "Article 2", published in Journal Name, in 2023.
# 
# This software is provided "as is", without warranty of any kind, express or
# implied, including but not limited to the warranties of merchantability,
# fitness for a particular purpose and noninfringement. In no event shall the
# authors or copyright holders be liable for any claim, damages or other
# liability, whether in an action of contract, tort or otherwise, arising from,
# out of or in connection with the software or the use or other dealings in
# the software.
# ---------------------------------------------------------------------------------

# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 14:08:26 2022

@author: Olivier Schirm
"""

import os
import re
import numpy as np
from PIL import Image
from glob import glob
import shutil
import rasterio

from definitions import channels_matrix, reset_dir

def preprocessTrainData(configuration, output_dir):
    """
    This function takes in a configuration data object and based on its parameters, it processes image data,
    dividing the input raster images into smaller patches, saving them into a directory, and applying data augmentation if required.

    Args:
    configuration (dict): A configuration dictionary that contains the following keys:
                          - 'datasets': List of dataset names to be processed.
                          - 'resolutions': List of resolution levels to be considered.
                          - 'channels': Channels to be included in the output.
                          - 'image_size': The desired output patch size.
                          - 'augment': Boolean flag indicating whether data augmentation should be performed.

    """

    # Reset and get the path to the output directory
    output_directory = reset_dir(output_dir)

    # Loop over each dataset and resolution in the configuration
    for dataset in configuration['datasets']:
        for resolution in configuration['resolutions']:
            resolution = str(resolution).replace('.', ',')
            
            # Create a specific destination path for each dataset and resolution
            specific_destination = os.path.join(output_directory, f'{dataset}_{resolution}')
            
            # Create directory if not exists
            os.makedirs(specific_destination, exist_ok=True)

            # Open the raster image file and stack it into a 3D numpy array
            raster = np.stack(rasterio.open(os.path.join('dataset', resolution, f'{dataset}.tif')).read(), axis=2)
            
            # For each channel in the configuration, create patches and save them
            for channel in configuration['channels']:
                patches = createImagePatches(raster[:,:,channels_matrix.index(channel)], configuration['image_size'])
                savePatches(patches, specific_destination, name=channel)

            # Open the ground truth raster image, create patches and save them
            rastertruth = rasterio.open(os.path.join(os.getcwd(), 'dataset truth', resolution, f'{dataset}.tif')).read(1)
            patches = createImagePatches(rastertruth, configuration['image_size'])
            savePatches(patches, specific_destination, name='truth')

    # Delete all empty 'truth.png' files in the output directory
    deleteEmptyFiles(output_directory, 'truth.png')

    # If the configuration specifies data augmentation, perform it
    if configuration['augment']:
        augmentDatasetWithRotation(output_directory)

def preprocessTestData(raster, confData, pas, output_directory='test_examples', channelToPermute=False):
    """
    This function is used to preprocess the test data.

    Args:
    raster (str): The raster file to preprocess
    confData (dict): The configuration data which contains channels and image size.
    pas (int): The step size for moving the window to create patches.
    output_directory (str): path to directory where examples will be store.
    channelToPermute (bool): Whether to permute a channel. Default is False.

    Returns:
    None
    """
    # Reset the output directory
    output_directory = reset_dir(output_directory)

    # Stack the raster data
    raster = np.stack(raster.read(), axis=2)
    
    # For each channel in the configuration data
    for channel in confData['channels'] :
        
        # Create image patches
        patchs = createImageTestPatches(raster[:,:,channels_matrix.index(channel)], confData['image_size'], int(pas))
        
        # Save the patches to the output directory
        savePatches(patchs, output_directory, name=channel, channelToPermute = channel if channel == channelToPermute else False)

def createImagePatches(dataset_raster, patch_size):
    """
    This function takes a raster dataset and a patch size, then divides the dataset into patches of the given size.
    Patches are created in a grid pattern starting from the top-left. Padding is added where necessary.

    Args:
    dataset_raster (numpy.ndarray): The input 2D raster image to be divided into patches.
    patch_size (int): The size of the square patches to be created.

    Returns:
    numpy.ndarray: A 4D array where the first two dimensions are the patch indices, and the last two dimensions are the patch content.
    """

    # Initialize an empty array to store the patches
    patches = np.empty((dataset_raster.shape[1]//patch_size + 1, dataset_raster.shape[0]//patch_size + 1, patch_size, patch_size))

    # Loop over the raster image creating patches
    for h in range(0, dataset_raster.shape[0]//patch_size):
        for w in range(0, dataset_raster.shape[1]//patch_size):
            patch = dataset_raster[h*patch_size:h*patch_size+patch_size, w*patch_size:w*patch_size+patch_size] 
            patches[w,h,:,:] = patch

    # Handle padding for remaining part in width
    remaining_width = patch_size - dataset_raster.shape[1]%patch_size
    w = dataset_raster.shape[1]//patch_size
    for h in range(0, dataset_raster.shape[0]//patch_size):
        patch = dataset_raster[h*patch_size:h*patch_size+patch_size, w*patch_size:w*patch_size+patch_size]
        patch = np.pad(patch, ((0,0), (0,remaining_width)))
        patches[w,h,:,:] = patch

    # Handle padding for remaining part in height
    remaining_height = patch_size - dataset_raster.shape[0]%patch_size
    h = dataset_raster.shape[0]//patch_size
    for w in range(0, dataset_raster.shape[1]//patch_size):
        patch = dataset_raster[h*patch_size:h*patch_size+patch_size, w*patch_size:w*patch_size+patch_size]
        patch = np.pad(patch, ((0,remaining_height), (0,0)))
        patches[w,h,:,:] = patch

    # Handle padding for the bottom right corner
    h = dataset_raster.shape[0]//patch_size
    w = dataset_raster.shape[1]//patch_size
    patch = dataset_raster[h*patch_size:h*patch_size+patch_size, w*patch_size:w*patch_size+patch_size]
    patch = np.pad(patch, ((0,remaining_height), (0,remaining_width)))
    patches[w,h,:,:] = patch

    return patches

def createImageTestPatches(dataset_raster, size, pas):
    """
    This function creates image patches from the given raster dataset for testing.

    Args:
    dataset_raster (np.array): The raster data of the image in the form of a numpy array.
    size (int): The size of the patches to be created.
    pas (int): The step size for moving the window to create patches.

    Returns:
    np.array: A numpy array containing the image patches.
    """
    # Create an empty numpy array to store the patches
    patchs = np.empty((dataset_raster.shape[1] // pas, dataset_raster.shape[0] // pas, size, size))

    # Iterate over the raster data to create patches
    for w in range(patchs.shape[0]):
        for h in range(patchs.shape[1]):
            # Get a patch from the raster data
            patch = dataset_raster[h*pas:h*pas+size, w*pas:w*pas+size]
            # If the patch size is not as expected, pad it to make it of the desired size
            if patch.shape[0] != size or patch.shape[1] != size : 
                patch = np.pad(patch, ((0, size-patch.shape[0]), (0, size-patch.shape[1])))
            # Store the patch in the patches array
            patchs[w, h, :, :] = patch

    return patchs


def savePatches(patches, destination, name, channelToPermute=False):
    """
    This function takes a 4D array of patches, a destination directory, a name, and a boolean flag for channel permutation.
    It then normalizes and optionally permutes the patches, and saves them as PNG images in the specified directory.

    Args:
    patches (numpy.ndarray): A 4D array where the first two dimensions are the patch indices, and the last two dimensions are the patch content.
    destination (str): The output directory where the patches should be saved.
    name (str): The base name of the output files.
    permute (bool, optional): A flag indicating whether to permute the channels of the patches. Defaults to False.

    """

    # Loop over each patch
    for w in range(patches.shape[0]):
        for h in range(patches.shape[1]):
            
            # Create a new directory for each patch if it does not exist
            path = os.path.join(destination, f'{w}_{h}')
            if not os.path.exists(path): 
                os.mkdir(path) 

            # Normalize and optionally permute the patch
            patch = patches[w,h,:,:]
            patch = normalize(patch, name)
            patch = permuteImageChannel(patch) if channelToPermute == name else patch
            
            # Convert the patch to an image and save it
            patch_img = Image.fromarray(patch * 255).convert('L')
            patch_img.save(os.path.join(path, f'{name}.png'))

def loadPatches(confData, directory="test_examples"):
    """
    This function is used to load the image patches.

    Args:
    directory (str): The directory where the image patches are located.
    confData (dict): The configuration data which contains channels.

    Returns:
    np.array: A numpy array containing all the image patches.
    """
    # Initialize a dictionary to store the patches
    patches_dict = {}

    # Get a list of all the subdirectories in the directory
    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    # Iterate over each subdirectory
    for subdirectory in subdirectories:
        # Extract the width and height indices from the subdirectory name
        w, h = map(int, re.match(r'(\d+)_(\d+)', subdirectory).groups())

        # Initialize a dictionary to store the patches for this subdirectory
        patches_subdict = {}

        # Iterate over each channel
        for channel in confData['channels']:
            # Open the image file
            img = Image.open(os.path.join(directory, subdirectory, f'{channel}.png'))

            # Convert the image into a numpy array
            img_arr = np.array(img)

            # Add the image array to the patches subdictionary
            patches_subdict[channel] = img_arr

        # Convert the patches subdictionary to a multi-channel numpy array and add it to the patches dictionary
        patches_dict[(w, h)] = np.dstack(list(patches_subdict.values()))

    # Calculate the dimensions of the output array
    max_w = max(key[0] for key in patches_dict.keys()) + 1
    max_h = max(key[1] for key in patches_dict.keys()) + 1
    patch_size = list(patches_dict.values())[0].shape

    # Initialize an empty array to store the patches
    patches = np.empty((max_w, max_h, *patch_size))

    # Assign each patch to its position in the output array
    for (w, h), patch in patches_dict.items():
        patches[w, h, :, :, :] = patch

    return patches

def deleteEmptyFiles(directory, filename):
    """
    This function checks all subdirectories of the provided directory for a file with the provided name.
    If the file is found and its content sums to zero (indicating it is an empty image), the containing directory is removed.

    Args:
    directory (str): The path of the directory to check.
    filename (str): The name of the file to check for.

    """

    # Loop over all subdirectories in the specified directory
    for subdir in glob(directory + '//*//*'):
        # Open the file, if it exists, and normalize its content
        image = np.array(Image.open(os.path.join(subdir, filename))) / 255

        # If the sum of the image's content is zero (indicating an empty image), remove the containing directory
        if image.sum() == 0:
            shutil.rmtree(subdir)

def normalize(chunk, name):
    """
    This function normalizes an array (a chunk of an image) based on its type, indicated by the provided name.
    It either scales the data to the range [0, 1] by dividing by the maximum value, or it scales and shifts the data to the range [0, 1] by subtracting the minimum value and dividing by the range.
    If all the values in the chunk are zero, it is returned as is.

    Args:
    chunk (numpy.ndarray): The input 2D array to be normalized.
    name (str): The name indicating the type of data in the array, used to determine the normalization method.

    Returns:
    numpy.ndarray: The normalized 2D array.

    """

    # If the chunk has no positive values, return it as is
    if not chunk[chunk > 0].any():
        return chunk

    # For certain types of data, divide by the maximum value to scale to [0, 1]
    if name in ['heatmap', 'distance', 'speed', 'acceleration', 'bearing_deviation', 'bearing_difference']:
        chunk = chunk / np.max(chunk)

    # For other types of data, subtract the minimum value and divide by the range to scale and shift to [0, 1]
    elif name in ['altitude', 'slope']:
        min_val = np.min(chunk[chunk > 0])  # compute the minimum excluding zeros
        chunk = (chunk - min_val) / (np.max(chunk) - min_val + np.finfo(float).eps)

    return chunk

def augmentDatasetWithRotation(folderPath):
    """
    This function performs data augmentation by rotating each image in the dataset by multiples of 90 degrees, then cropping it back to its original size.
    The augmented images are saved in new directories, following the original directory structure.

    Args:
    folderPath (str): The path of the directory containing the dataset to augment.
    """

    # Get the paths to the example directories
    example_paths = glob(folderPath + '/*/*')

    # Loop over each example directory
    for ep in example_paths:

        # Get the paths to the channel images in the current example directory
        channel_image_paths = glob(ep + '/*')

        # Loop over each channel image
        for cip in channel_image_paths:
            # Load the current channel image
            ci = Image.open(cip)

            # Save the original image size
            original_size = (np.array(ci).shape[0], np.array(ci).shape[0])  # Original input size

            # Generate rotated versions of the image
            for i in range(1, 4):  # Create 3 rotations
                # Calculate the rotation angle (90 degrees multiplied by the rotation index)
                rotation_angle = i * 90 

                # Define the output directory and ensure it exists
                dirpath = os.path.dirname(cip) + '_' + str(rotation_angle)
                filepath = os.path.basename(cip)
                if channel_image_paths.index(cip) == 0: os.mkdir(dirpath)

                # Rotate the image
                ci_rotated = ci.rotate(-rotation_angle, resample=Image.BICUBIC, expand=True)

                # Crop the image back to the original size
                width, height = ci_rotated.size
                left = (width - original_size[0]) / 2
                top = (height - original_size[1]) / 2
                right = (width + original_size[0]) / 2
                bottom = (height + original_size[1]) / 2
                ci_rotated_cropped = ci_rotated.crop((left, top, right, bottom))

                # Save the cropped, rotated image
                ci_rotated_cropped.save(os.path.join(dirpath, filepath))

def permuteImageChannel(image):
    """
    This function permutes the values of an input image randomly and returns the permuted image.

    Args:
    image (numpy.ndarray): The input 2D array representing the image.

    Returns:
    numpy.ndarray: The permuted image.
    """

    # Create a copy of the input image
    permuted_image = image.copy()

    # Randomly permute the pixels of the image
    permuted_image[:, :] = np.random.permutation(permuted_image[:, :].ravel()).reshape(image.shape[0], image.shape[1])
    
    return permuted_image
