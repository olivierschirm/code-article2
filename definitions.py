# ---------------------------------------------------------------------------------
# File: definitions.py
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

import os
import shutil
import rasterio

channels_matrix = [
    'binary',
    'heatmap',
    'distance',
    'speed',
    'acceleration',
    'altitude',
    'slope',
    'bearing1',
    'bearing2',
    'bearing3',
    'bearing4',
    'bearing5',
    'bearing6',
    'bearing7',
    'bearing8',
    'bearing_deviation',
    'bearing_difference',
]

EVALUATION_RESOLUTION = 2.7

def convert_meters_to_degrees(meters):
    """
    Converts a distance in meters to degrees.
    
    Parameters:
        meters (float): Distance in meters.
    
    Returns:
        float: Distance in degrees.
    """
    meters_per_degree = 111139  # approximate value
    degrees = meters / meters_per_degree
    return degrees

def reset_dir(path):
    """
    Resets the specified destination directory by removing its contents, if it exists, and creates a new empty directory.

    Parameters:
        path (str): The path to the directory.

    Returns:
        path (str): The path to the directory.
    
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

    return path

def loadRaster(dataset, resolution):
    """
    Load a raster image file using the rasterio library.
    
    Parameters:
    dataset (str): The name of the dataset (corresponding to the image filename without extension).
    resolution (float): The resolution of the image. Used to locate the image in the directory structure.

    Returns:
    raster (rasterio.io.DatasetReader): An open raster file.
    """
    # Join the directory path, convert decimal point in resolution to comma (for compatibility with directory names), and add the filename
    path = os.path.join('dataset', str(resolution).replace('.', ','), f'{dataset}.tif')
    
    # Open and return the raster file
    raster = rasterio.open(path)
    
    return raster

def convertResultToRaster(result, origin_raster, output_path='result.tif'):
    """
    Convert the given 2D numpy array (result) into a raster file, 
    using the transformation and coordinate reference system (CRS) 
    of an original raster file. The new raster file is saved to the specified path.

    Parameters:
    result (numpy.ndarray): 2D numpy array containing the result to be converted to a raster file.
    origin_raster (rasterio.io.DatasetReader): An open raster file whose properties 
                                               (like CRS and transformation) will be used for the new raster file.
    output_path (str, optional): Path where the new raster file will be saved. 
                                 Default is 'result.tif' in the current working directory.

    Returns:
    dst (rasterio.io.DatasetWriter): The newly created and written raster file.
    """
    # Get the dimensions of the result
    height, width = result.shape

    # Get the transformation and CRS from the original raster
    transform = origin_raster.transform
    crs = origin_raster.crs

    # Get the data type of the result
    dtype = result.dtype

    # Create and write to the new raster file
    with rasterio.open(output_path, 'w', driver='GTiff', height=height, width=width, count=1,
                        dtype=dtype, transform=transform, crs=crs) as dst:
        dst.write(result, 1)
        dst.close()

    return rasterio.open(output_path)

