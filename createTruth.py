# ---------------------------------------------------------------------------------
# File: createTruth.py
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
import json
import numpy as np
from math import sqrt
import rasterio
from definitions import convert_meters_to_degrees

datasets = [
    'labaroche',
    'hunawihr',
    #'linthal',
    #'blaesheim',
    #'ribeauville',
    #'haguenau',
]

#resolution are given in meters
resolutions = [
    2.2,
    2.7,
    3.1,
]

# Iterate over each dataset
for dataset in datasets:
    segments = []  # Initialize an empty list to store line segments
    
    # Open the GeoJSON file corresponding to the dataset
    with open(os.path.join('traces truth', dataset + '.geojson')) as tracesFile:
        # Load the GeoJSON data and iterate over its features
        for jsonTrace in json.load(tracesFile)['features']:
            # Extract line segments based on the geometry type
            if jsonTrace['geometry']['type'] == "MultiLineString":
                segments.append(jsonTrace['geometry']['coordinates'][0])
            if jsonTrace['geometry']['type'] == "LineString":
                segments.append(jsonTrace['geometry']['coordinates'])
    
    # Iterate over each line segment
    for segment in segments:
        i = 0
        
        # Continue processing until the current point equals the last point in the segment
        while segment[i] != segment[len(segment) - 1]:
            # Insert intermediate points between consecutive points if the distance between them is greater than a threshold
            while sqrt((segment[i + 1][0] - segment[i][0]) ** 2 + (segment[i + 1][1] - segment[i][1]) ** 2) > 0.000004:
                newPoint = (segment[i][0] + (segment[i + 1][0] - segment[i][0]) / 2, segment[i][1] + (segment[i + 1][1] - segment[i][1]) / 2)
                segment.insert(i + 1, newPoint)
            
            i = i + 1
    
    # Iterate over each resolution
    for resolution_meters in resolutions:
        resolution = convert_meters_to_degrees(resolution_meters)
        
        # Open the raster file corresponding to the dataset and resolution
        raster = rasterio.open(os.path.join('dataset', str(resolution_meters).replace('.', ','), dataset + '.tif'))
        result = np.zeros(raster.shape)  # Create an empty result array
        
        # Iterate over each line segment
        for segment in segments:
            # Iterate over each point in the segment
            for point in segment:
                # Calculate the pixel coordinates in the raster corresponding to the point
                x = int((raster.bounds[2] - point[0]) / raster.res[0])
                y = int((raster.bounds[3] - point[1]) / raster.res[1])
                
                # Update the result array if the pixel coordinates are within the raster's dimensions
                if x < raster.shape[1] and y < raster.shape[0]:
                    result[y][x] = 1
        
        result = np.flip(result, axis=1)  # Flip the result array horizontally
        
        # Create the directory for storing the processed data (if it doesn't exist)
        path = os.path.join('dataset truth', str(resolution_meters).replace('.', ','))
        os.makedirs(path, exist_ok=True)
        
        # Set the file path for the GeoTIFF file
        file_path = os.path.join(path, dataset + '.tif')
        
        # Create a new GeoTIFF file and write the result array to it
        writer = rasterio.open(
            os.path.join(file_path),
            'w',
            driver='GTiff',
            height=raster.shape[0],
            width=raster.shape[1],
            dtype=rasterio.dtypes.float64,
            count=1,
            crs=raster.crs,
            transform=raster.transform
        )
        
        writer.write(result, 1)
       
        print(f"Done {dataset} at resolution {str(resolution_meters)}m")