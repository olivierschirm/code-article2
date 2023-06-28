# ---------------------------------------------------------------------------------
# File: evaluator.py
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

import json
import numpy as np
from shapely.geometry import Point
from geopandas import GeoDataFrame, read_file
from geopy.distance import great_circle
import matplotlib.pyplot as plt
import rasterio
from rasterio.features import rasterize

from definitions import convert_meters_to_degrees, EVALUATION_RESOLUTION

def getIntersections(path):
    """
    This function reads a GeoJSON file and extracts the coordinates of points marked as "Point" in the GeoJSON.
    These points are typically the intersections of various segments within the GeoJSON.

    Args:
        path (str): The path to the GeoJSON file.

    Returns:
        list: A list of tuples representing the intersection points.
             Each tuple contains two float numbers (x, y) representing the coordinates of a point.
    """

    # Create an empty list to store the intersections
    intersections = []
    
    # Open the GeoJSON file
    with open(path) as resultGeojson:
        # Load the GeoJSON file as a Python dictionary
        data = json.load(resultGeojson)
        # Loop through each feature in the GeoJSON
        for jsonSegment in data['features']:
            # If the type of the feature is a Point, it is an intersection
            if jsonSegment['geometry']['type'] == "Point":
                # Append the coordinates of the intersection to the list
                intersections.append((jsonSegment['geometry']['coordinates'][0], jsonSegment['geometry']['coordinates'][1]))
    
    # Return the list of intersections
    return intersections

def mapIntersections(intersections_r, intersections_t, tmatch=50):
    """
    This function maps intersections from one set (referred to as 'r') to another set (referred to as 't'). 

    Args:
        intersections_r (list): A list of intersection coordinates from the 'r' set.
        intersections_t (list): A list of intersection coordinates from the 't' set.
        tmatch (int, optional): A threshold value to match the intersections. Defaults to 50.

    Returns:
        list: A list of dictionaries containing the mapping results.
    """
    
    # Initialize an empty dictionary to store the mapping
    mapping = {}
    
    # Iterate through each intersection in the 'r' set
    for intersection_r in intersections_r:
        # Try to find a mapping for the current intersection in the 't' set
        result = find_mapping(intersection_r, intersections_t, tmatch)
        
        # If a mapping is found
        if result is not None:
            intersectionTruth, intersection_r, distance = result
            key = tuple(intersectionTruth)
            
            # If the key does not exist in the mapping dictionary or the current distance is less than the stored distance
            if key not in mapping or mapping[key][1] >= distance:
                # Store the current intersection and distance in the mapping dictionary
                mapping[key] = (intersection_r, distance)
                
    # Initialize an empty list to store the final results
    result = []
    
    # Iterate through each item in the mapping dictionary
    for key, value in mapping.items():
        # Initialize an empty dictionary to store the current result
        res = {}
        
        # Store the 'r' intersection, 't' intersection, and distance in the result dictionary
        res['intersection_r'] = value[0]
        res['intersection_t'] = key
        res['distance'] = value[1]
        
        # Append the result dictionary to the result list
        result.append(res)
    
    # Return the result list
    return result

def compute_distance(intersection_r, intersection_t):
    """
    This function computes the geographical distance (in meters) between two intersections using the haversine formula. 
    It's implemented by the 'great_circle' method of the geopy.distance module.
    
    Args:
        intersection_r (tuple): A tuple containing the latitude and longitude coordinates of the 'r' intersection.
        intersection_t (tuple): A tuple containing the latitude and longitude coordinates of the 't' intersection.

    Returns:
        float: The geographical distance (in meters) between the two intersections.
    """

    # Compute and return the geographical distance (in meters) between the two intersections
    return great_circle(intersection_r, intersection_t).meters


def find_mapping(intersection_r, intersections_t, tmatch):
    """
    This function finds the mapping between intersection_r and the intersections in intersections_t. 
    It returns the mapped intersection from intersections_t, intersection_r, and their distance if the minimum distance is less than tmatch.
    
    Args:
        intersection_r (tuple): A tuple containing the latitude and longitude coordinates of the 'r' intersection.
        intersections_t (list): A list of tuples, each containing the latitude and longitude coordinates of 't' intersections.
        tmatch (float): The threshold distance for matching 'r' and 't' intersections.
    
    Returns:
        tuple: A tuple containing the 't' intersection matched with intersection_r, intersection_r, and their distance.
        None: If no 't' intersection is found within the tmatch distance from intersection_r.
    """

    # Compute distances between intersection_r and all intersections in intersections_t
    distances = [compute_distance(intersection_r, intersection_t) for intersection_t in intersections_t]

    # Find the minimum distance and the corresponding 't' intersection
    distance = min(distances)
    intersectionTruth = intersections_t[distances.index(distance)]

    # Check if the minimum distance is less than tmatch
    if distance < tmatch:
        # If yes, return the matched 't' intersection, intersection_r, and their distance
        return (intersectionTruth, intersection_r, distance)

    # If no 't' intersection is found within the tmatch distance from intersection_r, return None
    return None

###FScore computation

def Stat(r, t, mapping) :
    """
    This function calculates statistics such as True Positive (TP), False Positive (FP), True Negative (TN), 
    and False Negative (FN) based on the mapping between 'r' and 't' intersections.
    
    Args:
        r (list): A list of 'r' intersections.
        t (list): A list of 't' intersections.
        mapping (list): A list of dictionaries, each representing a mapping between 'r' and 't' intersections.
    
    Returns:
        tuple: A tuple containing TP, FP, TN, and FN.
    """
    TP = len(mapping)
    FP = len(r) - TP
    TN = TP
    FN = len(t) - TP

    return TP, FP, TN, FN

def FScore(TP, FP, FN) :     
    """
    This function calculates the precision, recall, and F-score using True Positive (TP), False Positive (FP), and False Negative (FN).
    
    Args:
        TP (int): The number of True Positives.
        FP (int): The number of False Positives.
        FN (int): The number of False Negatives.
    
    Returns:
        tuple: A tuple containing precision, recall, and F-score, each rounded to two decimal places and divided by 100.
    """
    precision = np.around(TP / (TP + FP) * 100, 2)
    recall = np.around(TP / (TP + FN)  * 100, 2)
    fscore = np.around((2 * recall * precision) / (recall + precision), 2)

    return  precision / 100, recall / 100, fscore / 100

### Conversions
def create_segment_raster(geojson, raster_resolution):
    """
    Create a raster representation of a set of line segments provided in GeoJSON format. 

    Args:
        geojson (str): Path to a GeoJSON file containing line features to be rasterized.
        raster_resolution (float): The resolution of the raster, in the units of the GeoJSON's CRS.

    Returns:
        rasterio.io.DatasetReader: An in-memory rasterio dataset representing the rasterized line features.
    """
    # Convert the GeoJSON file to a GeoDataFrame
    lines_gdf = read_file(geojson)

    # Make sure that the GeoDataFrame is in WGS84 (EPSG 4326) CRS. If not, convert it.
    if lines_gdf.crs != 'epsg: 4326':
        lines_gdf = lines_gdf.to_crs('epsg:4326')

    # Get the bounding box of the GeoDataFrame
    minx, miny, maxx, maxy = lines_gdf.geometry.total_bounds

    # Compute the dimensions of the raster based on the provided resolution
    raster_width = int((maxx - minx) / raster_resolution)
    raster_height = int((maxy - miny) / raster_resolution)
    raster_size = (raster_height, raster_width)

    # Create an empty raster (all zeros) with the computed dimensions
    empty_raster = np.zeros(raster_size, dtype=np.uint8)

    # Create a transformation that maps raster pixels to geographic coordinates
    transform = rasterio.transform.from_origin(minx, maxy, raster_resolution, raster_resolution)

    # Rasterize the line features from the GeoDataFrame, setting cells that intersect lines to 1
    segment_raster = rasterize(lines_gdf.geometry, out=empty_raster, transform=transform, fill=0, default_value=1)

    # Write the raster to an in-memory file and return the dataset
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(driver='GTiff', height=raster_height, width=raster_width, count=1, 
                          dtype=str(segment_raster.dtype), crs='+proj=latlong', transform=transform) as dataset:
            dataset.write(segment_raster, 1)
        dataset = memfile.open()
    
    return dataset

def extend_to_common_extent(raster1, raster2):
    """
    Extend the extent of two raster datasets to a common extent.
    
    Parameters:
    raster1, raster2: rasterio.io.DatasetReader
        Two raster datasets.

    Returns:
    dataset1, dataset2: rasterio.io.DatasetReader
        Two raster datasets with the same geographic extent.
    """
    
    # Get the geographic bounds of both input raster datasets.
    minx1, miny1, maxx1, maxy1 = raster1.bounds
    minx2, miny2, maxx2, maxy2 = raster2.bounds

    # Determine the minimum and maximum coordinates for the combined extent of both raster datasets.
    minx = min(minx1, minx2)
    miny = min(miny1, miny2)
    maxx = max(maxx1, maxx2)
    maxy = max(maxy1, maxy2)

    # Calculate the width and height (in pixels) for the new raster datasets.
    width = int((maxx - minx) / raster1.res[0])
    height = int((maxy - miny) / raster1.res[1])

    # Define the geotransformation for the new raster datasets.
    merged_transform = rasterio.transform.from_origin(minx, maxy, raster1.res[0], raster1.res[1])

    # Create new raster datasets with the shared geographic extent.
    with rasterio.io.MemoryFile() as memfile1, rasterio.io.MemoryFile() as memfile2:
        with memfile1.open(driver='GTiff', height=height, width=width, count=1, 
                          dtype=str(raster1.dtypes[0]), crs=raster1.crs, transform=merged_transform) as dataset1:
            # Reproject the first raster dataset onto the new geographic extent.
            rasterio.warp.reproject(
                source=rasterio.band(raster1, 1),
                destination=rasterio.band(dataset1, 1),
                src_transform=raster1.transform,
                src_crs=raster1.crs,
                dst_transform=merged_transform,
                dst_crs=raster1.crs,
                resampling=rasterio.warp.Resampling.nearest)
        with memfile2.open(driver='GTiff', height=height, width=width, count=1, 
                          dtype=str(raster2.dtypes[0]), crs=raster2.crs, transform=merged_transform) as dataset2:
            # Reproject the second raster dataset onto the new geographic extent.
            rasterio.warp.reproject(
                source=rasterio.band(raster2, 1),
                destination=rasterio.band(dataset2, 1),
                src_transform=raster2.transform,
                src_crs=raster2.crs,
                dst_transform=merged_transform,
                dst_crs=raster2.crs,
                resampling=rasterio.warp.Resampling.nearest)
        dataset1 = memfile1.open()
        dataset2 = memfile2.open()

    # Return the new raster datasets with the same geographic extent.
    return dataset1, dataset2

def compute_matched_unmatched(pred_data, truth_data):
    """
    Compute masks for matched and unmatched "marbles" (pixels with value 1) in predicted and ground truth data.

    Parameters:
    pred_data: numpy.array
        2D array representing predicted data.
    truth_data: numpy.array
        2D array representing ground truth data.

    Returns:
    matched_marbles: numpy.array
        Boolean mask where True represents matched marbles.
    matched_holes: numpy.array
        Boolean mask where True represents matched holes.
    unmatched_marbles: numpy.array
        Boolean mask where True represents unmatched marbles.
    unmatched_holes: numpy.array
        Boolean mask where True represents unmatched holes.
    """

    # Initialize empty masks for matched marbles and holes
    matched_marbles = np.zeros_like(truth_data, dtype=bool)
    matched_holes = np.zeros_like(truth_data, dtype=bool)

    # Iterate over the interior pixels of the rasters
    # Edge pixels are not considered as they don't have a full 3x3 neighborhood
    for i in range(1, truth_data.shape[0] - 1):
        for j in range(1, truth_data.shape[1] - 1):

            # If a predicted marble has a true marble within its 3x3 neighborhood, it's a match
            if pred_data[i, j] == 1 and np.any(truth_data[i-1:i+2, j-1:j+2] == 1):
                matched_marbles[i, j] = True

            # If a true hole (position of a marble in ground truth) has a predicted marble within its 3x3 neighborhood, it's a match
            if truth_data[i, j] == 1 and np.any(pred_data[i-1:i+2, j-1:j+2] == 1):
                matched_holes[i, j] = True

    # The unmatched marbles are the predicted marbles that weren't matched
    unmatched_marbles = (pred_data == 1) & ~matched_marbles

    # The unmatched holes are the true holes that weren't matched
    unmatched_holes = (truth_data == 1) & ~matched_holes

    return matched_marbles, matched_holes, unmatched_marbles, unmatched_holes

def computeFScore(matched_marbles, matched_holes, unmatched_marbles, unmatched_holes):
    """
    Compute the precision, recall, and F-score based on matched and unmatched "marbles" and "holes".

    Parameters:
    matched_marbles: numpy.array
        Boolean mask where True represents matched marbles.
    matched_holes: numpy.array
        Boolean mask where True represents matched holes.
    unmatched_marbles: numpy.array
        Boolean mask where True represents unmatched marbles.
    unmatched_holes: numpy.array
        Boolean mask where True represents unmatched holes.

    Returns:
    precision: float
        The precision of the prediction.
    recall: float
        The recall of the prediction.
    fscore: float
        The F-score of the prediction.
    """
    
    # Precision is the ratio of correctly predicted positive observations (matched marbles) 
    # to the total predicted positive observations (matched and unmatched marbles)
    precision = np.count_nonzero(matched_marbles) / (np.count_nonzero(matched_marbles) + np.count_nonzero(unmatched_marbles))
    
    # Recall (Sensitivity) - the ratio of correctly predicted positive observations (matched holes) 
    # to the all observations in actual class (matched and unmatched holes)
    recall = np.count_nonzero(matched_holes) / (np.count_nonzero(matched_holes) + np.count_nonzero(unmatched_holes))
    
    # F-score is the harmonic mean of Precision and Recall
    fscore = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, fscore

def create_intersection_raster(intersections, truth_raster):
    """
    Create a raster of intersections based on a ground truth raster.

    Parameters:
    intersections: list of tuples
        Each tuple contains the (longitude, latitude) coordinates of an intersection.
    truth_raster: rasterio.io.DatasetReader
        The ground truth raster dataset.

    Returns:
    intersection_raster: numpy.ndarray
        A raster that maps the intersections.
    """
    
    # Convert intersections to a GeoDataFrame.
    # Each intersection point is represented by a shapely Point object.
    intersections_gdf = GeoDataFrame(geometry=[Point(xy) for xy in intersections])

    # Assign a geographic coordinate system to the GeoDataFrame
    # and then transform it to match the coordinate system of the truth raster.
    intersections_gdf.crs = 'epsg: 4326'
    intersections_gdf = intersections_gdf.to_crs('epsg:4326')

    # Create an empty raster with the same shape as the truth raster. 
    # The raster is initialized to zero (no intersections).
    empty_raster = np.zeros(truth_raster.shape, dtype=rasterio.uint8)

    # Rasterize the intersections, setting intersections to 1 in the empty raster.
    intersection_raster = rasterize(intersections_gdf.geometry, out=empty_raster, transform=truth_raster.transform, fill=0, default_value=1)

    # Create a boolean mask from the truth raster.
    # This mask is True for non-zero pixels (i.e., where there is data in the truth raster).
    truth_mask = truth_raster.read(1) > 0

    # Apply the truth mask to the intersection raster.
    # This step sets non-intersecting areas (areas where there is no data in the truth raster) to 0.
    intersection_raster = intersection_raster * truth_mask

    return intersection_raster

def compute_matched_unmatched_circle(pred_data, truth_data, intersections, radius):
    """
    Compare predicted data (marbles) to ground truth data within a certain radius and categorize pixels as matched/unmatched.

    Parameters:
    pred_data: numpy.ndarray
        The predicted raster data.
    truth_data: numpy.ndarray
        The ground truth raster data.
    intersections: numpy.ndarray
        Raster that maps the intersections.
    radius: int
        The radius around each intersection point to consider when comparing predicted data to ground truth data.

    Returns:
    matched_marbles: numpy.ndarray
        A boolean mask of matched predicted data.
    matched_holes: numpy.ndarray
        A boolean mask of matched ground truth data.
    unmatched_marbles: numpy.ndarray
        A boolean mask of unmatched predicted data.
    unmatched_holes: numpy.ndarray
        A boolean mask of unmatched ground truth data.
    """

    # Create boolean masks for matched pixels, initializing all pixels to False
    matched_marbles = np.zeros_like(truth_data, dtype=bool)
    matched_holes = np.zeros_like(truth_data, dtype=bool)

    # Create boolean masks for unmatched pixels, initializing all pixels to False
    unmatched_marbles = np.full_like(truth_data, False, dtype=bool)
    unmatched_holes = np.full_like(truth_data, False, dtype=bool)

    # Iterate over each pixel in the intersection raster
    for row in range(intersections.shape[0]):
        for col in range(intersections.shape[1]):
            # If this pixel is an intersection
            if intersections[row, col] > 0:
                # Define the bounds of the circular radius around the intersection point
                row_start = max(0, row-radius)
                row_end = min(truth_data.shape[0], row+radius)
                col_start = max(0, col-radius)
                col_end = min(truth_data.shape[1], col+radius)

                # Iterate over each pixel within the defined radius
                for i in range(row_start, row_end):
                    for j in range(col_start, col_end):
                        # If a predicted pixel matches a ground truth pixel, mark it as matched
                        if pred_data[i, j] == 1 and np.any(truth_data[i-1:i+2, j-1:j+2] == 1):
                            matched_marbles[i, j] = True
                        if truth_data[i, j] == 1 and np.any(pred_data[i-1:i+2, j-1:j+2] == 1):
                            matched_holes[i, j] = True

                        # If a predicted or ground truth pixel does not have a match, mark it as unmatched
                        if pred_data[i, j] == 1 and not matched_marbles[i, j]:
                            unmatched_marbles[i, j] = True
                        if truth_data[i, j] == 1 and not matched_holes[i, j]:
                            unmatched_holes[i, j] = True

    return matched_marbles, matched_holes, unmatched_marbles, unmatched_holes

def visualize_results(matched_marbles, matched_holes, unmatched_marbles, unmatched_holes, truth_data):
    """
    Visualize the matched and unmatched pixels from the predicted and truth data on a single image. 

    Each category of pixels is given a unique color:
    - Matched ground truth data (holes) are shown in blue.
    - Matched predicted data (marbles) are shown in green.
    - Unmatched predicted data (marbles) are shown in red.
    - Unmatched ground truth data (holes) are shown in violet.

    Parameters:
    matched_marbles: numpy.ndarray
        A boolean mask of matched predicted data.
    matched_holes: numpy.ndarray
        A boolean mask of matched ground truth data.
    unmatched_marbles: numpy.ndarray
        A boolean mask of unmatched predicted data.
    unmatched_holes: numpy.ndarray
        A boolean mask of unmatched ground truth data.
    truth_data: numpy.ndarray
        The ground truth raster data.

    Returns:
    None
    """
    
    # Create a single raster with different values for each category of pixels
    visualization = np.zeros_like(truth_data, dtype=np.uint8)
    visualization[matched_holes] = 1     # Matched ground truth data are marked as 1
    visualization[matched_marbles] = 2   # Matched predicted data are marked as 2
    visualization[unmatched_marbles] = 3 # Unmatched predicted data are marked as 3
    visualization[unmatched_holes] = 4   # Unmatched ground truth data are marked as 4

    # Define a custom color map: matched holes are blue, matched marbles are green, unmatched marbles are red, and unmatched holes are violet
    cmap = plt.cm.colors.ListedColormap(['black', 'blue', 'green', 'red', 'violet'])

    # Display the visualization raster using the custom color map
    plt.imshow(visualization, cmap=cmap)

    # Display the color bar
    plt.colorbar()

    # Show the plot
    plt.show()

def main_eval(prediction_geojson_path, truth_geojson_path):
    """
    Perform evaluation of predicted and ground truth data.

    This function computes various evaluation metrics to assess the accuracy of the predicted data compared to the ground truth data.
    It performs evaluation on both the segment level and the intersection level.

    Parameters:
    prediction_geojson_path: str
        The file path to the GeoJSON file containing the predicted data.
    truth_geojson_path: str
        The file path to the GeoJSON file containing the ground truth data.

    Returns:
    fint: tuple
        A tuple containing the precision, recall, and F-score for the intersection level evaluation.
    ftop: tuple
        A tuple containing the precision, recall, and F-score for the segment level evaluation without considering the intersection information.
    fito: tuple
        A tuple containing the precision, recall, and F-score for the segment level evaluation considering the intersection information.
    """

    intersections_r = getIntersections(prediction_geojson_path)
    intersections_t = getIntersections(truth_geojson_path)
    prediction_raster = create_segment_raster(prediction_geojson_path, convert_meters_to_degrees(EVALUATION_RESOLUTION))
    truth_raster = create_segment_raster(truth_geojson_path, convert_meters_to_degrees(EVALUATION_RESOLUTION))
    prediction_raster, truth_raster = extend_to_common_extent(prediction_raster, truth_raster)
    intersections_raster = create_intersection_raster(intersections_t, truth_raster)
    prediction_raster, truth_raster = prediction_raster.read(1), truth_raster.read(1)

    #FINT
    mapping = mapIntersections(intersections_r, intersections_t)
    TP, FP, TN, FN = Stat(intersections_r, intersections_t, mapping)
    FINT = FScore(TP, FP, FN)

    #FTOP
    matched_marbles, matched_holes, unmatched_marbles, unmatched_holes = compute_matched_unmatched(prediction_raster, truth_raster)
    FTOP = computeFScore(matched_marbles, matched_holes, unmatched_marbles, unmatched_holes)
    #visualize_results(matched_marbles, matched_holes, unmatched_marbles, unmatched_holes, truth_raster)

    #FITO
    matched_marbles, matched_holes, unmatched_marbles, unmatched_holes = compute_matched_unmatched_circle(prediction_raster, truth_raster, intersections_raster, radius=10)
    FITO = computeFScore(matched_marbles, matched_holes, unmatched_marbles, unmatched_holes)
    #visualize_results(matched_marbles, matched_holes, unmatched_marbles, unmatched_holes, truth_raster)

    return FINT, FTOP, FITO