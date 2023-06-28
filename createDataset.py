# ---------------------------------------------------------------------------------
# File: createDataset.py
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
import gpxpy
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from math import ceil, floor, isnan
from PIL import Image
import rasterio

from definitions import convert_meters_to_degrees

dslist = {
    'labaroche' : 68,
    'hunawihr' : 68,
    'ribeauville' : 68,
    #'linthal' : 68,
    #'blaesheim' : 67,
    #'haguenau' : 67,
}

#resolution are given in meters
resolutions = [
    2.2,
    2.7,
    3.1,
]

tracesDirectory = 'traces'
datasetDirectory = 'dataset'

for dataset, mnt in dslist.items() :
    xmin, xmax, ymin, ymax = 100000000, -1, 10000000, -1 
    directory = os.path.join(tracesDirectory, dataset)
    
    segments = []
    starting_points = []
    for filename in os.listdir(directory):
        gpx = gpxpy.parse(open(os.path.join(directory, filename), 'r' , encoding="utf8"))

        for track in gpx.tracks:
            for segment in track.segments:
                
                if (segment.points[0].latitude, segment.points[0].longitude) in starting_points :
                    continue
                else :
                    starting_points.append((segment.points[0].latitude, segment.points[0].longitude))

                record = []
                previous_point = None
                for point in segment.points:
                    if previous_point is not None:
                        # Compute distance between current point and previous point
                        distance = geodesic((previous_point.latitude, previous_point.longitude), (point.latitude, point.longitude)).meters
                        
                        if point.time is not None and previous_point.time is not None:
                            # Compute time difference in seconds
                            time_diff = (point.time - previous_point.time).total_seconds()
                            # Compute speed as distance/time in m/s, and convert to km/h
                            speed = (distance / time_diff) * 3.6 if time_diff > 0 else np.nan
                        else:
                            speed = np.nan
                    else:
                        distance = np.nan
                        speed = np.nan

                    record.append({
                        'latitude': point.latitude,
                        'longitude': point.longitude,
                        'time': point.time,
                        'distance': distance,
                        'speed': speed
                    })

                    # Save current point for the next iteration
                    previous_point = point

                df = pd.DataFrame(record)
                df = df.fillna(np.nan)
                
                if df.longitude.describe()['min'] < xmin : xmin = df.longitude.describe()['min']
                if df.longitude.describe()['max'] > xmax : xmax = df.longitude.describe()['max']
                if df.latitude.describe()['min'] < ymin : ymin = df.latitude.describe()['min']
                if df.latitude.describe()['max'] > ymax : ymax = df.latitude.describe()['max']

                ##Pretreatment
                # Remove consecutive duplicates based on 'latitude' and 'longitude'
                df = df.loc[(df[['latitude', 'longitude']].shift() != df[['latitude', 'longitude']]).any(axis=1)]

                # If 'distance' is greater than 100 (meters), set it to 100
                df['distance'] = df['distance'].apply(lambda x: np.nan if x > 100 else x)

                # If 'speed' is greater than 10 (km/h), set it to 10
                df['speed'] = df['speed'].apply(lambda x: np.nan if x > 10 else x)

                # If 'distance' is 0, set 'speed' to 0
                df.loc[df['distance'] == 0, 'speed'] = np.nan
                ##End Pretreatment

                
                # Calculate acceleration (in km/h^2) only if 'time' is not null
                if df['time'].notna().all():
                    df['acceleration'] = df['speed'].diff() / (df['time'].diff().dt.total_seconds() / 3600)
                else:
                    df['acceleration'] = np.nan


                df['bearing'] = np.degrees(np.arctan2(df.latitude.diff(periods=-1) * -1, df.longitude.diff(periods=-1) * -1))
                
                df['bearing_tmp'] = df['bearing'].apply(lambda x: x+360 if x < 0 else x)
                df['bearing_shifted'] = df['bearing_tmp'].shift(1)
                df['bearing_difference'] = df.apply(lambda row: min(abs(row['bearing_tmp'] - row['bearing_shifted']), 360 - abs(row['bearing_tmp'] - row['bearing_shifted'])), axis=1)
                df = df.drop(columns=['bearing_tmp'])
       
                segments.append(df)
                
    
    for resolution_meters in resolutions : 
        resolution = convert_meters_to_degrees(resolution_meters)

        largeur = ceil((xmax - xmin) / resolution)
        hauteur = ceil((ymax - ymin) / resolution)

        transform = rasterio.transform.from_bounds(xmin, ymin, xmin + largeur * resolution, ymin + hauteur * resolution, largeur, hauteur)

        #Binary 1
        binary = np.zeros((hauteur, largeur))
        for df in segments :
            for index, point in df.iterrows() : 
                x = floor((point.longitude - xmin) / resolution)
                y = floor((point.latitude - ymin) / resolution)
                binary[hauteur - y - 1][x] = 1
        
        #Heatmap 2
        heatmap = np.zeros((hauteur, largeur))
        for df in segments :
            for index, point in df.iterrows() : 
                x = floor((point.longitude - xmin) / resolution)
                y = floor((point.latitude - ymin) / resolution)
                heatmap[hauteur - y - 1][x] = heatmap[hauteur - y - 1][x] + 1
        
        #Distance 3
        distance = np.zeros((hauteur, largeur))
        for df in segments :
            for index, point in df.iterrows() :
                if isnan(point.distance) : continue
                x = floor((point.longitude - xmin) / resolution)
                y = floor((point.latitude - ymin) / resolution)
                distance[hauteur - y - 1][x] = distance[hauteur - y - 1][x] + point.distance
        distance = np.divide(distance, heatmap, out=np.zeros_like(distance), where=heatmap!=0)

        #Speed 4
        speed = np.zeros((hauteur, largeur))
        for df in segments :
            for index, point in df.iterrows() :
                if isnan(point.speed) : continue
                x = floor((point.longitude - xmin) / resolution)
                y = floor((point.latitude - ymin) / resolution)
                speed[hauteur - y - 1][x] = speed[hauteur - y - 1][x] + point.speed
        speed = np.divide(speed, heatmap, out=np.zeros_like(speed), where=heatmap!=0)

        #Acceleration 5
        acceleration = np.zeros((hauteur, largeur))
        for df in segments :
            for index, point in df.iterrows() :
                if isnan(point.acceleration) : continue
                x = floor((point.longitude - xmin) / resolution)
                y = floor((point.latitude - ymin) / resolution)
                acceleration[hauteur - y - 1][x] = acceleration[hauteur - y - 1][x] + point.acceleration
        acceleration = np.divide(acceleration, heatmap, out=np.zeros_like(acceleration), where=heatmap!=0)
        
        #Altitude 6
        altitude = rasterio.open(os.path.join('DEM altitude', f"{str(mnt)}.tif"))
        altitude = altitude.read(1, window=rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, altitude.transform))
        altitude = np.array(Image.fromarray(altitude).resize((largeur, hauteur)))
        altitude = np.where(heatmap==0, 0, altitude)
        
        #Slope 7
        slope = np.zeros_like(heatmap)
        slope = rasterio.open(os.path.join('DEM slope', f"{str(mnt)}.tif"))
        slope = slope.read(1, window=rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, slope.transform))
        slope = np.array(Image.fromarray(slope).resize((largeur, hauteur)))
        slope = np.where(heatmap==0, 0, slope)

        #Bearing 8 - 15
        bearing = np.zeros((hauteur, largeur, 8))
        for df in segments :
            for index, point in df.iterrows() :
                if isnan(point.bearing) : continue 
                x = floor((point.longitude - xmin) / resolution)
                y = floor((point.latitude - ymin) / resolution)
                if int(point.bearing) in range(0,45) : bearing[hauteur - y - 1][x][0] = bearing[hauteur - y - 1][x][0] + 1 
                if int(point.bearing) in range(45,90) : bearing[hauteur - y - 1][x][1] = bearing[hauteur - y - 1][x][1] + 1 
                if int(point.bearing) in range(90,135) : bearing[hauteur - y - 1][x][2] = bearing[hauteur - y - 1][x][2] + 1 
                if int(point.bearing) in range(135,181) : bearing[hauteur - y - 1][x][3] = bearing[hauteur - y - 1][x][3] + 1 
                if int(point.bearing) in range(-45,0) : bearing[hauteur - y - 1][x][4] = bearing[hauteur - y - 1][x][4] + 1 
                if int(point.bearing) in range(-90,-45) : bearing[hauteur - y - 1][x][5] = bearing[hauteur - y - 1][x][5] + 1 
                if int(point.bearing) in range(-135,-90) : bearing[hauteur - y - 1][x][6] = bearing[hauteur - y - 1][x][6] + 1 
                if int(point.bearing) in range(-181,-135) : bearing[hauteur - y - 1][x][7] = bearing[hauteur - y - 1][x][7] + 1 
                
        for a in range(0, 8) :
            bearing[:,:,a] = np.divide(bearing[:,:,a], heatmap, out=np.zeros_like(bearing[:,:,a]), where=heatmap!=0)

        #Directional diversity 16
        directional_diversity = np.zeros((hauteur, largeur))
        for y in range(hauteur):
            for x in range(largeur):
                # Calculate the entropy of the 8 directions for the current cell
                prob = bearing[y, x, :]
                prob = prob[prob != 0]  # remove zero entries
                entropy = -np.sum(prob * np.log2(prob))
                directional_diversity[y][x] = entropy

        #Bearing difference 17
        bearing_difference = np.zeros((hauteur, largeur))
        for df in segments :
            for index, point in df.iterrows() :
                if isnan(point.bearing_difference) : continue
                x = floor((point.longitude - xmin) / resolution)
                y = floor((point.latitude - ymin) / resolution)
                bearing_difference[hauteur - y - 1][x] = bearing_difference[hauteur - y - 1][x] + point.bearing_difference
        bearing_difference = np.divide(bearing_difference, heatmap, out=np.zeros_like(bearing_difference), where=heatmap!=0)

        #Compile and save the raster file
        path = os.path.join(datasetDirectory, str(resolution_meters).replace('.', ','))
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, dataset + '.tif')

        new_dataset = rasterio.open(file_path, 'w', driver='GTiff',
                                    height = hauteur, width = largeur,
                                    count=17, dtype=heatmap.dtype,
                                    crs=rasterio.open(os.path.join('DEM ALTITUDE', '68.tif')).crs,
                                    transform=transform)
        
        new_dataset.write(binary, 1)
        new_dataset.write(heatmap, 2)
        new_dataset.write(distance, 3)
        new_dataset.write(speed, 4)
        new_dataset.write(acceleration, 5)
        new_dataset.write(altitude, 6)
        new_dataset.write(slope, 7)
        new_dataset.write(bearing[:,:,0], 8)
        new_dataset.write(bearing[:,:,1], 9)
        new_dataset.write(bearing[:,:,2], 10)
        new_dataset.write(bearing[:,:,3], 11)
        new_dataset.write(bearing[:,:,4], 12)
        new_dataset.write(bearing[:,:,5], 13)
        new_dataset.write(bearing[:,:,6], 14)
        new_dataset.write(bearing[:,:,7], 15)
        new_dataset.write(directional_diversity, 16)
        new_dataset.write(bearing_difference, 17)
        new_dataset.close()
        
        print(f"Done {dataset} at resolution {str(resolution_meters)}m")