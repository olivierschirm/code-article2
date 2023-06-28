# ---------------------------------------------------------------------------------
# File: converter.py
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

import numpy as np

def resultToGeojson(raster, levels, output_file='result.geojson') :
    '''
    Function for converting a raster image into GeoJSON. 
    
    This function takes an input raster image and a list of levels, performs thresholding and 
    skeletonization operations on the raster, then converts the resultant segments and intersections into 
    GeoJSON format. The output is written to a specified file.

    Parameters:
    raster (rasterio object): Input raster image.
    levels (list of int): Threshold levels for raster image.
    output_file (str): Name of output GeoJSON file. Defaults to 'result.geojson'.

    Returns:
    output_file (str): similaire to the parameter.

    '''

    # Extract raster parameters
    res = raster.res
    bounds = raster.bounds
    height, width = raster.shape[0], raster.shape[1]
    raster_array = raster.read(1)

    # Initialize the skeleton matrix
    SkelMat = np.zeros((height + 2, width + 2), dtype=np.uint8)
     
    # Threshold the raster at each level and add to the skeleton matrix
    for level in levels :
        # Initialize threshold matrix
        ThresMat = np.zeros((height + 2, width + 2), dtype=np.bool)
        
        # Find pixels above threshold level
        relevantpixels = np.where((raster_array >= level))
        checkpixels = zip(relevantpixels[0], relevantpixels[1])
        for (row, column) in checkpixels:
            ThresMat[row + 1, column + 1] = True
    
        # Add thresholded pixels to skeleton matrix
        SkelMat = np.add(ThresMat, SkelMat)
        convergence = True

        # Apply Zhang-Suen thinning algorithm
        while convergence :
            convergence = False
    
            # Subset 1
            newSkelMat = SkelMat.copy()
            relevantpixels = np.where(SkelMat >= 1)
            checkpixels = zip(relevantpixels[0], relevantpixels[1])
            for (row, column) in checkpixels:
                B = 0
                A = 0
                P2 = SkelMat[row - 1, column]
                P3 = SkelMat[row - 1, column + 1]
                P4 = SkelMat[row, column + 1]
                P5 = SkelMat[row + 1, column + 1]
                P6 = SkelMat[row + 1, column]
                P7 = SkelMat[row + 1, column - 1]
                P8 = SkelMat[row, column - 1]
                P9 = SkelMat[row - 1, column - 1]
                if P2 > 0: 
                    B = B + 1
                    if P9 == 0:
                        A = A + 1
                if P3 > 0:
                    B = B + 1
                    if P2 == 0:
                        A = A + 1
                if P4 > 0:
                    B = B + 1
                    if P3 == 0:
                        A = A + 1
                if P5 > 0:
                    B = B + 1
                    if P4 == 0:
                        A = A + 1
                if P6 > 0:
                    B = B + 1
                    if P5 == 0:
                        A = A + 1
                if P7 > 0:
                    B = B + 1
                    if P6 == 0:
                        A = A + 1
                if P8 > 0:
                    B = B + 1
                    if P7 == 0:
                        A = A + 1
                if P9 > 0:
                    B = B + 1
                    if P8 == 0:
                        A = A + 1
                        
                tozero = (B >= 2 and B <= 6 and A == 1 and (P2 == 0 or P4 == 0 or P6 == 0) and (P4 == 0 or P6 == 0 or P8 == 0))
    
                if tozero:
                    convergence = True
                    newSkelMat[row, column] = 0
     
            SkelMat = newSkelMat

            #subset2
            newSkelMat = SkelMat.copy()
            relevantpixels = np.where(SkelMat >= 1)
            checkpixels = zip(relevantpixels[0], relevantpixels[1])
    
            for (row, column) in checkpixels:
                B = 0
                A = 0
                P9 = SkelMat[row - 1, column - 1]
                P2 = SkelMat[row - 1, column]
                P3 = SkelMat[row - 1, column + 1]
                P4 = SkelMat[row, column + 1]
                P5 = SkelMat[row + 1, column + 1]
                P6 = SkelMat[row + 1, column]
                P7 = SkelMat[row + 1, column - 1]
                P8 = SkelMat[row, column - 1]
                if P2 > 0:
                    B = B + 1
                    if P9 == 0:
                        A = A + 1
                if P3 > 0:
                    B = B + 1
                    if P2 == 0:
                        A = A + 1
                if P4 > 0:
                    B = B + 1
                    if P3 == 0:
                        A = A + 1
                if P5 > 0:
                    B = B + 1
                    if P4 == 0:
                        A = A + 1
                if P6 > 0:
                    B = B + 1
                    if P5 == 0:
                        A = A + 1
                if P7 > 0:
                    B = B + 1
                    if P6 == 0:
                        A = A + 1
                if P8 > 0:
                    B = B + 1
                    if P7 == 0:
                        A = A + 1
                if P9 > 0:
                    B = B + 1
                    if P8 == 0:
                        A = A + 1
                tozero = (B >= 2 and B <= 6 and A == 1 and (P2 == 0 or P4 == 0 or P8 == 0) and (P2 == 0 or P6 == 0 or P8 == 0))
    
                if tozero:
                    convergence = True
                    newSkelMat[row, column] = 0
     
            SkelMat = newSkelMat
        
        #step2
        newSkelMat = SkelMat.copy()
        relevantpixels = np.where(SkelMat >= 1)
        checkpixels = zip(relevantpixels[0], relevantpixels[1])
        for (row, column) in checkpixels:
            B = 0
            P2 = SkelMat[row - 1, column]
            P3 = SkelMat[row - 1, column + 1]
            P4 = SkelMat[row, column + 1]
            P5 = SkelMat[row + 1, column + 1]
            P6 = SkelMat[row + 1, column]
            P7 = SkelMat[row + 1, column - 1]
            P8 = SkelMat[row, column - 1]
            P9 = SkelMat[row - 1, column - 1]
            if P2:
                B = B + 1
            if P3:
                B = B + 1
            if P4:
                B = B + 1
            if P5:
                B = B + 1
            if P6:
                B = B + 1
            if P7:
                B = B + 1
            if P8:
                B = B + 1
            if P9:
                B = B + 1
    
            tozero = (B >= 7)
            if tozero:
                newSkelMat[row, column] = 0
    
        SkelMat = newSkelMat
    
    SkelMat = np.where(SkelMat >= 1, True, SkelMat)
    SkelMat = np.where(SkelMat == 0, False, SkelMat)
    
    SkelMat = np.rot90(SkelMat, k=3)
    
    
    #Create simple lines with cells connection.
    lines = []
    for (cellX, cellY), value in np.ndenumerate(SkelMat) :
        if not value : continue
    
        P2 = SkelMat[cellX - 1, cellY]
        P3 = SkelMat[cellX - 1, cellY + 1]
        P4 = SkelMat[cellX, cellY + 1]
        P5 = SkelMat[cellX + 1, cellY + 1]
        P6 = SkelMat[cellX + 1, cellY]
        P7 = SkelMat[cellX + 1, cellY - 1]
        P8 = SkelMat[cellX, cellY - 1]
        P9 = SkelMat[cellX - 1, cellY - 1]
        
        if P3 and not P2 and not P4 :
            lines.append(np.array([(cellX, cellY), (cellX - 1, cellY + 1)]) - 1)
        
        if P5 and not P4 and not P6 :
            lines.append(np.array([(cellX, cellY), (cellX + 1, cellY + 1)]) - 1)
        
        if P7 and not P6 and not P8 : 
            lines.append(np.array([(cellX, cellY), (cellX + 1, cellY - 1)]) - 1)
        
        if P9 and not P8 and not P2 :
            lines.append(np.array([(cellX, cellY), (cellX - 1, cellY - 1)]) - 1)
            
        if P2 :
            lines.append(np.array([(cellX, cellY), (cellX - 1, cellY)]) - 1)
            
        if P4 :
            lines.append(np.array([(cellX, cellY), (cellX, cellY + 1)]) - 1)
        
        if P6 :
            lines.append(np.array([(cellX, cellY), (cellX + 1, cellY)]) - 1)
    
        if P8 :
            lines.append(np.array([(cellX, cellY), (cellX, cellY - 1)]) - 1)
    
    
    #Remove duplicate lines
    i = 0
    while True :
        if i == len(lines) :break
        line_inv = np.flip(lines[i], axis=0)
        
        for l in range(i, len(lines)) :
            if (lines[l] == line_inv).all() :
                lines.pop(l)
                i = i + 1
                break
    
    ## create segments and intersections
    points = []
    for line in lines :
        points.append(tuple(line[0]))
        points.append(tuple(line[1]))
        
    intersections = list(dict.fromkeys([ point for point in points if points.count(point) > 2 ]))
    ends = [ point for point in points if points.count(point) == 1]
    
    
    def isendingline(line) :
        pt1in = tuple((line[0][0], line[0][1])) in intersections or tuple((line[0][0], line[0][1])) in ends
        pt2in = tuple((line[1][0], line[1][1])) in intersections or tuple((line[1][0], line[1][1])) in ends
        return pt1in or pt2in
    
     
    def findNextLine(point) :
        for l in lines :
            if l[0] == point or l[1] == point :
                return l
        return False
        
    
    
    lines = [ [tuple((l[0][0], l[0][1])), tuple((l[1][0], l[1][1]))] for l in lines]
    
    
    segments = []
    while len(lines) > 0:
        segment = []
        
        for line in lines :
            if not isendingline(line) : continue
        
            lines.pop(lines.index(line))
            
            if line[0] in intersections or line[0] in ends :
                segment.append(line[0])
                segment.append(line[1])
            else :
                segment.append(line[1])
                segment.append(line[0])
            
            if (line[0] in intersections or line[0] in ends) and (line[1] in intersections or line[1] in ends) :
                break
    
            
            while True :
                line2 = findNextLine(segment[-1])
                if line2[0] == segment[-1] :
                    segment.append(line2[1])
                else :
                    segment.append(line2[0])
                
                lines.pop(lines.index(line2))
                
                if isendingline(line2) : break
                    
            break
        
        segments.append(segment)
    
    SkelMat = SkelMat[1:-1, 1:-1]
     
    #Create geojson
    xmin = bounds[0]
    ymin = bounds[1]
    for segment in segments :
        for (x,y) in segment:
            segment[segment.index((x,y))] = (xmin + x * res[0] + res[0]/2, ymin + y * res[1] + res[1]/2)
    
    for intersection in intersections :
        intersections[intersections.index(intersection)] = (xmin + intersection[0] * res[0] + res[0]/2, ymin + intersection[1] * res[1] + res[1]/2)
    
    
    from geojson import Point, Feature, FeatureCollection, dump, LineString
    features = []
    
    for segment in segments :
        lineString = LineString(segment)
        features.append(Feature(geometry=lineString))
    
    for intersection in intersections :
        features.append(Feature(geometry=Point(intersection), properties={'marker-color': '#3fc700'}))
    
    feature_collection = FeatureCollection(features)
    
    # Write GeoJSON to file
    with open(output_file, 'w') as f:
       dump(feature_collection, f)

    return output_file


