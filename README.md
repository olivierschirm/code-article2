# Article 2: Code and Experiments

This code is associated with the paper ...

This code is the property of Visorando company.

## How to Run Your Own code:
1. Create the dataset associated with the traces directory you want to work with (use _createDataset.py_).
2. Create the truth associated with the traces truth directory you want to work with (use _createTruth.py_).
3. Follow the instructions in the _main.ipynb_ Jupyter notebook.

The implementation aims to be highly flexible to allow testing of multiple configurations.

## Directory Descriptions:
- **DEM Altitude, DEM Slope**: Altitude and Slope data found on https://geoservices.ign.fr/telechargement
- **Traces**: Extractions of GPS traces from the Visorando database.
- **Traces Truth**: Corresponding manually labeled truth vector used for training.

## File Descriptions:
- **main.ipynb**: Contains the entire process from datasets to exploitable GeoJSON result, including evaluation.
- **evaluator.py**: Includes all the evaluation functions. The process is described in `main_evaluation()`.
- **createDataset.py**: Creates raster inputs from GPX traces.
- **createTruth.py**: Creates truth raster from manually labeled vector files in the traces truth directory.
- **imageTiling.py**: Contains all the functions needed to tile rasters.
- **nn.py**: TensorFlow file where training and predictions are set.
- **models.py**: Contains neuronal network architectures and loss functions.
- **definitions.py**: Includes common functions and constant definitions.
- **truthhome.geojson**: Example of homemade truth for evaluation, based on Ribeauville truth vector.

## Experiments and results
Additionnaly we provide your experiment results in **channel impact** and **channel importance** folder.
