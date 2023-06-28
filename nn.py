# ---------------------------------------------------------------------------------
# File: nn.py
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
import tensorflow as tf
from glob import glob
import numpy as np
from PIL import Image
import rasterio

from models import UNet, dice_loss

def trainModel(confTraining, confData, train_directory='train_examples'):
    """
    This function creates and trains a machine learning model based on the provided configurations.

    Args:
    confTraining (dict): The configuration for training, which includes settings such as the batch size, number of epochs, etc.
    confData (dict): The configuration for the data, which includes settings such as the channels, image size, etc.

    Returns:
    tuple: A tuple containing the trained model and the history of its training process.
    """

    # Build the input pipeline
    dataset = tf.data.Dataset.list_files(glob(f'{train_directory}/*/*'), shuffle=True)

    dataset = dataset.map(
        lambda x : loadData(x, confData['channels']),
        num_parallel_calls=tf.data.AUTOTUNE)

    # Split the dataset into training and validation sets
    train = dataset.take(int(0.8 * len(dataset))).cache().batch(confTraining['batch_size']).prefetch(buffer_size=tf.data.AUTOTUNE)
    val = dataset.skip(int(0.8 * len(dataset))).batch(confTraining['batch_size'])

    # Define the model input
    inputs = tf.keras.layers.Input((confData['image_size'], confData['image_size'], dataset.element_spec[0].shape[2]))

    # Build the model using the specified network function
    function = globals()[confTraining['network']]
    model = tf.keras.models.Model(inputs=inputs,outputs=function(inputs, units=confTraining['units']))

    # Create the early stopping callback
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=5,
        verbose=1,
        mode="min",
        baseline=None,
        restore_best_weights=True,
    )

    # Create the learning rate schedule callback
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.1, 
        patience=3, 
        min_lr=0.001/100000, 
        verbose=1
    )

    # Compile the model
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=globals()[confTraining['loss']],
        metrics=["accuracy"]
    )

    # Train the model
    history = model.fit(
        train,
        validation_data=val,
        epochs = confTraining['epoch'],
        callbacks=[lr_scheduler, callback]
    )

    return model, history

def doPredictions(patchs, model, confData, pas, pad, directory='test_examples'):
    """
    This function is used to generate predictions for each image patch saved in a directory, and then combine them into a single larger image.
    
    Parameters:
    model (keras.Model): The trained model used for predictions.
    confData (dict): The configuration data containing the image size and channels.
    pas (int): The number of pixels that the window is moved in each step for the sliding window approach.
    pad (int): The padding size to be added.
    directory (str): The directory where the image patches are stored.

    Returns:
    result (np.array): The final combined image prediction.
    """

    # Perform prediction for each patch and save the results
    predicted_patchs = np.zeros((patchs.shape[0], patchs.shape[1], patchs.shape[2], patchs.shape[3]))
    for w in range(patchs.shape[0]):
        for h in range(patchs.shape[1]):
            patch = patchs[w, h, :, :, :]
            patch = patch / 255
            predicted_patchs[w, h, :, :] = model.predict(np.expand_dims(patch, axis=0), verbose=0)[0, :, :, 0]

    # Combine all the predicted patches into a single image
    result = np.zeros(shape=(patchs.shape[1] * int(pas) + confData['image_size'],  patchs.shape[0] * int(pas) + confData['image_size']))
    for w in range(patchs.shape[0]):
        for h in range(patchs.shape[1]):
            patch = predicted_patchs[w, h, :, :]
            j = w * int(pas)
            i = h * int(pas)
            result[i+pad:i+confData['image_size']-pad, j+pad:j+confData['image_size']-pad] = np.mean( np.array([ result[i+pad:i+confData['image_size']-pad, j+pad:j+confData['image_size']-pad], patch[pad:confData['image_size']-pad, pad:confData['image_size']-pad] ]), axis=0 )
      
    return result

def loadData(folder, channels):
    """
    This function reads the image files from a given folder path and returns the features and labels.
    
    Args:
    folder (str): The path to the folder where the image files are located.
    channels (list): The list of channel names to read.

    Returns:
    tuple: A tuple containing the features and labels as numpy arrays.
    """

    features = []
    
    # Loop through each channel and load the corresponding image file
    for c in channels:
        # Decode the jpeg image file and append to the features list
        features.append(tf.experimental.numpy.squeeze(tf.io.decode_jpeg(tf.io.read_file(folder + '/' + c + '.png'), channels=1), axis=2) / 255)

    # Stack the features along the third dimension
    f = tf.experimental.numpy.stack(features, axis=2)

    # Load the ground truth image file and convert to labels
    l = tf.experimental.numpy.squeeze(tf.io.decode_jpeg(tf.io.read_file(folder + '/truth.png')), axis=2) / 255

    return f, l

def loadModel(path, confTraining):
    """
    This function loads a pre-trained Keras model from a specified path.

    Args:
    path (str): The file path where the pre-trained model is stored.
    confTraining (dict): The configuration dictionary that includes the loss function used in training.

    Returns:
    keras.Model: The loaded Keras model.
    """

    # Load the Keras model. 
    # 'custom_objects' is used to load the custom loss function from the global namespace.
    return tf.keras.models.load_model(path, custom_objects={confTraining["loss"]: globals()[confTraining["loss"]]}, compile=False)

