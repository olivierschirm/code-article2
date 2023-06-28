# ---------------------------------------------------------------------------------
# File: models.py
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

from tensorflow.keras.layers import MaxPooling2D, Dropout, UpSampling2D, concatenate, Conv2D, BatchNormalization, Activation
from keras import backend as K

def UNet(x, units, kernel_size= 3, dropout = 0.1, batchnorm = True):
    """
    This function builds a U-Net model for semantic segmentation.

    Args:
    x (tf.Tensor): The input tensor.
    units (int): The number of output units for the convolutional layers.
    kernel_size (int, optional): The size of the kernel to use in the convolutional layers. Default is 3.
    dropout (float, optional): The dropout rate to use after each layer. Default is 0.1.
    batchnorm (bool, optional): Whether to apply batch normalization after each convolutional layer. Default is True.

    Returns:
    tf.Tensor: The output tensor of the model.
    """

    # Contracting path
    c1 = unet_conv_block(x, units * 1, kernel_size, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = unet_conv_block(p1, units * 2, kernel_size, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = unet_conv_block(p2, units * 4, kernel_size, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = unet_conv_block(p3, units * 8, kernel_size, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = unet_conv_block(p4, units * 16, kernel_size, batchnorm = batchnorm)
    
    # Expansive path
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = unet_conv_block(u6, units * 8, kernel_size, batchnorm = batchnorm)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = unet_conv_block(u7, units * 4, kernel_size, batchnorm = batchnorm)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = unet_conv_block(u8, units * 2, kernel_size, batchnorm = batchnorm)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1], axis = 3)
    u9 = Dropout(dropout)(u9)
    c9 = unet_conv_block(u9, units * 1, kernel_size, batchnorm = batchnorm)
    
    # Final output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    return outputs

def unet_conv_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """
    This function creates a two-layer Conv2D block with optional batch normalization.

    Args:
    input_tensor (tf.Tensor): The input tensor.
    n_filters (int): The number of output filters for the convolutional layers.
    kernel_size (int, optional): The size of the kernel to use in the convolutional layers. Default is 3.
    batchnorm (bool, optional): Whether to apply batch normalization after each convolutional layer. Default is True.

    Returns:
    tf.Tensor: The output tensor of the block.
    """

    # First layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def dice_loss(y_true, y_pred):
    """
    This function computes the Dice loss between the predicted and true segmentation masks.

    Dice loss is a commonly used loss function for image segmentation tasks. It measures 
    the overlap between two samples and is calculated as: 
        Dice Loss = 1 - (2 * |Intersection| + smooth) / (|y_true| + |y_pred| + smooth)

    Args:
    y_true (tf.Tensor): The ground truth segmentation masks.
    y_pred (tf.Tensor): The predicted segmentation masks.

    Returns:
    float: The Dice loss between the ground truth and predicted masks.
    """

    smooth = 1.  # Smoothing factor to avoid division by zero

    # Flatten the input tensors
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    # Calculate intersection of predicted and ground truth masks
    intersection = K.sum(y_true_f * y_pred_f)
    
    # Compute Dice loss
    return 1 - ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
