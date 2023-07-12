from keras.optimizers import RMSprop
from keras.losses import binary_crossentropy

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, Input, Reshape, Activation, Add, Multiply, LeakyReLU
from keras.layers import concatenate, Conv2DTranspose, BatchNormalization
from keras import backend as K
from keras.layers import Input
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import *
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import Adam
from keras.layers import LeakyReLU

from tensorflow import keras

























def dlinknet_residual_block(input_tensor, num_filters):
    x = Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = Conv2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    input_tensor = Conv2D(num_filters, (1, 1), padding='same')(input_tensor)
    input_tensor = BatchNormalization()(input_tensor)
    res_tensor = Add()([input_tensor, x])
    res_tensor = Activation('relu')(res_tensor)
    return res_tensor


def dlinknet_dilated_center_block(input_tensor, num_filters):

    dilation_1 = Conv2D(num_filters, kernel_size=(3, 3), dilation_rate=(1, 1), padding='same')(input_tensor)
    dilation_1 = Activation('relu')(dilation_1)

    dilation_2 = Conv2D(num_filters, kernel_size=(3, 3), dilation_rate=(2, 2), padding='same')(dilation_1)
    dilation_2 = Activation('relu')(dilation_2)

    dilation_4 = Conv2D(num_filters, kernel_size=(3, 3), dilation_rate=(4, 4), padding='same')(dilation_2)
    dilation_4 = Activation('relu')(dilation_4)

    dilation_8 = Conv2D(num_filters, kernel_size=(3, 3), dilation_rate=(8, 8), padding='same')(dilation_4)
    dilation_8 = Activation('relu')(dilation_8)

    final_diliation = Add()([input_tensor, dilation_1, dilation_2, dilation_4, dilation_8])

    return final_diliation


def dlinknet_decoder_block(input_tensor, num_filters):
    decoder_tensor = Conv2D(num_filters, (1, 1), padding='same')(input_tensor)
    decoder_tensor = BatchNormalization()(decoder_tensor)
    decoder_tensor = Activation('relu')(decoder_tensor)

    decoder_tensor = Conv2DTranspose(num_filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(decoder_tensor)
    decoder_tensor = BatchNormalization()(decoder_tensor)
    decoder_tensor = Activation('relu')(decoder_tensor)

    decoder_tensor = Conv2D(num_filters, (1, 1), padding='same')(decoder_tensor)
    decoder_tensor = BatchNormalization()(decoder_tensor)
    decoder_tensor = Activation('relu')(decoder_tensor)
    return decoder_tensor


def dlinknet_encoder_block(input_tensor, num_filters, num_res_blocks):
    encoded = dlinknet_residual_block(input_tensor, num_filters)
    while num_res_blocks > 1:
        encoded = dlinknet_residual_block(encoded, num_filters)
        num_res_blocks -= 1
    encoded_pool = MaxPooling2D((2, 2), strides=(2, 2))(encoded)
    return encoded, encoded_pool


def DLinkNet(x, units):

    inputs_ = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    inputs_ = BatchNormalization()(inputs_)
    inputs_ = Activation('relu')(inputs_)
    max_pool_inputs = MaxPooling2D((2, 2), strides=(2, 2))(inputs_)

    encoded_1, encoded_pool_1 = dlinknet_encoder_block(max_pool_inputs, num_filters=64, num_res_blocks=3)
    encoded_2, encoded_pool_2 = dlinknet_encoder_block(encoded_pool_1, num_filters=128, num_res_blocks=4)
    encoded_3, encoded_pool_3 = dlinknet_encoder_block(encoded_pool_2, num_filters=256, num_res_blocks=6)
    encoded_4, encoded_pool_4 = dlinknet_encoder_block(encoded_pool_3, num_filters=512, num_res_blocks=3)

    center = dlinknet_dilated_center_block(encoded_4, 512)

    decoded_1 = Add()([dlinknet_decoder_block(center, 256), encoded_3])
    decoded_2 = Add()([dlinknet_decoder_block(decoded_1, 128), encoded_2])
    decoded_3 = Add()([dlinknet_decoder_block(decoded_2, 64), encoded_1])
    decoded_4 = dlinknet_decoder_block(decoded_3, 64)

    final = Conv2DTranspose(32, kernel_size=(3, 3), padding='same')(decoded_4)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(final)

    return outputs























def UNet(x, units, kernel_size= 3, dropout = 0.1, batchnorm = True):
    """
    This function builds a U-Net model for semantic segmentation.

    Args:
    x (tf.Tensor): The input tensor.
    units (int): The number of output units for the convolutional
    kernel_size (int, optional): The size of the kernel to use in the convolutional  Default is 3.
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
    n_filters (int): The number of output filters for the convolutional
    kernel_size (int, optional): The size of the kernel to use in the convolutional  Default is 3.
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




















#https://github.com/bonlime/keras-deeplab-v3-plus
def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation(tf.nn.relu)(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation(tf.nn.relu)(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation(tf.nn.relu)(x)

    return x


def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    """Implements right 'same' padding for even kernel sizes
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)


def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                    rate=1, depth_activation=False, return_skip=False):
    """ Basic building block of modified Xception network
        Args:
            inputs: input tensor
            depth_list: number of filters in each SepConv layer. len(depth_list) == 3
            prefix: prefix before name
            skip_connection_type: one of {'conv','sum','none'}
            stride: stride at last depthwise conv
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & pointwise convs
            return_skip: flag to return additional tensor after 2 SepConvs for decoder
            """
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                kernel_size=1,
                                stride=stride)
        shortcut = BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
    in_channels = inputs.shape[-1]  # inputs._keras_shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'expanded_conv_{}_'.format(block_id)
    if block_id:
        # Expand

        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                   use_bias=False, activation=None,
                   name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'expand_BN')(x)
        x = Activation(tf.nn.relu6, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'
    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                        use_bias=False, padding='same', dilation_rate=(rate, rate),
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'depthwise_BN')(x)

    x = Activation(tf.nn.relu6, name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters,
               kernel_size=1, padding='same', use_bias=False, activation=None,
               name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'project_BN')(x)

    if skip_connection:
        return Add(name=prefix + 'add')([inputs, x])

    # if in_channels == pointwise_filters and stride == 1:
    #    return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x


def DeeplabV3Plus(input_tensor, units, weights=None, classes=1, backbone='mobilenetv2',
            alpha=1., activation='sigmoid'):

    input_shape=(input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3])
    """ Instantiates the Deeplabv3+ architecture
    Optionally loads weights pre-trained
    on PASCAL VOC or Cityscapes. This model is available for TensorFlow only.
    # Arguments
        weights: one of 'pascal_voc' (pre-trained on pascal voc),
            'cityscapes' (pre-trained on cityscape) or None (random initialization)
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: shape of input image. format HxWxC
            PASCAL VOC model was trained on (512,512,3) images. None is allowed as shape/width
        classes: number of desired classes. PASCAL VOC has 21 classes, Cityscapes has 19 classes.
            If number of classes not aligned with the weights used, last layer is initialized randomly
        backbone: backbone to use. one of {'xception','mobilenetv2'}
        activation: optional activation to add to the top of the network.
            One of 'softmax', 'sigmoid' or None
        OS: determines input_shape/feature_extractor_output ratio. One of {8,16}.
            Used only for xception backbone.
        alpha: controls the width of the MobileNetV2 network. This is known as the
            width multiplier in the MobileNetV2 paper.
                - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                    are used at each layer.
            Used only for mobilenetv2 backbone. Pretrained is only available for alpha=1.
    # Returns
        A Keras model instance.
    # Raises
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
        ValueError: in case of invalid argument for `weights` or `backbone`
    """

    if not (weights in {'pascal_voc', 'cityscapes', None}):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `pascal_voc`, or `cityscapes` '
                         '(pre-trained on PASCAL VOC)')

    if not (backbone in {'xception', 'mobilenetv2'}):
        raise ValueError('The `backbone` argument should be either '
                         '`xception`  or `mobilenetv2` ')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor

    if backbone == 'xception':
        if OS == 8:
            entry_block3_stride = 1
            middle_block_rate = 2  # ! Not mentioned in paper, but required
            exit_block_rates = (2, 4)
            atrous_rates = (12, 24, 36)
        else:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
            atrous_rates = (6, 12, 18)

        x = Conv2D(32, (3, 3), strides=(2, 2),
                   name='entry_flow_conv1_1', use_bias=False, padding='same')(img_input)
        x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
        x = Activation(tf.nn.relu)(x)

        x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
        x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
        x = Activation(tf.nn.relu)(x)

        x = _xception_block(x, [128, 128, 128], 'entry_flow_block1',
                            skip_connection_type='conv', stride=2,
                            depth_activation=False)
        x, skip1 = _xception_block(x, [256, 256, 256], 'entry_flow_block2',
                                   skip_connection_type='conv', stride=2,
                                   depth_activation=False, return_skip=True)

        x = _xception_block(x, [728, 728, 728], 'entry_flow_block3',
                            skip_connection_type='conv', stride=entry_block3_stride,
                            depth_activation=False)
        for i in range(16):
            x = _xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                                skip_connection_type='sum', stride=1, rate=middle_block_rate,
                                depth_activation=False)

        x = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                            skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                            depth_activation=False)
        x = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                            skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                            depth_activation=True)

    else:
        OS = 8
        first_block_filters = _make_divisible(32 * alpha, 8)
        x = Conv2D(first_block_filters,
                   kernel_size=3,
                   strides=(2, 2), padding='same', use_bias=False,
                   name='Conv' if input_shape[2] == 3 else 'Conv_')(img_input)
        x = BatchNormalization(
            epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
        x = Activation(tf.nn.relu6, name='Conv_Relu6')(x)

        x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                                expansion=1, block_id=0, skip_connection=False)

        x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                                expansion=6, block_id=1, skip_connection=False)
        x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                                expansion=6, block_id=2, skip_connection=True)

        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                                expansion=6, block_id=3, skip_connection=False)
        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                expansion=6, block_id=4, skip_connection=True)
        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                expansion=6, block_id=5, skip_connection=True)

        # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,  # 1!
                                expansion=6, block_id=6, skip_connection=False)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=7, skip_connection=True)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=8, skip_connection=True)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=9, skip_connection=True)

        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=10, skip_connection=False)
        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=11, skip_connection=True)
        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=12, skip_connection=True)

        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2,  # 1!
                                expansion=6, block_id=13, skip_connection=False)
        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                                expansion=6, block_id=14, skip_connection=True)
        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                                expansion=6, block_id=15, skip_connection=True)

        x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4,
                                expansion=6, block_id=16, skip_connection=False)

    # end of feature extractor

    # branching for Atrous Spatial Pyramid Pooling

    # Image Feature branch
    shape_before = tf.shape(x)
    b4 = GlobalAveragePooling2D()(x)
    b4_shape = tf.keras.backend.int_shape(b4)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Reshape((1, 1, b4_shape[1]))(b4)
    b4 = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation(tf.nn.relu)(b4)
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    b4 = tf.keras.layers.experimental.preprocessing.Resizing(
            *size_before[1:3], interpolation="bilinear"
        )(b4)
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation(tf.nn.relu, name='aspp0_activation')(b0)

    # there are only 2 branches in mobilenetV2. not sure why
    if backbone == 'xception':
        # rate = 6 (12)
        b1 = SepConv_BN(x, 256, 'aspp1',
                        rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
        # rate = 12 (24)
        b2 = SepConv_BN(x, 256, 'aspp2',
                        rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
        # rate = 18 (36)
        b3 = SepConv_BN(x, 256, 'aspp3',
                        rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

        # concatenate ASPP branches & project
        x = Concatenate()([b4, b0, b1, b2, b3])
    else:
        x = Concatenate()([b4, b0])

    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation(tf.nn.relu)(x)
    x = Dropout(0.1)(x)
    # DeepLab v.3+ decoder

    if backbone == 'xception':
        # Feature projection
        # x4 (x2) block
        skip_size = tf.keras.backend.int_shape(skip1)
        x = tf.keras.layers.experimental.preprocessing.Resizing(
                *skip_size[1:3], interpolation="bilinear"
            )(x)
        dec_skip1 = Conv2D(48, (1, 1), padding='same',
                           use_bias=False, name='feature_projection0')(skip1)
        dec_skip1 = BatchNormalization(
            name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
        dec_skip1 = Activation(tf.nn.relu)(dec_skip1)
        x = Concatenate()([x, dec_skip1])
        x = SepConv_BN(x, 256, 'decoder_conv0',
                       depth_activation=True, epsilon=1e-5)
        x = SepConv_BN(x, 256, 'decoder_conv1',
                       depth_activation=True, epsilon=1e-5)

    # you can use it with arbitary number of classes
    if (weights == 'pascal_voc' and classes == 21) or (weights == 'cityscapes' and classes == 19):
        last_layer_name = 'logits_semantic'
    else:
        last_layer_name = 'custom_logits_semantic'

    x = Conv2D(classes, (1, 1), padding='same', name=last_layer_name)(x)
    size_before3 = tf.keras.backend.int_shape(img_input)
    x = tf.keras.layers.experimental.preprocessing.Resizing(
            *size_before3[1:3], interpolation="bilinear"
        )(x)
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.


    if activation in {'softmax', 'sigmoid'}:
        x = tf.keras.layers.Activation(activation)(x)

    return x



































from tensorflow.keras.applications import VGG16



def Segnet_conv_bn_relu(x, filters, kernel_size=1, strides=1):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def Segnet(x, units, num_classes=1):
    base_model = VGG16(weights=None, include_top=False, input_tensor=x)
    x = base_model.output

    # Decoder
    x = UpSampling2D(size=(2, 2))(x)
    x = Segnet_conv_bn_relu(x, 512, 3, strides=1)
    x = Segnet_conv_bn_relu(x, 512, 3, strides=1)
    x = Segnet_conv_bn_relu(x, 512, 3, strides=1)

    x = UpSampling2D(size=(2, 2))(x)
    x = Segnet_conv_bn_relu(x, 512, 3, strides=1)
    x = Segnet_conv_bn_relu(x, 512, 3, strides=1)
    x = Segnet_conv_bn_relu(x, 256, 3, strides=1)

    x = UpSampling2D(size=(2, 2))(x)
    x = Segnet_conv_bn_relu(x, 256, 3, strides=1)
    x = Segnet_conv_bn_relu(x, 256, 3, strides=1)
    x = Segnet_conv_bn_relu(x, 128, 3, strides=1)

    x = UpSampling2D(size=(2, 2))(x)
    x = Segnet_conv_bn_relu(x, 128, 3, strides=1)
    x = Segnet_conv_bn_relu(x, 64, 3, strides=1)

    x = UpSampling2D(size=(2, 2))(x)
    x = Segnet_conv_bn_relu(x, 64, 3, strides=1)
    x = Conv2D(num_classes, 1, strides=1, activation='sigmoid')(x)

    outputs = x
    return outputs





























from tensorflow.keras.applications import ResNet50

#https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture/blob/main/TensorFlow/resnet50_unet.py
def ResUnet_conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def ResUnet_decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = ResUnet_conv_block(x, num_filters)
    return x

def ResUnet(x, units):
    """ Pre-trained ResNet50 Model """
    resnet50 = ResNet50(include_top=False, weights=None, input_tensor=x)

    """ Encoder """
    s1 = resnet50.get_layer("input_1").output           ## (512 x 512)
    s2 = resnet50.get_layer("conv1_relu").output        ## (256 x 256)
    s3 = resnet50.get_layer("conv2_block3_out").output  ## (128 x 128)
    s4 = resnet50.get_layer("conv3_block4_out").output  ## (64 x 64)

    """ Bridge """
    b1 = resnet50.get_layer("conv4_block6_out").output  ## (32 x 32)

    """ Decoder """
    d1 = ResUnet_decoder_block(b1, s4, 512)                     ## (64 x 64)
    d2 = ResUnet_decoder_block(d1, s3, 256)                     ## (128 x 128)
    d3 = ResUnet_decoder_block(d2, s2, 128)                     ## (256 x 256)
    d4 = ResUnet_decoder_block(d3, s1, 64)                      ## (512 x 512)

    """ Output """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    return outputs
































#https://github.com/niecongchong/HRNet-keras-semantic-segmentation

def HRNet_conv3x3(x, out_filters, strides=(1, 1)):
    x = Conv2D(out_filters, 3, padding='same', strides=strides, use_bias=False, kernel_initializer='he_normal')(x)
    return x


def HRNet_basic_Block(input, out_filters, strides=(1, 1), with_conv_shortcut=False):
    x = HRNet_conv3x3(input, out_filters, strides)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = HRNet_conv3x3(x, out_filters)
    x = BatchNormalization(axis=3)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, input])

    x = Activation('relu')(x)
    return x


def HRNet_bottleneck_Block(input, out_filters, strides=(1, 1), with_conv_shortcut=False):
    expansion = 4
    de_filters = int(out_filters / expansion)

    x = Conv2D(de_filters, 1, use_bias=False, kernel_initializer='he_normal')(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(de_filters, 3, strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(out_filters, 1, use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, input])

    x = Activation('relu')(x)
    return x


def HRNet_stem_net(input):
    x = Conv2D(64, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    # x = Conv2D(64, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=3)(x)
    # x = Activation('relu')(x)

    x = HRNet_bottleneck_Block(x, 256, with_conv_shortcut=True)
    x = HRNet_bottleneck_Block(x, 256, with_conv_shortcut=False)
    x = HRNet_bottleneck_Block(x, 256, with_conv_shortcut=False)
    x = HRNet_bottleneck_Block(x, 256, with_conv_shortcut=False)

    return x


def HRNet_transition_layer1(x, out_filters_list=[32, 64]):
    x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x0 = BatchNormalization(axis=3)(x0)
    x0 = Activation('relu')(x0)

    x1 = Conv2D(out_filters_list[1], 3, strides=(2, 2),
                padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x1 = BatchNormalization(axis=3)(x1)
    x1 = Activation('relu')(x1)

    return [x0, x1]


def HRNet_make_branch1_0(x, out_filters=32):
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def HRNet_make_branch1_1(x, out_filters=64):
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def HRNet_fuse_layer1(x):
    x0_0 = x[0]
    x0_1 = Conv2D(32, 1, use_bias=False, kernel_initializer='he_normal')(x[1])
    x0_1 = BatchNormalization(axis=3)(x0_1)
    x0_1 = UpSampling2D(size=(2, 2))(x0_1)
    x0 = add([x0_0, x0_1])

    x1_0 = Conv2D(64, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x1_0 = BatchNormalization(axis=3)(x1_0)
    x1_1 = x[1]
    x1 = add([x1_0, x1_1])
    return [x0, x1]


def HRNet_transition_layer2(x, out_filters_list=[32, 64, 128]):
    x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x0 = BatchNormalization(axis=3)(x0)
    x0 = Activation('relu')(x0)

    x1 = Conv2D(out_filters_list[1], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x1 = BatchNormalization(axis=3)(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(out_filters_list[2], 3, strides=(2, 2),
                padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x2 = BatchNormalization(axis=3)(x2)
    x2 = Activation('relu')(x2)

    return [x0, x1, x2]


def HRNet_make_branch2_0(x, out_filters=32):
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def HRNet_make_branch2_1(x, out_filters=64):
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def HRNet_make_branch2_2(x, out_filters=128):
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def HRNet_fuse_layer2(x):
    x0_0 = x[0]
    x0_1 = Conv2D(32, 1, use_bias=False, kernel_initializer='he_normal')(x[1])
    x0_1 = BatchNormalization(axis=3)(x0_1)
    x0_1 = UpSampling2D(size=(2, 2))(x0_1)
    x0_2 = Conv2D(32, 1, use_bias=False, kernel_initializer='he_normal')(x[2])
    x0_2 = BatchNormalization(axis=3)(x0_2)
    x0_2 = UpSampling2D(size=(4, 4))(x0_2)
    x0 = add([x0_0, x0_1, x0_2])

    x1_0 = Conv2D(64, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x1_0 = BatchNormalization(axis=3)(x1_0)
    x1_1 = x[1]
    x1_2 = Conv2D(64, 1, use_bias=False, kernel_initializer='he_normal')(x[2])
    x1_2 = BatchNormalization(axis=3)(x1_2)
    x1_2 = UpSampling2D(size=(2, 2))(x1_2)
    x1 = add([x1_0, x1_1, x1_2])

    x2_0 = Conv2D(32, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x2_0 = BatchNormalization(axis=3)(x2_0)
    x2_0 = Activation('relu')(x2_0)
    x2_0 = Conv2D(128, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x2_0)
    x2_0 = BatchNormalization(axis=3)(x2_0)
    x2_1 = Conv2D(128, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x2_1 = BatchNormalization(axis=3)(x2_1)
    x2_2 = x[2]
    x2 = add([x2_0, x2_1, x2_2])
    return [x0, x1, x2]


def HRNet_transition_layer3(x, out_filters_list=[32, 64, 128, 256]):
    x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x0 = BatchNormalization(axis=3)(x0)
    x0 = Activation('relu')(x0)

    x1 = Conv2D(out_filters_list[1], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x1 = BatchNormalization(axis=3)(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(out_filters_list[2], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[2])
    x2 = BatchNormalization(axis=3)(x2)
    x2 = Activation('relu')(x2)

    x3 = Conv2D(out_filters_list[3], 3, strides=(2, 2),
                padding='same', use_bias=False, kernel_initializer='he_normal')(x[2])
    x3 = BatchNormalization(axis=3)(x3)
    x3 = Activation('relu')(x3)

    return [x0, x1, x2, x3]


def HRNet_make_branch3_0(x, out_filters=32):
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def HRNet_make_branch3_1(x, out_filters=64):
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def HRNet_make_branch3_2(x, out_filters=128):
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def HRNet_make_branch3_3(x, out_filters=256):
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    x = HRNet_basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def HRNet_fuse_layer3(x):
    x0_0 = x[0]
    x0_1 = Conv2D(32, 1, use_bias=False, kernel_initializer='he_normal')(x[1])
    x0_1 = BatchNormalization(axis=3)(x0_1)
    x0_1 = UpSampling2D(size=(2, 2))(x0_1)
    x0_2 = Conv2D(32, 1, use_bias=False, kernel_initializer='he_normal')(x[2])
    x0_2 = BatchNormalization(axis=3)(x0_2)
    x0_2 = UpSampling2D(size=(4, 4))(x0_2)
    x0_3 = Conv2D(32, 1, use_bias=False, kernel_initializer='he_normal')(x[3])
    x0_3 = BatchNormalization(axis=3)(x0_3)
    x0_3 = UpSampling2D(size=(8, 8))(x0_3)
    x0 = concatenate([x0_0, x0_1, x0_2, x0_3], axis=-1)
    return x0


def HRNet_final_layer(x, classes=1):
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(classes, 1, use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('sigmoid', name='Classification')(x)
    return x


def HRNet(x, units):
    x = HRNet_stem_net(x)

    x = HRNet_transition_layer1(x)
    x0 = HRNet_make_branch1_0(x[0])
    x1 = HRNet_make_branch1_1(x[1])
    x = HRNet_fuse_layer1([x0, x1])

    x = HRNet_transition_layer2(x)
    x0 = HRNet_make_branch2_0(x[0])
    x1 = HRNet_make_branch2_1(x[1])
    x2 = HRNet_make_branch2_2(x[2])
    x = HRNet_fuse_layer2([x0, x1, x2])

    x = HRNet_transition_layer3(x)
    x0 = HRNet_make_branch3_0(x[0])
    x1 = HRNet_make_branch3_1(x[1])
    x2 = HRNet_make_branch3_2(x[2])
    x3 = HRNet_make_branch3_3(x[3])
    x = HRNet_fuse_layer3([x0, x1, x2, x3])

    out = HRNet_final_layer(x, classes=1)

    return out


















#https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet121

def Densenet_conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def Densenet_decoder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = Densenet_conv_block(x, num_filters)
    return x

def DenseNet(x, units):
    """ Pre-trained DenseNet121 Model """
    densenet = DenseNet121(include_top=False, weights=None, input_tensor=x)

    """ Encoder """
    s1 = densenet.get_layer("input_1").output       ## 512
    s2 = densenet.get_layer("conv1/relu").output    ## 256
    s3 = densenet.get_layer("pool2_relu").output ## 128
    s4 = densenet.get_layer("pool3_relu").output  ## 64

    """ Bridge """
    b1 = densenet.get_layer("pool4_relu").output  ## 32

    """ Decoder """
    d1 = Densenet_decoder_block(b1, s4, 512)             ## 64
    d2 = Densenet_decoder_block(d1, s3, 256)             ## 128
    d3 = Densenet_decoder_block(d2, s2, 128)             ## 256
    d4 = Densenet_decoder_block(d3, s1, 64)              ## 512

    """ Outputs """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    return outputs





















































#https://github.com/mribrahim/inception-unet/blob/master/Inception.py

def Inception_conv_block_simple(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = BatchNormalization(name=prefix + "_bn")(conv)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv

def Inception_conv_block_simple_no_bn(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv

def Inception_block(prevlayer, a, b, pooling):
    conva = Conv2D(a, (3, 3), activation='relu', padding='same')(prevlayer)
    conva = BatchNormalization()(conva)
    conva = Conv2D(b, (3, 3), activation='relu', padding='same')(conva)
    conva = BatchNormalization()(conva)
    if True == pooling:
        conva = MaxPooling2D(pool_size=(2, 2))(conva)
    
    
    convb = Conv2D(a, (5, 5), activation='relu', padding='same')(prevlayer)
    convb = BatchNormalization()(convb)
    convb = Conv2D(b, (5, 5), activation='relu', padding='same')(convb)
    convb = BatchNormalization()(convb)
    if True == pooling:
        convb = MaxPooling2D(pool_size=(2, 2))(convb)

    convc = Conv2D(b, (1, 1), activation='relu', padding='same')(prevlayer)
    convc = BatchNormalization()(convc)
    if True == pooling:
        convc = MaxPooling2D(pool_size=(2, 2))(convc)
        
    convd = Conv2D(a, (3, 3), activation='relu', padding='same')(prevlayer)
    convd = BatchNormalization()(convd)
    convd = Conv2D(b, (1, 1), activation='relu', padding='same')(convd)
    convd = BatchNormalization()(convd)
    if True == pooling:
        convd = MaxPooling2D(pool_size=(2, 2))(convd)
        
    up = concatenate([conva, convb, convc, convd])
    return up
def Inception(inputs, units):    
    conv1 = Inception_block(inputs, 8, 16, True)  # Divide filter numbers by 2
    
    conv2 = Inception_block(conv1, 16, 32, True)  # Divide filter numbers by 2

    conv3 = Inception_block(conv2, 32, 64, True)  # Divide filter numbers by 2
    
    conv4 = Inception_block(conv3, 64, 128, True)  # Divide filter numbers by 2
    
    conv5 = Inception_block(conv4, 128, 256, True)  # Divide filter numbers by 2
    
    # **** decoding ****
    xx = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)  # Divide filter numbers by 2
    up1 = Inception_block(xx, 256, 64, False)  # Divide filter numbers by 2
    
    xx = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(up1), conv3], axis=3)  # Divide filter numbers by 2
    up2 = Inception_block(xx, 128, 32, False)  # Divide filter numbers by 2
    
    xx = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(up2), conv2], axis=3)  # Divide filter numbers by 2
    up3 = Inception_block(xx, 64, 16, False)  # Divide filter numbers by 2

    xx = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(up3), conv1], axis=3)  # Divide filter numbers by 2
    up4 = Inception_block(xx, 32, 8, False)  # Divide filter numbers by 2

    xx = concatenate([Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(up4), inputs], axis=3)  # Divide filter numbers by 2

    xx = Conv2D(16, (3, 3), activation='relu', padding='same')(xx)  # Divide filter numbers by 2
    
    xx = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(xx)

    return xx