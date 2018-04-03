import math

from keras.layers import Input, Conv2D, MaxPooling2D, Cropping2D, \
                         UpSampling2D, concatenate, ZeroPadding2D
from keras.models import Model


def downconv_block(i, filters, shape, activation='relu', padding='same',
                   data_format='channels_first'):
    """Create a downsampling block.

    The downsampling block is described as two 3x3 convolutions followed by a
    2x2 max pooling operation.

    Parameters
    ----------
    i : `keras.layers.Layer`
        Input layer for this upconv block.
    filters : int
        The number of filters for the first two convolutional layers in the
        block. The final layer has ``filters / 2`` layers.
    shape : int or tuple of int
        Shape of the convolutional filters.

    Other Parameters
    ----------------
    activation : str or `keras.layers.Activation`
        The activation function to apply to the layers.
    padding : {'same','valid'}
        The padding to apply to the convolutional layers.
    data_format : {'channels_first','channels_last'}
        Determines which dimension the convolutional filters are defined along.

    Returns
    -------
    downsample : `keras.layers.MaxPooling2D`
        The max pooling layer
    conv : `keras.layers.Conv2D`
        The final convolutional layer for use in a bypass connection.
    """
    c1 = Conv2D(filters, shape, activation=activation,
                padding=padding, data_format=data_format)(i)
    c2 = Conv2D(filters, shape, activation=activation,
                padding=padding, data_format=data_format)(c1)
    p = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                     data_format=data_format)(c2)
    return p, c2


def upconv_block(i, filters, shape, activation='relu', padding='same',
                 data_format='channels_first'):
    """Create an up-convolution block.

    The UpConv block is described as two 3x3 convolutions followed by a 2x2
    upsampling and a 2x2 convolution.

    Parameters
    ----------
    i : `keras.layers.Layer`
        Input layer for this upconv block.
    filters : int
        The number of filters for the first two convolutional layers in the
        block. The final layer has ``filters / 2`` layers.
    shape : int or tuple of int
        Shape of the convolutional filters.

    Other Parameters
    ----------------
    activation : str or `keras.layers.Activation`
        The activation function to apply to the layers.
    padding : {'same','valid'}
        The padding to apply to the convolutional layers.
    data_format : {'channels_first','channels_last'}
        Determines which dimension the convolutional filters are defined along.

    Returns
    -------
    upconv : `keras.layers.Conv2D`
    """
    c1 = Conv2D(filters, shape, activation=activation,
                padding=padding, data_format=data_format)(i)
    c2 = Conv2D(filters, shape, activation=activation,
                padding=padding, data_format=data_format)(c1)
    u = UpSampling2D(size=(2, 2), data_format=data_format)(c2)
    c3 = Conv2D(int(filters / 2),
                (2, 2),
                activation=activation,
                padding=padding,
                data_format=data_format)(u)
    return c3


def crop(larger, smaller):
    """Crop data for concatenation of bypass connections.

    Parameters
    ----------
    larger : keras.layers.Layer
        The larger layer to crop.
    smaller : keras.layers.Layer
        The smaller layer to use as reference.

    Returns
    -------
    cropped : `keras.layers.Cropping2D`
        A cropped version of ``larger`` if ``larger`` is not equal to
        ``smaller`` in size, otherwise ``larger``.
    """
    cs = float(larger._keras_shape[1] - smaller._keras_shape[1]) / 2.0
    if cs != 0:
        xy = (int(cs), int(cs))
        return Cropping2D(cropping=(xy, xy))(larger)
    else:
        return larger


def unet(input_shape, data_format='channels_first'):
    """Create a U-Net model using 2D operations.

    This model is based on the U-Net model described by Ronneberger *et al*
    [UNET]_ for solving membrane segmentation in electron microscopic images.

    Parameters
    ----------
    input_shape : tuple of int
        A 3-tuple with the dimensions ``(channels, height, width)``
    data_format : {'channels_first','channels_last'}
        Whether the channels dimension is at the start or end of the array.

    Returns
    -------
    unet : `keras.model.Model`

    Notes
    -----
    It is recommended to use ``'same'`` padding.

    References
    ----------
    .. [UNET] Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net:
       Convolutional networks for biomedical image segmentation. In
       International Conference on Medical image computing and computer-
       assisted intervention (pp. 234-241). Springer, Cham.
    """
    i = Input(shape=input_shape)

    down1, c1 = downconv_block(i, 16, 3)
    down2, c2 = downconv_block(down1, 32, 3)
    down3, c3 = downconv_block(down2, 64, 3)

    up1 = upconv_block(down3, 128, 3)
    crop1 = crop(c3, up1)
    merge_block = [up1, crop1]
    concat = concatenate(merge_block, axis=1)

    up2 = upconv_block(concat, 64, 3)
    crop2 = crop(c2, up2)
    merge_block = [up2, crop2]
    concat = concatenate(merge_block, axis=1)

    up3 = upconv_block(concat, 32, 3)
    crop3 = crop(c1, up3)
    merge_block = [up3, crop3]
    concat = concatenate(merge_block, axis=1)

    out1 = Conv2D(16, 3, activation='relu',
                  padding='same', data_format=data_format)(concat)
    out2 = Conv2D(16, 3, activation='relu',
                  padding='same', data_format=data_format)(out1)
    out3 = Conv2D(1, 1, activation='relu',
                  padding='same', data_format=data_format)(out2)

    model = Model(inputs=[i], outputs=[out3])

    return model


if __name__ == '__main__':
    model = unet((1, 256, 256))
