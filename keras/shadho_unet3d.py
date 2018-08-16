import math

from keras.layers import Input, Conv3D, MaxPooling3D, Cropping3D, \
                         UpSampling3D, concatenate, ZeroPadding3D, \
                         Activation, LeakyReLU, PReLU, ThresholdedReLU
from keras.models import Model
from keras.optimizers import SGD


def get_optimizer(name, params=None):
    """Set up an optimizer by name.

    Parameters
    ----------
    name : str or dict
        The name of the optimizer. If a `dict`, must have the format
        `{<name>: <params>}`
    params : dict
        Paramters for instantiating the optimizer.

    Returns
    -------
    optimizer : str or `keras.optimizers.SGD`
        The optimizer to compile with a model.
    """
    if isinstance(name, dict):
        name, params = list(name.values())[0]

    if params is None:
        params = {}

    optimizers = {
        'sgd': SGD,
        'rmsprop': 'rmsprop',
        'adagrad': 'adagrad',
        'adadelta': 'adadelta',
        'adam': 'adam',
        'adamax': 'adamax',
        'nadam': 'nadam'
    }
    try:
        optimizer = optimizers[name]
    except KeyError:
        optimizer='adam'

    if optimizer is SGD:
        optimizer = optimizer(**params)

    return optimizer


def get_activation(name, params=None):
    """Set up an activation function by name.

    Parameters
    ----------
    name : str
        The name of the function. See the `activations` dictionary below.
    params : dict, optional
        Key/value pairs of activaiton function parameters. Only necessary for
        'leaky_relu', 'prelu', and 'thresholded_relu'.

    Returns
    -------
    activation
        The activation function, parameterized if necessary.
    """
    activations = {
        None: 'relu',
        'elu': 'elu',
        'hard_sigmoid': 'hard_sigmoid',
        'leaky_relu': LeakyReLU,
        'prelu': PReLU,
        'relu': 'relu',
        'sigmoid': 'sigmoid',
        'softmax': 'softmax',
        'softplus': 'softplus',
        'softsign': 'softsign',
        'tanh': 'tanh',
        'thresholded_relu': ThresholdedReLU
    }

    try:
        activation = activations[name]
    except KeyError:
        activation = 'relu'

    if isinstance(activation, str):
        activation = Activation(activation)
    else:
        activation = activation(**params)

    return activation


def downconv_block(i, filters, params=None, padding='same',
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
    downsample : `keras.layers.MaxPooling3D`
        The max pooling layer
    conv : `keras.layers.Conv3D`
        The final convolutional layer for use in a bypass connection.
    """
    # Set up convolutional layer 1
    cp = params['conv1']
    cp['activation'] = \
        get_activation(list(cp['activation'].keys())[0],
                       params=list(cp['activation'].values())[0])
    ks = cp.pop('kernel_size')
    c1 = Conv3D(filters, ks, padding=padding, data_format=data_format, **cp)(i)

    # Set up convolutional layer 2
    cp = params['conv2']
    cp['activation'] = \
        get_activation(list(cp['activation'].keys())[0],
                       params=list(cp['activation'].values())[0])
    ks = cp.pop('kernel_size')
    c2 = Conv3D(filters, ks, padding=padding, data_format=data_format, **cp)(c1)

    p = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                     data_format=data_format)(c2)
    return p, c2


def upconv_block(i, filters, params=None, padding='same',
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
    upconv : `keras.layers.Conv3D`
    """
    # Set up convolutional layer 1
    cp = params['conv1']
    cp['activation'] = \
        get_activation(list(cp['activation'].keys())[0],
                       params=list(cp['activation'].values())[0])
    ks = cp.pop('kernel_size')
    c1 = Conv3D(filters, ks, padding=padding, data_format=data_format, **cp)(i)

    # Set up convolutional layer 2
    cp = params['conv2']
    cp['activation'] = \
        get_activation(list(cp['activation'].keys())[0],
                       params=list(cp['activation'].values())[0])
    ks = cp.pop('kernel_size')
    c2 = Conv3D(filters, ks, padding=padding, data_format=data_format, **cp)(c1)

    # Set up up-conv
    u = UpSampling3D(size=(1, 2, 2), data_format=data_format)(c2)
    c3 = Conv3D(int(filters / 2),
                (1, 2, 2),
                activation='relu',
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
    cropped : `keras.layers.Cropping3D`
        A cropped version of ``larger`` if ``larger`` is not equal to
        ``smaller`` in size, otherwise ``larger``.
    """
    cs = float(larger._keras_shape[1] - smaller._keras_shape[1]) / 2.0
    if cs != 0:
        xy = (int(cs), int(cs))
        z = (int(csZ), int(csZ))  # (int(math.floor(csZ)), int(math.ceil(csZ)))
        return Cropping3D(cropping=(z, xy, xy))(larger)
    else:
        return larger


def unet3d(input_shape, data_format='channels_first', params=None):
    """Create a U-Net model using 3D operations.

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

    min_filters = params['min_filters']

    down1, c1 = downconv_block(i, min_filters, params=params['down1'])
    down2, c2 = downconv_block(down1, int(min_filters * 2),
                               params=params['down2'])
    down3, c3 = downconv_block(down2, int(min_filters * 4),
                               params=params['down3'])

    up1 = upconv_block(down3, int(min_filters * 8), params=params['up1'])
    crop1 = crop(c3, up1)
    merge_block = [up1, crop1]
    concat = concatenate(merge_block, axis=1)

    up2 = upconv_block(concat, int(min_filters * 4), params=params['up2'])
    crop2 = crop(c2, up2)
    merge_block = [up2, crop2]
    concat = concatenate(merge_block, axis=1)

    up3 = upconv_block(concat, int(min_filters * 2), params=params['up3'])
    crop3 = crop(c1, up3)
    merge_block = [up3, crop3]
    concat = concatenate(merge_block, axis=1)

    cp = params['out']['conv1']
    cp['activation'] = \
        get_activation(list(cp['activation'].keys())[0],
                       params=list(cp['activation'].values())[0])
    ks = cp.pop('kernel_size')
    out1 = Conv3D(min_filters, ks, padding='same', data_format=data_format,
                  **cp)(concat)

    cp = params['out']['conv2']
    cp['activation'] = \
        get_activation(list(cp['activation'].keys())[0],
                       params=list(cp['activation'].values())[0])
    ks = cp.pop('kernel_size')
    out2 = Conv3D(min_filters, ks, padding='same', data_format=data_format,
                  **cp)(out1)

    out3 = Conv3D(1, 1, activation='sigmoid', padding='same', data_format=data_format)(out2)

    model = Model(inputs=[i], outputs=[out3])

    return model


if __name__ == '__main__':
    # model = unet((1, 256, 256))

    params = {
        'optimizer': {'sgd': {'lr': 1e-3, 'momentum': 0.9}},
        'min_filters': 32,
        'down1': {
            'conv1': {'kernel_size': 3, 'activation': {None: None}},
            'conv2': {'kernel_size': 5, 'activation': {'relu': 'relu'}, 'kernel_initializer': 'zeros', 'bias_initializer': 'ones'}
        },
        'down2': {
            'conv1': {'kernel_size': 3, 'activation': {'elu': 'elu'}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None},
            'conv2': {'kernel_size': 5, 'activation': {'elu': 'elu'}, 'kernel_initializer': 'glorot_normal', 'bias_initializer': 'glorot_uniform'}
        },
        'down3': {
            'conv1': {'kernel_size': 3, 'activation': {'tanh': 'tanh'}},
            'conv2': {'kernel_size': 5, 'activation': {'sigmoid': 'sigmoid'}, 'kernel_initializer': 'zeros', 'bias_initializer': 'ones'}
        },
        'up1': {
            'conv1': {'kernel_size': 3, 'activation': {'prelu': {'alpha_initializer': 'he_normal', 'alpha_regularizer': 'l2', 'alpha_constraint': 'unit_norm'}}, 'kernel_constraint': 'non_neg', 'bias_constraint': 'unit_norm'},
            'conv2': {'kernel_size': 5, 'activation': {'thresholded_relu': {'theta': 0.75}}, 'kernel_initializer': 'zeros', 'bias_initializer': 'ones'}
        },
        'up2': {
            'conv1': {'kernel_size': 3, 'activation': {None: None}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None},
            'conv2': {'kernel_size': 5, 'activation': {None: None}, 'kernel_initializer': 'glorot_normal', 'bias_initializer': 'glorot_uniform'}
        },
        'up3': {
            'conv1': {'kernel_size': 3, 'activation': {None: None}},
            'conv2': {'kernel_size': 5, 'activation': {None: None}, 'kernel_initializer': 'zeros', 'bias_initializer': 'ones'}
        },
        'out': {
            'conv1': {'kernel_size': 3, 'activation': {None: None}, 'kernel_regularizer': 'l1', 'bias_regularizer': 'l2', 'activity_regularizer': 'l1_l2'},
            'conv2': {'kernel_size': 5, 'activation': {None: None}, 'kernel_initializer': 'random_normal', 'bias_initializer': 'random_uniform'}
        },
    }

    model = unet3d((1, 20, 256, 256), params=params)

