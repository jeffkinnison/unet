import math

from keras.layers import Input, Conv3D, MaxPooling3D, Cropping3D, \
                         UpSampling3D, concatenate, ZeroPadding3D
from keras.models import Model
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.io import imread


def augmenting_generator(raw_data, gt_data, dists, input_shape, output_shape,
                         batch_size, ignore_distance=35.0, channel_idx=0):
    """Create a generator that augments data for training.
    This generator chooses random subsets of the supplied data and has a chance
    to randomly flip the images along one or more axes or scale/shift thup1e
    grayscale intensities.
    Parameters
    ----------
    raw_data : array-like
        Volume containing the raw images.
    gt_data : array-like
        Volume containing the ground truth.
    dists : array-like
        Volume containing the element-wise distance from each 0-element of
        ``gt`` to the nearest non-zero element.
    input_shape : tuple of int
        The shape of the data to feed into the neural network model.
    output_shape : tuple of int
        The expected shape of the neural network model output.
    batch_size : int
        The number of augmented images to include in a batch.
    ignore_distance : float, default 35.0
        The maximum average distance that augmented data is allowed to be from
        the ground truth. This prevents training on a subset of the data that
        contains no ground truth.
    channel_idx : int, default 0
        The index of the image channel dimension, typically first or last
        depending on the neural network framework.
    Yields
    ------
    aug : array-like
        The ``batch_size``-by-``input_shape`` augmented data to feed into a
        neural network.
    gt : array-like
        The ``output_shape`` ground truth subvolume associated with ``aug``.
    """
    # drop the channel idx
    input_shape = [d for i, d in enumerate(input_shape) if i != channel_idx]
    output_shape = [d for i, d in enumerate(output_shape) if i != channel_idx]
    raw_to_gt_offsets = [int((i - o) / 2)
                         for i, o in zip(input_shape, output_shape)]

    while True:
        batch = []
        for idx in range(batch_size):
            while True:
                lo_corner = [np.random.randint(rd - i) if rd - i > 0 else 0
                             for rd, i in zip(raw_data.shape, input_shape)]
                slices = [slice(l, l + i) for l, i in zip(lo_corner,
                                                          input_shape)]
                subraw = raw_data[tuple(slices)].astype(np.float32) / 255.0

                slices = [slice(l + o, l + o + i)
                          for l, i, o in zip(lo_corner, output_shape,
                                             raw_to_gt_offsets)]
                subgt = (gt_data[tuple(slices)] > 0).astype(np.float32)
                subdist = dists[tuple(slices)].astype(np.float32)
                subgt[(subdist <= ignore_distance) & (subgt == 0)] = 0.5

                # make sure we have enough positive pixels
                if subgt.mean() > 0.015:
                    break

            # flips
            if np.random.randint(2) == 1:
                subraw = subraw[::-1, :, :]
                subgt = subgt[::-1, :, :]
            if np.random.randint(2) == 1:
                subraw = subraw[:, ::-1, :]
                subgt = subgt[:, ::-1, :]
            if np.random.randint(2) == 1:
                subraw = subraw[:, :, ::-1]
                subgt = subgt[:, :, ::-1]
            if np.random.randint(2) == 1:
                subraw = np.transpose(subraw, [0, 2, 1])
                subgt = np.transpose(subgt, [0, 2, 1])

            # random scale/shift of intensities
            scale = np.random.uniform(0.8, 1.2)
            offset = np.random.uniform(-0.2, .2)
            subraw = subraw * scale + offset

            batch.append((subraw, subgt))

        subraws, subgts = zip(*batch)

        # after stack, channel index shifts over one
        yield (np.expand_dims(np.stack(subraws, axis=0), channel_idx + 1),
               np.expand_dims(np.stack(subgts, axis=0), channel_idx + 1))


def downconv_block(i, filters, shape, activation='relu', padding='same', data_format='channels_first'):
    c1 = Conv3D(filters, shape, activation=activation, padding=padding, data_format=data_format)(i)
    print(c1._keras_shape)
    c2 = Conv3D(filters, shape, activation=activation, padding=padding, data_format=data_format)(c1)
    print(c2._keras_shape)
    p = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), data_format=data_format)(c2)
    print(p._keras_shape)
    return p, c2


def upconv_block(i, filters, shape, activation='relu', padding='same', data_format='channels_first'):
    c1 = Conv3D(filters, shape, activation=activation, padding=padding, data_format=data_format)(i)
    print(c1._keras_shape)
    c2 = Conv3D(filters, shape, activation=activation, padding=padding, data_format=data_format)(c1)
    print(c2._keras_shape)
    u = UpSampling3D(size=(1, 2, 2), data_format=data_format)(c2)
    print(u._keras_shape)
    c3 = Conv3D(int(filters / 2),
                (1, 2, 2),
                activation=activation,
                padding=padding,
                data_format=data_format)(u)
    print(c3._keras_shape)
    return c3


def crop(larger, smaller):
    cs = float(larger._keras_shape[2] - smaller._keras_shape[2]) / 2.0
    csZ = float(larger._keras_shape[1] - smaller._keras_shape[1]) / 2.0
    if cs != 0:
        xy = (int(cs), int(cs))  # (int(math.floor(cs)), int(math.ceil(cs)))
        z = (int(csZ), int(csZ))  # (int(math.floor(csZ)), int(math.ceil(csZ)))
        return Cropping3D(cropping=(z, xy, xy))(larger)
    else:
        return larger


def unet3d(input_shape, data_format='channels_first'):
    i = Input(shape=input_shape)
    print(i._keras_shape)

    down1, c1 = downconv_block(i, 16, 3)
    down2, c2 = downconv_block(down1, 32, 3)
    down3, c3 = downconv_block(down2, 64, 3)

    up1 = upconv_block(down3, 128, 3)
    crop1 = crop(c3, up1)
    crop2 = crop(down2, up1)
    crop3 = crop(MaxPooling3D(pool_size=(1, 2, 2),
                              strides=(1, 2, 2),
                              data_format=data_format)(down1),
                 up1)
    print(crop1._keras_shape)
    print(crop2._keras_shape)
    print(crop3._keras_shape)
    merge_block = [up1, crop1, crop2, crop3]
    concat = concatenate(merge_block, axis=1)
    print(concat._keras_shape)

    up2 = upconv_block(concat, 64, 3)
    crop1 = crop(UpSampling3D(size=(1, 2, 2), data_format=data_format)(c3), up2)
    crop2 = crop(c2, up2)
    crop3 = crop(down1, up2)
    print(crop1._keras_shape)
    print(crop2._keras_shape)
    print(crop3._keras_shape)
    merge_block = [up2, crop2, crop1, crop3]
    concat = concatenate(merge_block, axis=1)
    print(concat._keras_shape)

    up3 = upconv_block(concat, 32, 3)
    crop1 = crop(
        UpSampling3D(size=(1, 2, 2), data_format=data_format)(
            UpSampling3D(size=(1, 2, 2), data_format=data_format)(c3)),
        up3)
    crop2 = crop(UpSampling3D(size=(1, 2, 2), data_format=data_format)(c2), up3)
    crop3 = crop(c1, up3)
    print(crop1._keras_shape)
    print(crop2._keras_shape)
    print(crop3._keras_shape)
    merge_block = [up3, crop3, crop2, crop1]
    concat = concatenate(merge_block, axis=1)
    print(concat._keras_shape)

    out1 = Conv3D(16, 3, activation='relu', padding='same', data_format=data_format)(concat)
    out2 = Conv3D(16, 3, activation='relu', padding='same', data_format=data_format)(out1)
    out3 = Conv3D(1, 1, activation='relu', padding='same', data_format=data_format)(out2)

    model = Model(inputs=[i], outputs=[out3])

    return model


if __name__ == '__main__':
    train_data = imread('/scratch0/isbi2012/train-volume.tif')
    train_gt = imread('/scratch0/isbi2012/train-labels.tif')
    dists = distance_transform_edt(train_gt == 0, (30, 4, 4))

    input_shape = (1, 30, 512, 512)
    output_shape = (1, 30, 512, 512)

    train_generator = augmenting_generator(
        train_data,
        train_gt,
        dists,
        input_shape=input_shape,
        output_shape=output_shape,
        batch_size=1,
        channel_idx=0)
    n = next(train_generator)
    print(n[0].shape)

    model = unet3d(input_shape)
    model.compile('sgd', 'binary_crossentropy')
    model.summary()
    model.fit_generator(train_generator, 1, epochs=1, verbose=2)
