# Keras U-Net Models

## Backend

We recommend using the Tensorflow backend for these models because Theano is no
longer in active development.

## Loading the Model

Add this directory to `PYTHONPATH`, then load in the models with the following
code snippets. The input shape does not have to be powers of two.

### U-Net

```python
from unet2d import unet

# For 256x256px grayscale data
m = unet((1, 256, 256))
```

### 3D U-Net

```python
from unet3d import unet3d

# For 32x256x256px grayscale data
m = unet3d((1, 32, 256, 256))
```
