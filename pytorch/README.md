# Pytorch U-Net Models

## Loading the Model

Add this directory to `PYTHONPATH`, then load in the models with the following
code snippets. The input shape does not have to be powers of two.

### U-Net

```python
from unet2d import UNet

# For grayscale data
m = unet(1)
```

### 3D U-Net

```python
from unet3d import UNet3D

# For grayscale data
m = unet3d(1)
```
