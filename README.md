# FaculAI

![](https://img.shields.io/badge/version-2.0.1-green)

This Python library is designed to detect solar faculae in solar images taken with the [Helioseismic and Magnetic Imager](http://hmi.stanford.edu/) (HMI) instrument on the Solar Dynamics Observatory (SDO). The library uses a [U-Net](https://arxiv.org/abs/1505.04597) deep learning model to detect the faculae and extracts useful information and statistics from the detected faculae, such as their areas, positions, $B_{LOS}$, etc.

## Installation

To install the library, simply run in terminal:

``` bash
pip install git+https://github.com/rawkul/faculai.git
```

Alternatively, you can download the code and then run in a terminal:

``` bash
pip install path/to/faculai/code
```

Replace `path/to/faculai/code` with the actual code directory path.

### Requirements

-   Python version >= 3.10.x.
-   TensorFlow version >= 2.11.1, <2.12  (Higher versions may produce bugs)
-   Keras
-   SciPy
-   NumPy
-   Pandas

## Usage

To use FaculAI, import `DetectionModel` class and the function `get_tables`

``` python
from faculai import DetectionModel, get_tables

data = ... # Dictionary with ml, lat, lon, ...

model = DetectionModel()
ftable, ntable = get_tables(model, data)
```

This will get you 2 pandas data frames stored in `ftable` and `ntable` variables. Type `help(get_tables)` in a python terminal for details of the function, the input data, and the output table columns.

`ftable` contains relevant parameters for all the faculae found in the images.
`ntable` contains pinformation for regions with no faculae.

<p class="callout info">

The first time you load DetectionModel class will take some time since the system needs to load all TensorFlow and Keras necessary libraries into your CPU/GPU. Also, the first time you use the model in `get_tables()` will take much longer since it will also load additional TensorFlow and Keras libraries . The rest of the time, it should work normally, lasting much less.

</p>

> **❗Note:** At the moment, this package is designed exclusively for polar faculae, 
that is, faculae at latitudes $\varphi\ge|60^\circ|$ (Quiet Sun regions). However, you can use it in other 
lower latitudes, but the U-Net is not designed to distinguish between other types of 
magnetic structures, so it may detect other bright structures as faculae if you provide 
different regions. For example, sunspots appear bright in linear polarization images, 
which are the images used for detecting polar faculae.

### Use other detection models

The line:

``` python
model = DetectionModel()
```

loads the default trained U-Net stored in the package folder as `"faculai/unet.h5"`. If you have the model stored in other folder, or other model you want to use, just load it as:

``` python
model = DetectionModel("path/to/your/model")
```

The model can be loaded either in `.h5` format or in Tensorflow SavedModel format. You are free to use your own keras model as long as it takes an input of shape `(numer_of_samples, x, y, 1)` and returns a segmentation mask of the same shape `(numer_of_samples, x, y, 1)`. `x` and `y` are arbitrary image dimensions.

## License

FaculAI is released under the MIT License. See the [LICENSE](LICENSE) file for more information.
