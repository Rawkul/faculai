---
output:
  pdf_document: default
  html_document: default
---
# FaculAI

This Python library is designed to detect solar faculae in solar images taken with the [Helioseismic and Magnetic Imager](http://hmi.stanford.edu/) (HMI) instrument on the Solar Dynamics Observatory. The library uses a [U-Net](https://arxiv.org/abs/1505.04597) deep learning model to detect the faculae and extracts useful information and statistics from the detected faculae, such as its area, position, etc.

## Installation

To install the library, simply run in terminal:
``` bash
pip install git+https://github.com/Rawkul/faculai.git
```

Or download the code and place yourself in the faculai main folder, then run in a terminal:

``` bash
pip install .
```

Other alternative way

### Requirements

-   Python version 3.10.x. Latest version may produce bugs.
-   TensorFlow **version 2.10**. Other versions may produce bugs loading convolution layers of the U-Net model.
-   Keras
-   SciPy
-   NumPy
-   Pandas

## Usage

To use FaculAI, import `DetectionModel` class and the function `get_table`

``` python
from faculai import DetectionModel, get_table

data = ... # Dictionary with ml, lat, lon, ...

model = DetectionModel()
table = get_table(model, data)
```

This will get you a pandas data frame stored in `table` variable. Type `help(get_table)` in a python terminal for details of the function, the input data, and the output table columns.

<p class="callout info">

The first time you load DetectionModel class will take some time since the system needs to load all TensorFlow and Keras necessary libraries into your CPU/GPU. Also, the first time you use the model in `get_table()` will take much longer since it will also load additional TensorFlow and Keras libraries . The rest of the time, it should work normally, lasting much less.

</p>

### Use other detection models

The line:

``` python
model = DetectionModel()
```

loads the default trained U-Net stored in the package folder as `"faculai/unet.h5"`. If you have the model stored in other folder, or other model you want to use, just load it as:

``` python
model = DetectionModel("path/to/your/model")
```

The model can be loaded either in `.h5` format or in Tensorflow SavedModel format. You are free to use your own keras model as long as it takes an input of shape `(numer_of_samples, 400, 380, 1)` and returns a segmentation mask of the same shape `(numer_of_samples, 400, 380, 1)`.

## License

FaculAI is released under the MIT License. See the [LICENSE](LICENSE) file for more information.
