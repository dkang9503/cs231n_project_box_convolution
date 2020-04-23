# Exploring Box Convolutional Layers

This repo contains code testing using box convolutional layers (see [paper](https://papers.nips.cc/paper/7859-deep-neural-networks-with-box-convolutions), [github](https://github.com/shrubb/box-convolutions)), which learn the size and location of a convolutional layer, in place of the usual 3x3.

This was a final project for CS231n at Stanford in Spring 2019.

Please see the [final report](https://github.com/dkang9503/cs231n_project_box_convolution/blob/master/Final_Report.pdf) for the full writeup of our findings.


## Authors

* **Harry Emeric** - [github](https://github.com/harryem)
* **David Kang** - [github](https://github.com/dkang9503)


## Getting Started

### Installing

To run the models in this repo you must install the Python package `box_convolution`:

```
python3 -m pip install git+https://github.com/shrubb/box-convolutions.git
python3 -m box_convolution.test # if throws errors, please open a GitHub issue
```

### Usage

```python
import torch
from box_convolution import BoxConv2d

box_conv = BoxConv2d(16, 8, 240, 320)
help(BoxConv2d)
```




`ssd/box_ssd.py`







### Data

- Tiny ImageNet: https://tiny-imagenet.herokuapp.com/
- Pascal VOC 2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
