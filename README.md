# DCNN-caffe
## Introduction
This project is related to CVPR2013: Deep Convolutional Network Cascade for Facial Point Detection. In this paper, it contains 23 Convolution Network in 3 levels. It is a Deep Network which cascade 23 network can detect 5 points. The cascade structure makes this net faster with high accuracy than original VGG net.
In the first level, it contains 3 net: F1, EN1 and NM1.
The second level and the third level each contains 10 nets(2 for each point).
The project is based on the Caffe. Tools are written in Python. Train and Test Data can be found in the Data List.
Have Fun!!!
## Dependencies
* [Caffe](http://caffe.berkeleyvision.org)
* [python](https://www.python.org)
* [opencv](http://opencv.org)

