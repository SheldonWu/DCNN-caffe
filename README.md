# DCNN-caffe
## Introduction
This project is related to CVPR2013: [Deep Convolutional Network Cascade for Facial Point Detection](http://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Sun_Deep_Convolutional_Network_2013_CVPR_paper.pdf). In this paper, it focus on the structural design of individual networks and their combining strategies. 
Figure 2 is an overview of our approach. There are five facial points to be detected: left eye center (LE), right eye center (RE), nose tip (N), left mouth corner (LM), and right mouth corner (R- M). We cascade three levels of convolutional networks to make coarse-to-fine prediction. At the first level, we em- ploy three deep convolutional networks, F1, EN1, and N- M1, whose input regions cover the whole face (F1), eyes and nose (EN1), nose and mouth (NM1). Each network si
Have Fun!!!
## Dependencies
* [Caffe](http://caffe.berkeleyvision.org)
* [python](https://www.python.org)
* [opencv](http://opencv.org)

