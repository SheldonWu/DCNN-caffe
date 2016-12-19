caffe_root = '/home/cmcc/caffe/python'
import sys
import os
sys.path.insert(0, caffe_root)
import caffe
caffe.set_mode_gpu()
import cv2
import numpy as np
import timeit
caffe_net_root = "/home/cmcc/Desktop/DCNNCascade_pts/train_img/processed_img/"
caffe_model_level1 = "/model/_iter_1000000.caffemodel"
#39 * 39
net_F1 = caffe.Net( caffe_net_root + "deploy_net1.prototxt",
                    caffe_net_root + "F1" + caffe_model_level1,
                    caffe.TEST)
#39 * 31
net_EN1 = caffe.Net( caffe_net_root + "deploy_net2.prototxt",
                    caffe_net_root + "EN1" + caffe_model_level1,
                    caffe.TEST)
net_NM1 = caffe.Net( caffe_net_root + "deploy_net2.prototxt",
                    caffe_net_root + "NM1" + caffe_model_level1,
                    caffe.TEST)
# find the face from the image (lt_x,lt_y)(rd_x,rd_y)
def cropface(img,lt_x,lt_y,rd_x,rd_y):
    mid_x = (lt_x + rd_x)/2
    mid_y = (lt_y + rd_y)/2
    if rd_x - lt_x > rd_y - lt_y:
        side = rd_x - lt_x
    else:
        side = rd_y - lt_y
    lt_x = mid_x - side/2.0
    lt_y = mid_y - side/2.0
    rd_x = mid_x + side/2.0
    rd_y = mid_y + side/2.0
    F1 = img[lt_y:rd_y,lt_x:rd_x]
    F1 = cv2.resize(F1,(39,39))
    EN1 = F1[0:31,0:39]
    NM1 = F1[8:39,0:39]
    return F1,EN1,NM1,side
def cropfaceOrgan(img,pts,rate):
    mid_x = float(pts[0])
    mid_y = float(pts[1])
    lt_x = mid_x - 39*rate
    lt_y = mid_y - 39*rate
    rd_x = mid_x + 39*rate
    rd_y = mid_y + 39*rate
    img = img[lt_y:rd_y,lt_x:rd_x]
    img = cv2.resize(img,(15,15))
    return img
#this net is designed for F1
def net1(image,net):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    net.blobs['data'].data[...] = transformer.preprocess('data', image)
    scores = net.forward()
    pts1 = (float(scores['ip2'][0][0]),float(scores['ip2'][0][1]))
    pts2 = (float(scores['ip2'][0][2]),float(scores['ip2'][0][3]))
    pts3 = (float(scores['ip2'][0][4]),float(scores['ip2'][0][5]))
    pts4 = (float(scores['ip2'][0][6]),float(scores['ip2'][0][7]))
    pts5 = (float(scores['ip2'][0][8]),float(scores['ip2'][0][9]))
    return pts1,pts2,pts3,pts4,pts5
#this net is designed for NM1 and EN1
def net2(image,net):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    net.blobs['data'].data[...] = transformer.preprocess('data', image)
    scores = net.forward()
    pts1 = (float(scores['ip2'][0][0]),float(scores['ip2'][0][1]))
    pts2 = (float(scores['ip2'][0][2]),float(scores['ip2'][0][3]))
    pts3 = (float(scores['ip2'][0][4]),float(scores['ip2'][0][5]))
    return pts1,pts2,pts3


def calFeaturePoint(img,lt_x,lt_y,rd_x,rd_y,rate):
     #set for Level1 input
    F1, EN1, NM1, side = cropface(img,lt_x,lt_y,rd_x,rd_y)
    
    F1_pts1,F1_pts2,F1_pts3,F1_pts4,F1_pts5 = net1(F1,net_F1)
    EN1_pts1,EN1_pts2,EN1_pts3              = net2(EN1,net_EN1)
    NM1_pts3,NM1_pts4,NM1_pts5              = net2(NM1,net_NM1)
    NM1_pts3 = (NM1_pts3[0], NM1_pts3[1] + 8.0)
    NM1_pts4 = (NM1_pts4[0], NM1_pts4[1] + 8.0)
    NM1_pts5 = (NM1_pts5[0], NM1_pts5[1] + 8.0)
    #set for Level2 input
    pts1 = ((EN1_pts1[0] + F1_pts1[0])/2.0,(EN1_pts1[1] + F1_pts1[1])/2.0)
    pts2 = ((EN1_pts2[0] + F1_pts2[0])/2.0,(EN1_pts2[1] + F1_pts2[1])/2.0)
    pts3 = ((F1_pts3[0] + NM1_pts3[0] + EN1_pts3[0])/3.0,
            (F1_pts3[1] + NM1_pts3[1] + EN1_pts3[1])/3.0)
    pts4 = ((NM1_pts4[0] + F1_pts4[0])/2.0,(NM1_pts4[1] + F1_pts4[1])/2.0)
    pts5 = ((NM1_pts5[0] + F1_pts5[0])/2.0,(NM1_pts5[1] + F1_pts5[1])/2.0)
    LM21 = cropfaceOrgan(F1,pts4,rate) 
    return pts4,side,LM21 * 255
   
if __name__ == "__main__":
    os.system("cd train; rm *.*; cd ..")
    os.system("cd val; rm *.*; cd ..")
    os.system("rm train.list; rm val.list;")
    filelist = "/home/cmcc/Desktop/DCNNCascade_pts/train_img/org_data_img/trainImageList.txt"
    imagepath = "/home/cmcc/Desktop/DCNNCascade_pts/train_img/org_data_img/"
    
    savepath = "./train/"
    fileout = "./train.list"

    fid = open(filelist,'r')
    lines = fid.readlines()
    fid.close()
    fid = open(fileout,'w')
    count = 1
    rate = 0.16
    for line in lines:
        words = line.split()
        filename = imagepath + words[0]
        print filename
        img = cv2.imread(filename)/255.0
        lt_x = float(words[1])
        rd_x = float(words[2])
        lt_y = float(words[3])
        rd_y = float(words[4])

        point,side,cropImg = calFeaturePoint(img,lt_x,lt_y,rd_x,rd_y,rate)
	
        point_x = ((float(words[11])-lt_x) * (39.0/side) - (point[0] - 39 * rate)) * (15 / (39 * rate * 2))
        point_y = ((float(words[12])-lt_y) * (39.0/side) - (point[1] - 39 * rate)) * (15 / (39 * rate * 2))
        fid.write(str(count)+'.jpg')
        fid.write('\t' + str(point_x))
        fid.write('\t' + str(point_y) + '\n')
        img = cv2.imread(filename)
        cv2.imwrite(savepath + str(count) + ".jpg",cropImg)
	
        count += 1
   
    filelist = "/home/cmcc/Desktop/DCNNCascade_pts/train_img/org_data_img/testImageList.txt"
    savepath = "./val/"
    fileout = "./test.list"
    fid.close()
    fid = open(filelist,'r')
    lines = fid.readlines()
    fid.close()
    fid = open(fileout,'w')
    count = 1
    for line in lines:
        words = line.split()
        filename = imagepath + words[0]
        print filename
        img = cv2.imread(filename)/255.0
        lt_x = float(words[1])
        rd_x = float(words[2])
        lt_y = float(words[3])
        rd_y = float(words[4])

        point,side,cropImg = calFeaturePoint(img,lt_x,lt_y,rd_x,rd_y,rate)
	
        point_x = ((float(words[11])-lt_x) * (39.0/side) - (point[0] - 39 * rate)) * (15 / (39 * rate * 2))
        point_y = ((float(words[12])-lt_y) * (39.0/side) - (point[1] - 39 * rate)) * (15 / (39 * rate * 2))
        fid.write(str(count)+'.jpg')
        fid.write('\t' + str(point_x))
        fid.write('\t' + str(point_y) + '\n')
        img = cv2.imread(filename)
        cv2.imwrite(savepath + str(count) + ".jpg",cropImg)
	
        count += 1

    
