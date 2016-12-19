import numpy as np
import cv2
def show_image_with_point(img,points):
    img = cv2.resize(img,(150,150))
    for i in range(0,2,2):
        cv2.circle(img, ( int((float(points[i])+0)*10),int((float(points[i+1])+0)*10)) ,2, (255,0,0))
    cv2.namedWindow("image")
    cv2.imshow("image",img)
    cv2.waitKey(0)

if __name__ =="__main__":
    file_path = "./train/"
    file_list = "./train.list"
    fid = open(file_list)
    fid.readline()
    fid.readline()
    lines = fid.readlines()
    fid.close()
    for line in lines:
        items = line.split()
        file_name = file_path +  items[0]
        print "Checking: ",file_name
	points = np.zeros((2,))
        for i in range(2):
		points[i] =  round(float(items[i+1]))
        img = cv2.imread(file_name)
        show_image_with_point(img,points)
       
        
        
