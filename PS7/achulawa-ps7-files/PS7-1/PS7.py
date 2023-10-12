import cv2
import numpy as np
from matplotlib import pyplot as plt

def createPlyFile(data,file):
    # Opening the file
    f = open(f"ps7-ply/{file}.ply","w")
    # Creating the header
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write(f"element vertex {data.shape[0]}\n")
    f.write(f"property float32 x\n")
    f.write(f"property float32 y\n")
    f.write(f"property float32 z\n")
    f.write(f"property uint8 red\n")
    f.write(f"property uint8 green\n")
    f.write(f"property uint8 blue\n")
    f.write("end_header\n")
    # Adding the data
    for i in data:
        f.write(f"{i[0]} {i[1]} {i[2]} {i[3]} {i[4]} {i[5]}\n")
    # CLose file
    f.close()

file = input("Enter file id : ")
img_L = cv2.imread("ps7-images/"+file+"-left.png")
img_l = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)

img_R = cv2.imread("ps7-images/"+file+"-right.png")
img_r = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)

# Creating an object of StereoBM algorithm
'''In block matching or cv2.StereoBM_create() the disparity is computed by comparing the sum of absolute 
differences (SAD) of each 'block' of pixels. In semi-global block matching or cv2.StereoSGBM_create() 
forces similar disparity on neighbouring blocks. This creates a more complete disparity map but is more 
computationally expensive.
'''

stereo = cv2.StereoSGBM_create(minDisparity=38, numDisparities=60, blockSize = 30, P1=60, P2=100,
 speckleRange=6, speckleWindowSize=250, disp12MaxDiff=30) 
# stereo = cv2.StereoSGBM_create(minDisparity=40, numDisparities=60, blockSize = 12) 
k = 0.65
disparity = k*stereo.compute(img_l,img_r)

cv2.imshow('Disparity Map', np.uint8(disparity))
cv2.waitKey(0)
cv2.imwrite("ps7-images/"+file+"-disparity.png",np.uint8(disparity))
'''
# SPLIT INTO THREE CHANNELS. X AND Y POSITION OF ANY PIXEL. KxDisparity IS Z. RGB FROM THREE CHANNELS
'''
(B,G,R) = cv2.split(img_L)
data = np.array([[i,j,int(disparity[i,j]),R[i,j],G[i,j],B[i,j]] for j in range(38+60,img_L.shape[1]) for i in range(img_L.shape[0])])

createPlyFile(data, file)

cv2.destroyAllWindows()