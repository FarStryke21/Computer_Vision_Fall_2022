# Sunday, 29th October, 2022
# Problem Set 5 - Computer Vision for Engineers

import cv2 as cv
import numpy as np

def erode_dilate(img, fname): #Erode and Dilate Functions
    k_e = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
    img = cv.erode(img, k_e)
    img = cv.erode(img, k_e)
    img = cv.dilate(img, k_e) 
    
    cv.imwrite(fname[0]+"-blob."+fname[1], img)
    return img

def cont_detect(img, fname):    #Contour detection and drawing
    icopy = img.copy()
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(img_gray, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) 
    for i in range(len(contours)):
        color = list(np.random.random(size = 3) * 255) #For random colors
        icopy = cv.drawContours(icopy, contours[i], -1, color,  2)
    
    cv.imwrite(fname[0]+"-contours."+fname[1], icopy)
    return icopy

def thresholding(img, fname):
    img = cv.copyMakeBorder(img, 3,3,3,3, cv.BORDER_CONSTANT, value=[255,255,255]) # For edge contours

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(img_gray, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) 
    threshold_blob = 2000
    
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv.contourArea(cnt)  
        if area <= threshold_blob:  #Thresholding
            thresh = cv.drawContours(thresh, [cnt], -1, 255, -1)    #Filling contour with white color
    thresh = thresh[3:-1-2, 3:-1-2] #Cropping to initial size
    cv.imwrite(fname[0]+"-thresholded."+fname[1], thresh)
    return thresh

def central_axis(img, fname):
    img = cv.bitwise_not(img)
    fimage = img.copy()
    k_e = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
    thin = np.zeros(img.shape, dtype='uint8')
    while(np.sum(fimage)!=0):
        cimg = cv.erode(fimage, k_e)
        oimg = cv.morphologyEx(cimg, cv.MORPH_OPEN, k_e)
        subset = cimg - oimg
        thin = cv.bitwise_or(subset, thin)  #union of previous data and new thiined features
        fimage = cimg.copy()
    thin = cv.bitwise_not(thin)
    cv.imwrite(fname[0]+"-cracks."+fname[1], thin)
    return thin

fname = input("Enter Image Path: ")
img = cv.imread(fname)
fname = fname.split('.')

erode_dilate_img = erode_dilate(img, fname)
cv.imshow("Eroded and Dilated", erode_dilate_img)
cv.waitKey(100)

cont_detect_img = cont_detect(erode_dilate_img, fname)
cv.imshow("Contour Detected Image", cont_detect_img)
cv.waitKey(100)

thresholded_img = thresholding(erode_dilate_img, fname)
cv.imshow("Thresholded Image", thresholded_img)
cv.waitKey(100)

crack_image = central_axis(thresholded_img,fname)
cv.imshow("Central Axis Image", crack_image)
cv.waitKey(0)

cv.destroyAllWindows()