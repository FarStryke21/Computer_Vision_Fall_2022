import cv2 as cv
import numpy as np

def gammacorrection(img, gamma):    #main function
    (B,G,R) = cv.split(img)     #Split image into individual channels
    B = np.array(255*(B/255)**(1/gamma), dtype='uint8') #Applying correction formula on each channel
    G = np.array(255*(G/255)**(1/gamma), dtype='uint8')
    R = np.array(255*(R/255)**(1/gamma), dtype='uint8')
    img_out = cv.merge([B,G,R]) #merge channels
    return img_out

name = input("Enter file name = ")
img = cv.imread(name)
g = input("Enter Gamma value = ")
while(g != 'x'):
    cv.destroyAllWindows()      #Destroy all present windows
    nimg = gammacorrection(img, float(g))   #Gamma corrected image
    cv.imshow('Original Image', img)    #Display image
    cv.imshow('Edited Image', nimg)     #Display corrected image
    cv.waitKey(1000)                    #Stop windows from closing
    g = input("Enter Gamma value or press x to save image = ")  #new gamma values

name = name.split('.')
fname = name[0]+'_gcorrected.'+name[1]
cv.imwrite(fname, nimg) #save file
cv.destroyAllWindows()