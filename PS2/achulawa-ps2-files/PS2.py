#September 19th, 2022
#Aman Chulawala
#24678 PS2 Q1

from math import ceil
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np

#Convert input image to gray scale
def color2gray(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    hist_gen(gray)
    return gray

#Generate Lookup array for the intensity range specified in the image
def lookup_gen(img):
    mini = np.amin(img)
    maxi = np.amax(img)
    q1 = mini + ceil((maxi-mini)/4)
    q2 = mini + ceil((maxi-mini)/2)
    q3 = maxi - ceil((maxi-mini)/4)
    lookup = np.zeros((3,256), dtype = 'uint8')
    #Generating Blue channel
    lookup[0,mini:q1] = 255
    lookup[0, q2:] = 0
    lookup[0, q1:q2] = np.linspace(255,0,num = q2-q1)
    #Generating Green Channel
    lookup[1,mini:q1] = np.linspace(0,255,num = q1-mini)
    lookup[1,q1:q3] = 255
    lookup[1, q3:maxi] = np.linspace(255,0, num = maxi-q3)
    #Generating Red Channel
    lookup[2, 0:q2] = 0
    lookup[2, q2:q3] = np.linspace(0,255, num = q3-q2)
    lookup[2, q3:] = 255
    #Plotting lookuptable
    plt.figure(2)
    plt.title('Tone Curve')
    plt.plot(np.linspace(mini,255, num = len(lookup[0,mini:])), lookup[0,mini:], label = "Blue", color = 'blue')
    plt.plot(np.linspace(0,255, num = 256), lookup[1,:], label = "Green", color = 'green')
    plt.plot(np.linspace(0,255, num = 256), lookup[2,:], label = "Red", color = 'red')
    plt.legend()
    plt.show()
    return lookup, mini, maxi

#Plotting Histogram of intensities for reference
def hist_gen(img):
    histSize = 256
    histRange = (0, 256)
    hist_data = cv.calcHist(img, [0], None, [histSize], histRange, accumulate=False)
    plt.figure(1)
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist_data)
    plt.xlim([0, 256])

#Convert image to pseudocolored based on the lookup array   
def convert(img, lookup):
    b_channel = [[lookup[0,img[i,j]] for j in range(0,len(img[0]))] for i in range(0,len(img))]
    g_channel = [[lookup[1,img[i,j]] for j in range(0,len(img[0]))] for i in range(0,len(img))]
    r_channel = [[lookup[2,img[i,j]] for j in range(0,len(img[0]))] for i in range(0,len(img))]
    pseudocolored = cv.merge(np.array([b_channel, g_channel, r_channel]))
    return pseudocolored

#Draw circle and cross hairs on the image
def draw(img, draw_img, mini, maxi):
    means = np.where(img == maxi)
    y_mean = round(np.mean(means[0]))
    x_mean = round(np.mean(means[1]))
    cv.circle(draw_img, (x_mean, y_mean), 20, (255,255,255), 2)
    cv.line(draw_img, (x_mean-30, y_mean), (x_mean+30, y_mean), (255,255,255), 2)
    cv.line(draw_img, (x_mean, y_mean-30), (x_mean, y_mean+30), (255,255,255), 2)
    return draw_img

#Main function
def main(img):
    gray = color2gray(img)
    lookup, mini, maxi = lookup_gen(gray)
    pseudocolored = convert(gray, lookup)
    final_img = draw(gray, pseudocolored, mini, maxi)
    return final_img

#Input filename
fname = input("Enter Filename: ")
img = cv.imread(fname)
cv.imshow('Original Image', img)
final = main(img)

#Display final image
cv.imshow('Pseudocolored Image', final)
cv.waitKey(0)
cv.destroyAllWindows

#Save final image
name = fname.split('.')
n_name = name[0]+'-color.'+name[1]
cv.imwrite(n_name, final)