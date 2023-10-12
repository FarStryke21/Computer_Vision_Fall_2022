# November 7th
import cv2
import numpy as np
import argparse

# check size (bounding box) is square
def isSquare(siz):
    ratio = abs(siz[0] - siz[1]) / siz[0]
    #print (siz, ratio)
    if ratio < 0.1:
        return True
    else:
        return False

# check circle from the arc length ratio
def isCircle(cnt):
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    len = cv2.arcLength(cnt, True)
    ratio = abs(len - np.pi * 2.0 * radius) / (np.pi * 2.0 * radius)
    #print(ratio)
    if ratio < 0.1:
        return True
    else:
        return False

if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser(description='Hough Circles')
    parser.add_argument('-i', '--input', default = 'image/all-parts.png')

    args = parser.parse_args()
    # Read image
    img = cv2.imread(args.input)
    # Convert to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Binary
    thr,dst = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)

    # clean up
    # for i in range(1):
    #     dst = cv2.erode(dst, None)

    # Dilate and erode combination to seperate parts very close to each other
    dst = cv2.dilate(dst, None)
    dst = cv2.dilate(dst, None)
    dst = cv2.erode(dst, None)
    dst = cv2.dilate(dst, None)
    dst = cv2.dilate(dst, None)
    dst = cv2.erode(dst, None)

    # find contours with hierarchy
    cont, hier = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # img = cv2.drawContours(img, cont, -1, (0,0,255), 2)

    # each contour
    for i in range(len(cont)):
        c = cont[i]
        h = hier[0,i]
        if h[2] == -1 and h[3] == 0: #For Spade terminal
            # no child and parent is image outer
            img = cv2.drawContours(img, cont, i, (0,0,255),-1)  #Draws in red
        elif h[3] == 0 and hier[0,h[2]][2] == -1: # More Complex shapes
            # with child
            if isCircle(c): #  Washer and External lock washer
                if isCircle(cont[h[2]]): #Washer
                    # double circle
                    img = cv2.drawContours(img, cont, i, (0,255,0),-1)  #Draws in green
                #Add code for internal lock washer
                else:
                    img = cv2.drawContours(img, cont, i, (191,64,191), -1)
            else:   # Ring Terminal and internal lock washer
                # 1 child and shape bounding box is not square 
                if not isSquare(cv2.minAreaRect(c)[1]) and hier[0,h[2]][0] == -1 and hier[0,h[2]][1] == -1: #Ring Terminal
                    img = cv2.drawContours(img, cont, i, (255,0, 0),-1) #Draw in Blue
                # Add code for for external lock washer
                else:
                    img = cv2.drawContours(img, cont, i, (0,255,255), -1)
                    

    cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)

    cv2.imwrite("image/all-parts-output.png", img)
    cv2.waitKey()
