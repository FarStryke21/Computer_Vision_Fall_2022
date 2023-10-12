# November 7th
import cv2
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hough Circles')
    parser.add_argument('-i', '--input', default = 'image/spade-terminal.png')

    args = parser.parse_args()
    # Read image
    img = cv2.imread(args.input)
    # Convert to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Binary
    thr,dst = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    # D-E-E-D Routine for image cleaning
    dst = cv2.dilate(dst, None) 
    dst = cv2.erode(dst, None)
    dst = cv2.erode(dst, None)
    dst = cv2.dilate(dst, None)

    # find contours with hierarchy
    cont, hier = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    defects_id = []
    elements = len(cont) # Number of elements in the image
    uncertaininty = int(0.2 * elements) # Uncertaininty of elemnts being similar
    for id in range(len(cont)):
        contour = cont[id]
        h = hier[0,id]
        if h[2] == -1 and h[3] == 0:
            match = 0   # NUmber of matches for each element
            for i in range(len(cont)):
                compare = cont[i]
                if cv2.matchShapes(contour,compare,cv2.CONTOURS_MATCH_I2,0) < 1.4:
                    match+=1
             # Elements are similar if and only if they match most of the other elements
            if match <= elements - uncertaininty:
                defects_id.append(id)

    print(defects_id)
    for id in defects_id:
        img = cv2.drawContours(img, cont, id, (0,0,255), -1)
   
    cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)

    cv2.imwrite("image/spade-terminal-output.png", img)
    cv2.waitKey()
