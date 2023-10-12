import numpy as np
import cv2 as cv

f = input("Filename : ")
img = cv.imread(f)
f = f.split('.')
ch = "Choice"
func_list = []

while ch.lower() != "save":
    ch = input("Enter filter : ")
    cv.destroyAllWindows()
    cv.imshow("Original Image", img)
    cv.waitKey(100)
    if ch.lower() == "blur":
        k = int(input("Kernel Size : "))
        img = cv.blur(img, (k,k)) 
        func_list.append(ch+" ( kernel size = "+str(k)+" )")
        # method is used to blur an image using the normalized box filter
    elif ch.lower() == "gaussian blur":
        k = int(input("Kernel Size : "))
        img = cv.GaussianBlur(img, (k,k), cv.BORDER_DEFAULT) 
        func_list.append(ch+" ( kernel size = "+str(k)+" )")
        #any sharp edges in images are smoothed while minimizing too much blurring.
    elif ch.lower() == "median blur":
        k = int(input("Kernel Size : "))
        img = cv.medianBlur(img, k)
        func_list.append(ch+" ( kernel size = "+str(k)+" )")
        #Takes median of all pixels under kernel area. Effective in reducing salt-and-pepper noise.
    elif ch.lower() == "bilateral":
        d1 = int(input("Kernel Dia : "))
        img = cv.bilateralFilter(img, d1, 75, 75)
        func_list.append(ch+" ( kernel size = "+str(d1)+" )")
        #A bilateral filter is used for smoothening images and reducing noise, while preserving edges.
    elif ch.lower() == "sharpen":
        K = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        img = cv.filter2D(img, ddepth=-1, kernel=K)
        func_list.append(ch)
        #convolves an image with the kernel
    elif ch.lower() == "save":
        cv.imwrite(f[0]+"-improved."+f[1], img)
        func_list.append(ch)
    elif ch.lower() == "quit":
        break
    else:
        print("No such options!")
        continue
    cv.imshow(ch+" applied", img)
    cv.waitKey(100)

print("Filters Applied : ")
print(func_list)
cv.destroyAllWindows()