from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from sklearn.svm import SVC
# from sklearn.externals import joblib
import joblib
from skimage import color
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import cv2
import os
import glob

#Define HOG Parameters
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3

# define the sliding window:
def sliding_window(image, stepSize, windowSize):# image is the input, step size is the number of pixels needed to skip and windowSize is the size of the display window
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):# this for loop defines the sliding part and loops over the x and y coordinates
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y: y + windowSize[1], x:x + windowSize[0]])

# Uncomment each line depending on the model to use
model = joblib.load('./model_linear_kernel.npy') # Path to saved model created with linear kernel
# model = joblib.load('./model_poly_kernel.npy') # Path to saved model created with poly kernel
# model = joblib.load('./model_rbf_kernel.npy') # Path to saved model created with rbf kernel
# model = joblib.load('./model_sigmoid_kernel.npy') # Path to saved model created with sigmoid kernel

# Test the trained classifier on an image below
scale = 0
detections = []

# Uncomment each line depending on which image to test this on
path = 'ps8b-test-dataset/beetles.png'
# path = 'ps8b-test-dataset/football_field.jpg'
# path = 'ps8b-test-dataset/person_in_the_woods.png'

img = cv2.imread(path)
image_name = path.split('/')
name = image_name[-1].split('.')
# you can image if the image is too big
img= cv2.resize(img,(300,200)) # can change the size to default by commenting this code out our put in a random number

# defining the size of the sliding window (has to be the same as the size of the image in the training data)
(winW, winH)= (64,128)
windowSize=(winW,winH)
downscale=1.5

# Apply sliding window: Do not change this code!
for resized in pyramid_gaussian(img, max_layer = 0, downscale=1.5): 
    for (x,y,window) in sliding_window(resized, stepSize=10, windowSize=(winW,winH)):
        if window.shape[0] != winH or window.shape[1] !=winW: 
            continue
        window = color.rgb2gray(window)
        # Extract HOG features from the window captured, and predict whether it is a person or not
        fds = hog(window, orientations = orientations, pixels_per_cell= pixels_per_cell, cells_per_block= cells_per_block, block_norm='L2',
             feature_vector=True)
        #print(fds.shape)
        fds = fds.reshape(1,-1)
        pred = model.predict(fds)

        if pred == 1:
            #print('confirm')
            if model.decision_function(fds) > 0.6:  # set a threshold value for the SVM prediction i.e. only firm the predictions above probability of 0.6
                print("Detection:: Location -> ({}, {})".format(x, y))
                print("Scale ->  {} | Confidence Score {} \n".format(scale,model.decision_function(fds)))

                detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), model.decision_function(fds),
                                   int(windowSize[0]*(downscale**scale)), # create a list of all the predictions found
                                      int(windowSize[1]*(downscale**scale))))
    scale+=1


clone = resized.copy()
for (x_tl, y_tl, _, w, h) in detections:
    cv2.rectangle(img, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 255), thickness = 2)
rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections]) # do nms on the detected bounding boxes

sc = [score[0] for (x, y, score, w, h) in detections]
print("Detection confidence score: ", sc)
sc = np.array(sc)

# Non-maximal suppresion
pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)

for (xA, yA, xB, yB) in pick:
    cv2.rectangle(img, (xA, yA), (xB, yB), (0,255,0), 2)
    
cv2.imshow("Detections after NMS", img)
cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()
cv2.imwrite(f'result_images/{name[0]}-output.{name[1]}',img)

