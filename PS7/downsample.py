import cv2

image = cv2.imread('achulawa-ps7-files/PS7-1/ps7-images/achulawa-right.png')
print("Size of image before pyrDown: ", image.shape)

image = cv2.pyrDown(image)
print("Size of image after pyrDown: ", image.shape)
cv2.imwrite('achulawa-ps7-files/PS7-1/ps7-images/achulawa-right-down.png', image)