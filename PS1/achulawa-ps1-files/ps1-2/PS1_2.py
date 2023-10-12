import cv2 as cv

def rgb2gray (img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # dst = cv.calcHist(img_gray, [0], None, [256], [0,256])
    # plt.hist(img_gray.ravel(),256,[0,256])
    # plt.title('Histogram for gray scale image')
    # plt.show()
    return img_gray
    
def grey2bin (img, op):
    if op:              # For highlighting bright parts
        th, out_img = cv.threshold(img, 70, 255, cv.THRESH_BINARY)
        return out_img
    else:               # For highlighting dark parts
        th, out_img = cv.threshold(img, 163, 255, cv.THRESH_BINARY_INV)
        return out_img

def main (img, name, op):  #Main Function
    cv.imshow('Input Color Image', img)
    img_gray = rgb2gray(img)            #Function to Convert color image to greyscale
    fname = name[0]+'_grayscale'+'.'+name[1]
    cv.imwrite(fname, img_gray)
    
    img_bin = grey2bin(img_gray, op)    #Function to Convert to binary map based on User choice
    fname = name[0]+'_binary'+'.'+name[1]
    cv.imwrite(fname, img_bin)
    
    rows = len(img_bin)
    cols = len(img_bin[0])
    
    img_out = img
    for x in range(0, rows):            # Converts white parts of the binary image red in the final image
        for y in range(0, cols):
            if img_bin[x][y] == 0 and op == 0:  #For highlighting dark parts
                img_out[x][y][0] = 0
                img_out[x][y][1] = 0
                img_out[x][y][2] = 255
            elif img_bin[x][y] == 255 and op == 1:  #For highlighting bright parts
                img_out[x][y][0] = 0
                img_out[x][y][1] = 0
                img_out[x][y][2] = 255
            
    fname = name[0]+'_output'+'.'+name[1]
    cv.imwrite(fname, img_out)

    cv.imshow('Grayscale Image', img_gray)
    cv.imshow('Binary Image', img_bin)
    cv.imshow('Output Color Image', img_out)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
filename = input("Enter File name : ")
img = cv.imread(filename)
filename = filename.split('.')
choice = input("Highlight Bright or Dark Portions? : ")
op = 0
if choice.lower() == "bright":
    op = 1
main(img, filename, op)