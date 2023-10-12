from ast import Lambda
import numpy as np
import cv2 as cv
from tkinter import *

def sobel (img,f):
    Gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    sobel_img = np.zeros(shape=img.shape, dtype="uint8")
    img = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT)
    for i in range(3,img.shape[0] - 2):
        for j in range(3,img.shape[1] - 2):
            x = np.sum(Gx * img[i-1:i+2 , j-1:j+2])/4  # horizontal
            y = np.sum(Gy * img[i-1:i+2 , j-1:j+2])/4  # vertical
            sobel_img[i,j] = np.sqrt(x**2 + y**2)
    sobel_img = cv.bitwise_not(sobel_img)
    cv.imshow("Sobel Filtered Image", sobel_img/np.amax(sobel_img))
    cv.waitKey(0)
    cv.destroyAllWindows()
    f = f.split('.')
    cv.imwrite(f[0]+"-sobel."+f[1], sobel_img, )

def GUI(img,f):
    root = Tk()
    root.title("Control the Canny Edge Filter")
    root.geometry("500x500")
    s1 = IntVar()
    s2 = IntVar()
    s1.set(50)
    s2.set(200)
    l2 = BooleanVar()
    l2.set(False)
    apert = IntVar()
    apert.set(3)
    Scale(root, from_=0, to=255, label="Threshold 1", length=350 , variable=s1, orient=HORIZONTAL).pack()
    Scale(root, from_=0, to=255, label="Threshold 2", length=350 ,variable=s2, orient=HORIZONTAL).pack(pady=10)
    Label(root, text="L2 Gradient").pack(pady=10)
    Radiobutton(root, text = "True", variable=l2, value=True).pack()
    Radiobutton(root, text = "False", variable=l2, value=False).pack()
    Label(root, text="Aperture Size").pack(pady=10)
    apertmenu  = OptionMenu(root, apert, 3,5,7)
    apertmenu.pack()
    execute = Button(root,text="Run", command= lambda : canny(img, s1.get(), s2.get(), apert.get(), l2.get(),f))
    execute.pack(pady=10)
    vals = Text(root)
    valP = Button(root, text="Print Current Values", command = lambda: vals.replace("1.0" ,"end",f"Current Vals\nThreshold 1 = {s1.get()}\nThreshold 2 = {s2.get()}\nAperture Size = {apert.get()}\nL2 Gradient = {l2.get()} " ))
    valP.pack(pady=10)
    vals.pack()
    root.mainloop()

def canny(img,th1,th2,apert,l2,f):
    cv.destroyAllWindows()
    dst = cv.Canny(img, th1, th2, apertureSize=apert, L2gradient=l2)
    dst = cv.bitwise_not(dst)
    cv.imshow("Canny Filtered Image", dst)
    cv.waitKey(100)
    f = f.split('.')
    cv.imwrite(f[0]+"-canny."+f[1], dst)

f = input("Enter filename : ")
img = cv.imread(f)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
pad_img = np.pad(img, [(2,2),(2,2)], mode='constant', constant_values=0)
ch = input(f"Sobel or Canny?\n")
if ch.lower() == "sobel":
    sobel(img,f)
elif ch.lower() == "canny":
    GUI(pad_img,f)
else:
    print("Invalid Input")
