# Author Anderson Steckler
# Slider Code Adapted from https://www.timpoulsen.com/2018/getting-user-input-with-opencv-trackbars.html

import cv2
import numpy as np

fimg = None

circle_params = {
   'p1': 100,
   'p2': 20,
   'dpi': 1,
   'minD': 20,
   'Rmin': 5,
   'Rmax': 20
}

def main():
    global circle_params
    global fimg

    img = cv2.imread(r"FreqImgAnalysis\FFT.jpg")
    fimg = filter(img)

    cv2.namedWindow('frame')
    cv2.createTrackbar('P1', 'frame', 0, 300, p1_change)
    cv2.createTrackbar('P2', 'frame', 0, 100, p2_change)
    #cv2.createTrackbar('dpi', 'frame', 1, 10, dpi_change)
    #cv2.createTrackbar('minD', 'frame', 1, 300, minD_change)
    #cv2.createTrackbar('Rmin', 'frame', 1, 100, Rmin_change)
    #cv2.createTrackbar('Rmax', 'frame', 1, 100, Rmax_change)
    cv2.imshow('frame', fimg)

    circle()

    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            exit()

def filter(img):
    # Crop
    width = img.shape[0] * 0.5
    height = img.shape[1] * 0.5
    sqr = 70 # 0.5 width of cropped window

    crop = img[int(width-sqr): int(width+sqr), int(height-sqr):int(height+sqr)]

    img_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    alpha = 5 # contrast control
    beta = 2 # brightness control
    adjusted = cv2.convertScaleAbs(img_gray, alpha=alpha, beta=beta)

    # Brightness Band Pass Filter
    def bPass(val, min, max):
        if val > min and val < max:
            return val
        else:
            return 0

    bPassV = np.vectorize(bPass)

    bpassed = bPassV(adjusted, 88, 255)
    bpassed = cv2.normalize(src=bpassed, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Blur
    blur = cv2.resize(cv2.medianBlur(bpassed, 1), (500, 500))

    return blur

def p1_change(new_val):
    change_params("p1", new_val)

def p2_change(new_val):
    change_params("p2", new_val)

def dpi_change(new_val):
    change_params("dpi", new_val)

def minD_change(new_val):
    change_params("minD", new_val)

def Rmin_change(new_val):
    change_params("Rmin", new_val)

def Rmax_change(new_val):
    change_params("Rmax", new_val)

def change_params(name, value):
    global circle_params
    circle_params[name] = value
    circle()

# Circle
def circle():

    global circle_params
    p1 = circle_params["p1"]
    p2 = circle_params["p2"]
    dpi = circle_params["dpi"]
    minD = circle_params["minD"]
    Rmin = circle_params["Rmin"]
    Rmax = circle_params["Rmax"]

    global fimg

    circles = cv2.HoughCircles(fimg,cv2.HOUGH_GRADIENT,dpi,minD,
    param1=p1,param2=p2,minRadius=Rmin,maxRadius=Rmax)

    cp = cv2.cvtColor(fimg, cv2.COLOR_GRAY2RGB)
    cp = cv2.resize(cp, (500, 500)).copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(cp,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(cp,(i[0],i[1]),2,(0,0,255),3)

    cv2.imshow('frame', cp)

"""
cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)

trackbar_name = 'P1'
cv2.createTrackbar(trackbar_name, 'frame' , 0, 100, circle)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""

main()