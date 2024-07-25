import cv2
import numpy as np

scalef = 2

def filter(img):
    # Crop
    width = img.shape[0] * 0.5
    height = img.shape[1] * 0.5
    sqr = 100 # 0.5 width of cropped window

    #crop = img[int(width-sqr): int(width+sqr), int(height-sqr):int(height+sqr)]
    crop = img

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

    bpassed = bPassV(adjusted, 40, 255)
    bpassed = cv2.normalize(src=bpassed, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    bw = int(width * 2 * scalef)
    bh = int(height * 2 * scalef)

    # Blur
    #blur = cv2.resize(cv2.medianBlur(bpassed, 1), (bw, bh))
    blur = cv2.medianBlur(bpassed, 1)

    return blur

img = cv2.imread(r"FreqImgAnalysis\FFT394.jpg")
fimg = filter(img)

# Blob

# blob detection
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = False
params.minThreshold = 65
params.maxThreshold = 93
params.blobColor = 0
params.minArea = 10
params.maxArea = 100
params.filterByCircularity = False
params.filterByConvexity = False
params.minCircularity =.4
params.maxCircularity = 1

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(fimg)

im_with_keypoints = cv2.drawKeypoints(fimg, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

w = np.shape(fimg)[0] * 0.5
h = np.shape(fimg)[1] * 0.5

for i, key in enumerate(keypoints):
    print("{x:.4f}, {y:.4f}, {s:.4f}".format(x = keypoints[i].pt[0] - w, y = keypoints[i].pt[1] - h, s = keypoints[i].size))

# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)