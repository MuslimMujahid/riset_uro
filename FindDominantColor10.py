import cv2 
import numpy as np  
import time
import math

def detectLines(img_name):
    # Read image
    img = cv2.imread(img_name)

    # Resize image
    W = 480
    height, width, depth = img.shape
    imgScale = W/width
    newX, newY = img.shape[1]*imgScale, img.shape[0]*imgScale
    img = cv2.resize(img, (int(newX), int(newY)))

    # Blurring image
    img = cv2.GaussianBlur(img, (1, 1), 0)

    # Convert to HSV color
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Set dominant field color
    FieldColor = np.array([40, 188, 113])

    # Max gap to the dominant field color
    maxGap = 120

    # Convert to array 2D
    img2D = np.reshape(img, (img.shape[0] * img.shape[1],3))

    for idx in range(img2D.shape[0]):
        
        # Select current pixel
        cpixel = img2D[idx]

        # Gap to the dominant color
        gap = ((cpixel[0]-FieldColor[0])**2 + (cpixel[1]-FieldColor[1])**2 + (cpixel[2]-FieldColor[2])**2)**0.5

        # Set new color
        if (gap > maxGap):
            newColor = np.array([0, 0, 0])
        else:
            newColor = FieldColor

        # Apply new color to the image
        crow = math.ceil(idx/img.shape[1])-1
        ccol = idx-crow*img.shape[1]-1
        img[crow][ccol] = newColor

    # Convert to grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    # Set Threshold and edge
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    edge = cv2.Canny(gray, 50, 200, None, 3)

    cv2.imshow("newImage", thresh)


detectLines("kri17.jpg")

print(time.clock())
cv2.waitKey(0)
cv2.destroyAllWindows()
