# virtual_painter_fixed.py
import cv2
import time
import handtrackingmodule as htm
import numpy as np
import os

# Settings
WIDTH, HEIGHT = 1280, 720
HEADER_HEIGHT = 125

brushThickness = 25
eraserThickness = 100
drawColor = (255, 0, 255)  # default pink

xp, yp = 0, 0
imgCanvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

# Load header images (expected filenames)
folderPath = "Header"
# expected files (you can change names if needed)
expected = [
    "header_pink.png",
    "header_blue.png",
    "header_green.png",
    "header_eraser.png"
]

overlayList = []
# Try to load each expected file; if missing, fall back to the first image in the folder
for name in expected:
    path = os.path.join(folderPath, name)
    if os.path.exists(path):
        img = cv2.imread(path)
        overlayList.append(img)
    else:
        overlayList.append(None)

# If any is None, try to fill from whatever is in the folder
if any(x is None for x in overlayList):
    files = [f for f in os.listdir(folderPath) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files.sort()
    for i in range(len(overlayList)):
        if overlayList[i] is None:
            if i < len(files):
                overlayList[i] = cv2.imread(os.path.join(folderPath, files[i]))
            else:
                overlayList[i] = np.ones((HEADER_HEIGHT, WIDTH, 3), np.uint8) * 200  # placeholder

# Ensure all headers are resized to match width x HEADER_HEIGHT
for i in range(len(overlayList)):
    if overlayList[i] is None:
        overlayList[i] = np.ones((HEADER_HEIGHT, WIDTH, 3), np.uint8) * 200
    else:
        overlayList[i] = cv2.resize(overlayList[i], (WIDTH, HEADER_HEIGHT))

# default header
header = overlayList[0]

# Video capture
cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

detector = htm.handDetector(detectionCon=0.50, maxHands=1)

def set_tool_by_x(x):
    """Return (header_img, drawColor) for a given x coordinate."""
    # divide width into 4 equal zones
    zone = WIDTH // 4
    if 0 <= x < zone:
        return overlayList[0], (255, 0, 255)  # Pink
    elif zone <= x < 2 * zone:
        return overlayList[1], (255, 0, 0)    # Blue
    elif 2 * zone <= x < 3 * zone:
        return overlayList[2], (0, 255, 0)    # Green
    else:
        return overlayList[3], (0, 0, 0)      # Eraser

print("Press ESC to exit.")
while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera.")
        break

    img = cv2.flip(img, 1)
    img = cv2.resize(img, (WIDTH, HEIGHT))

    # Find hands and landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1], lmList[8][2]   # index finger tip
        x2, y2 = lmList[12][1], lmList[12][2]  # middle finger tip

        fingers = detector.fingersUp()  # list of 5 ints (0 or 1)

        # Selection mode: both index and middle up
        if fingers[1] == 1 and fingers[2] == 1:
            xp, yp = 0, 0
            # If the tip is in header region, select tool based on x
            if y1 < HEADER_HEIGHT:
                header, drawColor = set_tool_by_x(x1)
            # Visual selection rectangle
            cv2.rectangle(img, (x1 - 20, y1 - 25), (x2 + 20, y2 + 25), drawColor, cv2.FILLED)

        # Drawing mode: index up, middle down
        if fingers[1] == 1 and fingers[2] == 0:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            # draw on both screen and canvas
            if drawColor == (0, 0, 0):  # eraser
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    # Merge canvas and frame
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    # Combine so colored strokes show over camera feed
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Put header
    img[0:HEADER_HEIGHT, 0:WIDTH] = header

    cv2.imshow("Image", img)
    k = cv2.waitKey(1)
    if k == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
