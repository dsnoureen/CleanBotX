import cv2
import numpy as np

def floor_detection(frame):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Define structuring elements
    structuring_elements = [
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8),  # Identity
        np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8),  # All ones
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8),  # Cross
        np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.uint8)  # X shape
    ]
    
    # Threshold for floor detection
    threshold = 150
    
    # Create an empty image for storing floor detection results
    floor_frame = np.zeros_like(gray_frame, dtype=np.uint8)
    
    # Apply the floor detection algorithm using structuring elements
    for element in structuring_elements:
        # Perform morphological operation with the structuring element
        morph_result = cv2.morphologyEx(gray_frame, cv2.MORPH_HITMISS, element)
        
        # Threshold the morphological result to detect floor regions
        floor_mask = cv2.threshold(morph_result, threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Add the detected floor regions to the floor frame
        floor_frame = cv2.bitwise_or(floor_frame, floor_mask)
    
    return floor_frame



def thresholdingFrame(frame):
    frameHsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lowerWhite = np.array([80,0,0])
    # upperWhite = np.array([255,134,255])
    lowerWhite = np.array([0,0,134])
    upperWhite = np.array([255,21,205])
    maskWhite = cv2.inRange(frameHsv, lowerWhite, upperWhite)
    return maskWhite


def warpImg(frame, points, w, h, inv = False):
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0], [w,0], [0,h], [w,h]])
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2, pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
    frameWarp = cv2.warpPerspective(frame, matrix,(w,h))
    return frameWarp  

def drawPoints(frame, points):
    for x in range(4):
        cv2.circle(frame, (int(points[x][0]), int(points[x][1])),12,(0,0,255), cv2.FILLED,cv2.LINE_4)
    return frame

def nothing(a):
    pass


def initializeTrackbar(iniTrackbarVals, wT=600, hT=400):
    cv2.namedWindow("Trackbar")
    cv2.resizeWindow("Trackbar", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbar", iniTrackbarVals[0], wT//2, nothing)
    cv2.createTrackbar("Height Top", "Trackbar", iniTrackbarVals[1], hT, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbar", iniTrackbarVals[2], wT//2, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbar", iniTrackbarVals[3], hT, nothing)


def valTrackbar(wT = 600, hT = 400):
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbar")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbar")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbar")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbar")
    points = np.float32([(widthTop, heightTop), (wT-widthTop, heightTop), (widthBottom, heightBottom), (wT-widthBottom, heightBottom)])
    return points

def getHistogram(frame, minPer = 0.1, display=False, region=1):
    if region == 1:
        histValue = np.sum(frame, axis=0)
    else:
        histValue = np.sum(frame[frame.shape[0]//region:, :], axis=0)
    
    # print(histValue)
    maxValue = np.max(histValue)
    minValue = minPer*maxValue
    indexArr = np.where(histValue >= minValue)
    basePoint = int(np.average(indexArr))
    # print(basePoint)
    
    if display:
        frameHist = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
        for x, intensity in enumerate(histValue):
            cv2.line(frameHist, (x, frame.shape[0]), (x, frame.shape[0]-intensity//255//region), (255, 0, 255), 1)
            cv2.circle(frameHist, (basePoint, frame.shape[0]), 20, (0, 255, 255), cv2.FILLED)
        return basePoint, frameHist
    return basePoint

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver