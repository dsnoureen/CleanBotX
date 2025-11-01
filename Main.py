import utils
import cv2
import numpy as np
import requests
import socket

curveList = []
avgVal = 10

ESP32_IP = '192.168.43.106'
ESP32_PORT = 8080

def send_command_to_esp32(command):
    cmd = str(command)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((ESP32_IP, ESP32_PORT))
            s.sendall(command.encode())
            s.close()
    except Exception as e:
        s.close()
        print(f"Error: {e}")


def createPath(frame, initialtracbarVal, display=2):
    frame = cv2.resize(frame,(600, 400))
    imgResult = frame.copy()
    frameCpy = frame.copy()
    points = utils.valTrackbar()
    #if Tracbar not needed, then do cmt trackvar and undo cmt wT and points
    # wT = 600
    # points = np.float32([(initialtracbarVal[0], initialtracbarVal[1]), (wT-initialtracbarVal[0], initialtracbarVal[1]), (initialtracbarVal[2], initialtracbarVal[3]), (wT-initialtracbarVal[2], initialtracbarVal[3])])

    # Undo cmt one of three
    # frameThres = utils.floor_detection(frame)
    # frameThres = utils.thresholdingFrame(frame)
    frameThres = tempUtils.obstavle_detection(frame)

    
    hT, wT, c= frame.shape

    frameWarp = utils.warpImg(frameThres, points, wT, hT)
    frameWarpPoints = utils.drawPoints(frameCpy, points)


    middlePoint, frameHist = utils.getHistogram(frameWarp, display=True, minPer=0.5, region=4)
    curAvgPoint, frameHist = utils.getHistogram(frameWarp, display=True, minPer=0.9)
    curveRaw = curAvgPoint - middlePoint 

    curveList.append(curveRaw)
    if len(curveList) > avgVal:
        curveList.pop(0)
    curve = int(sum(curveList)/len(curveList))


    if display != 0:
       imgInvWarp = utils.warpImg(frameWarp, points, wT, hT,inv = True)
       imgInvWarp = cv2.cvtColor(imgInvWarp,cv2.COLOR_GRAY2BGR)
       imgInvWarp[0:hT//3,0:wT] = 0,0,0
       imgLaneColor = np.zeros_like(frame)
       imgLaneColor[:] = 0, 255, 0
       imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
       imgResult = cv2.addWeighted(imgResult,1,imgLaneColor,1,0)
       midY = 450
       cv2.putText(imgResult,str(curve),(wT//2-80,85),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),3)
       cv2.line(imgResult,(wT//2,midY),(wT//2+(curve*3),midY),(255,0,255),5)
       cv2.line(imgResult, ((wT // 2 + (curve * 3)), midY-25), (wT // 2 + (curve * 3), midY+25), (0, 255, 0), 5)
       for x in range(-30, 30):
           w = wT // 20
           cv2.line(imgResult, (w * x + int(curve//50 ), midY-10), (w * x + int(curve//50 ), midY+10), (0, 0, 255), 2)
    
    #    For FPS 
    #    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    #    cv2.putText(imgResult, 'FPS '+str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (230,50,50), 3)

    if display == 2:
        imgStacked = utils.stackImages(0.7,([frame,frameWarpPoints,frameWarp],
                                            [frameHist,imgLaneColor,imgResult]))
        cv2.imshow('ImageStack',imgStacked)
    elif display == 1:
        cv2.imshow('Resutlt',imgResult)

    # cv2.imshow("img Warp", frameWarp)
    # cv2.imshow("img warp points", frameWarpPoints)
    # cv2.imshow("Histogram", frameHist) 

    # Normalixation
    curve = curve / 100
    if curve > 1 : curve = 1
    if curve < -1 : curve = -1


    return curve


url = 'http://192.168.43.67/cam-mid.jpg'

if __name__ == "__main__":
    # cap = cv2.VideoCapture(requests.get(url))
    initialtracbarVal = [114, 280, 31, 367]
    utils.initializeTrackbar(initialtracbarVal)
    # Read an input frame
    #input_frame = cv2.imread("p1.jpg")
    while True:
        # ret, frame = cap.read()
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        frame = cv2.imdecode(img_arr, -1)
        
        
        # Perform floor detection
        curve = createPath(frame, initialtracbarVal)
        send_command_to_esp32(curve)
        # print(curve)
        # print(f"Threshold: {threshold}")
        # print(f"Entropy: {entropy}")
        # Display results
        # cv2.imshow("Obstacle Image", obstacle_image)
        # cv2.imshow("Main Image", frame)
        # cv2.imshow("Floor Image", floor_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # cap.release()
    cv2.destroyAllWindows()
