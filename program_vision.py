import cv2
import numpy as np
import math
import serial
import json

# Variabel Global
crop_img = None
angle = -1
k_buffer = 0
config = {}
# ser = serial.Serial(PORT_SERIAL, 9600, timeout=0, parity=serial.PARITY_NONE, rtscts=1)

cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)

# default_config = {
#     "H_MIN": 0,
#     "H_MAX": 12,
#     "S_MIN": 60,
#     "S_MAX": 255,
#     "V_MIN": 122,
#     "V_MAX": 255,
#     "p1": 69,
#     "p2": 19,
#     "minrad": 83,
#     "maxrad": 200,
#     "offsetx": 300,
#     "serial": "COM101",
#     "buffer": 4
# }
default_config = {"H_MIN": 0, "H_MAX": 16, "S_MIN": 119, "S_MAX": 255, "V_MIN": 73, "V_MAX": 255, "p1": 63, "p2": 23, "minrad": 39, "maxrad": 0, "offsetx": 296, "serial": "COM101", "buffer": 4}


H_min = H_max = S_min = S_max = V_min = V_max = param1 = param2 = minrad = maxrad = offsetx = 0
def save_config():
    with open("config.json", "w") as f:
        json.dump(config, f)
try:
    with open("config.json", "r") as f:
        config = json.load(f)

except:
    config = default_config
    with open("config.json", "w") as f:
        json.dump(default_config, f)

H_min = config["H_MIN"]
H_max = config["H_MAX"]
S_min = config["S_MIN"] 
S_max = config["S_MAX"]
V_min = config["V_MIN"]
V_max = config["V_MAX"]
param1 = config["p1"]
param2 = config["p2"]
minrad = config["minrad"]
maxrad = config["maxrad"]
offsetx = config["offsetx"]
PORT_SERIAL = config["serial"]
BUFFER = config["buffer"]

def on_trackbar(val):
    global config
    config["H_MIN"] = H_min
    config["H_MAX"] = H_max
    config["S_MIN"] = S_min
    config["S_MAX"] = S_max
    config["V_MIN"] = V_min
    config["V_MAX"] = V_max
    config["p1"] = param1
    config["p2"] = param2
    config["minrad"] = minrad
    config["maxrad"] = maxrad
    config["offsetx"] = offsetx
    save_config()
    pass

def creatTrackbar():
    # create trackbars for color range
    # cv2.createTrackbar('H_MIN', 'Trackbars', 0, 180, on_trackbar)
    # cv2.createTrackbar('H_MAX', 'Trackbars', 12, 180, on_trackbar)
    # cv2.createTrackbar('S_MIN', 'Trackbars', 60, 255, on_trackbar)
    # cv2.createTrackbar('S_MAX', 'Trackbars', 255, 255, on_trackbar)
    # cv2.createTrackbar('V_MIN', 'Trackbars', 122, 255, on_trackbar)
    # cv2.createTrackbar('V_MAX', 'Trackbars', 255, 255, on_trackbar)
    # cv2.createTrackbar('p1', 'Trackbars', 69, 200, on_trackbar)
    # cv2.createTrackbar('p2', 'Trackbars', 19, 200, on_trackbar)
    # cv2.createTrackbar('minrad', 'Trackbars', 83, 200, on_trackbar)
    # cv2.createTrackbar('maxrad', 'Trackbars', 200, 200, on_trackbar)
    # cv2.createTrackbar('offsetx', 'Trackbars', 300, 640, on_trackbar) 

    cv2.createTrackbar('H_MIN', 'Trackbars', H_min, 180, on_trackbar)
    cv2.createTrackbar('H_MAX', 'Trackbars', H_max, 180, on_trackbar)
    cv2.createTrackbar('S_MIN', 'Trackbars', S_min, 255, on_trackbar)
    cv2.createTrackbar('S_MAX', 'Trackbars', S_max, 255, on_trackbar)
    cv2.createTrackbar('V_MIN', 'Trackbars', V_min, 255, on_trackbar)
    cv2.createTrackbar('V_MAX', 'Trackbars', V_max, 255, on_trackbar)
    cv2.createTrackbar('p1', 'Trackbars', param1, 200, on_trackbar)
    cv2.createTrackbar('p2', 'Trackbars', param2, 200, on_trackbar)
    cv2.createTrackbar('minrad', 'Trackbars', minrad, 200, on_trackbar)
    cv2.createTrackbar('maxrad', 'Trackbars', maxrad, 200, on_trackbar)
    cv2.createTrackbar('offsetx', 'Trackbars', offsetx, 640, on_trackbar)
      
    

# untuk windows:

cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)
cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cam.set(cv2.CAP_PROP_EXPOSURE, -6.9)
cam.set(cv2.CAP_PROP_FOCUS, 0)
creatTrackbar()

def buffering(frame):
    global k_buffer
    global angle 
    if angle == -1:
       return
 
    if k_buffer > 0:
      k_buffer -= 1
      # put text buffer
      cv2.putText(frame, "Buffer: {}".format(k_buffer), (int(frame.shape[1]/2)-60, int(frame.shape[0]-60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
      
      # progress bar
      cv2.rectangle(frame, (int(frame.shape[1]/2)-60, int(frame.shape[0]-50)), (int(frame.shape[1]/2)+60, int(frame.shape[0]-40)), (0, 0, 0), -1)
      cv2.rectangle(frame, (int(frame.shape[1]/2)-60, int(frame.shape[0]-50)), (int((frame.shape[1]/2-60)+120*k_buffer/BUFFER), int(frame.shape[0]-40)), (0, 255, 0), -1)
    
    if k_buffer <= 0:
      angle = -1
      k_buffer = BUFFER


    
    
while True:
    # img = cv2.imread("/home/fire-x/Documents/program_vision/Hasil-depan37.jpg")
    ret, frame = cam.read()
    img = frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # cv2.imshow("HSV", hsv)
    H_min = cv2.getTrackbarPos('H_MIN', 'Trackbars')
    H_max = cv2.getTrackbarPos('H_MAX', 'Trackbars')
    S_min = cv2.getTrackbarPos('S_MIN', 'Trackbars')
    S_max = cv2.getTrackbarPos('S_MAX', 'Trackbars')
    V_min = cv2.getTrackbarPos('V_MIN', 'Trackbars')
    V_max = cv2.getTrackbarPos('V_MAX', 'Trackbars')

    param1 = cv2.getTrackbarPos('p1', 'Trackbars')
    param2 = cv2.getTrackbarPos('p2', 'Trackbars')
    minrad = cv2.getTrackbarPos('minrad', 'Trackbars')
    maxrad = cv2.getTrackbarPos('maxrad', 'Trackbars')

    offsetx = cv2.getTrackbarPos('offsetx', 'Trackbars')
    

    lower_hsv = np.array([H_min, S_min, V_min])
    higher_hsv = np.array([H_max, S_max, V_max])
    
    # lower_hsv = np.array([0, 92, 192], np.uint8)
    # higher_hsv = np.array([5, 255, 255], np.uint8)
    
    mask = cv2.inRange(hsv, lower_hsv, higher_hsv)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=3)

    # edged = cv2.Canny(mask, 30, 200)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    if len(contours) > 0:
        # Find the index of the largest contour
        # areas = [cv2.contourArea(c) for c in contours]
        # max_index = np.argmax(areas)
        # cnt=contours[max_index]
        # x,y,w,h = cv2.boundingRect(cnt)

        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)


        # Crop the image with padding 5px then validate using hough circle
        crop_img = frame[y-40 if y > 40 else y:y+h+40, x-40 if x > 40 else x:x+w+40]

        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        try:
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=param1, param2=param2, minRadius=minrad, maxRadius=maxrad)
        except:
            pass
        
        if circles is not None:
          circles = np.uint16(np.around(circles))
          # for i in circles[0,:]:
              # draw the outer circle
              # cv2.circle(crop_img,(i[0],i[1]),i[2],(255,0,0),2)
              # draw the center of the circle
              # cv2.circle(crop_img,(i[0],i[1]),2,(0,0,255),3)
              # print(i[0], i[1])


          cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
          center_x = int(x + (w*0.5))
          center_y = int(y + (h*0.5))
          if w > 5 and h > 5:
            titik_tengah = (int(offsetx), int(frame.shape[0]))
            cv2.line(frame, titik_tengah, (center_x, center_y), (0, 255, 0), 2, cv2.LINE_AA)
            angle = int(math.atan2(titik_tengah[1] - center_y, titik_tengah[0] - center_x) * 180 / math.pi)
            angle = angle - 90 if angle > 90 else angle + 270
            cv2.circle(frame, (center_x, center_y), 5, (0,0,255), -1)
            k_buffer = BUFFER
            pass
        else:
          buffering(frame)
    else:
      buffering(frame)

    # ser.write(str(angle).encode()+b"\n")

    # show angle to frame 
    cv2.putText(frame, str(angle), (int(frame.shape[1]/2)-20, int(frame.shape[0]-20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    #cv2.imshow('mask', mask)
    cv2.imshow('img', frame)
    # cv2.imshow('img2', crop_img)

    key = cv2.waitKey(1)
    if key == 27:  # exit on ESC
        break

cv2.destroyAllWindows()
