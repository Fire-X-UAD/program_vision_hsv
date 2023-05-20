import cv2
import numpy as np
import math
import serial
import json
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# Variabel Global
crop_img = None
angle = -1
k_buffer = 0
config = {}
H_min = H_max = S_min = S_max = V_min = V_max = param1 = param2 = minrad = maxrad = offsetx = 0
A_H_min = A_H_max = A_S_min = A_S_max = A_V_min = A_V_max = 0
show_result = False
ally = "cyan"
default_config = {
    "H_MIN": 0,
    "H_MAX": 16,
    "S_MIN": 119,
    "S_MAX": 255,
    "V_MIN": 73,
    "V_MAX": 255,
    "CYN_H_MIN": 93,
    "CYN_H_MAX": 105,
    "CYN_S_MIN": 151,
    "CYN_S_MAX": 255,
    "CYN_V_MIN": 196,
    "CYN_V_MAX": 255,
    "MAG_H_MIN": 0,
    "MAG_H_MAX": 180,
    "MAG_S_MIN": 0,
    "MAG_S_MAX": 255,
    "MAG_V_MIN": 0,
    "MAG_V_MAX": 255,
    "p1": 63,
    "p2": 23,
    "minrad": 39,
    "maxrad": 0,
    "offsetx": 296,
    "serial": "COM101",
    "buffer": 4,
    "show_result": False,
    "ally": "cyan"
}


def save_config():
    try:
        with open("config.json", "w") as f:
            json.dump(config, f)
    except:
        pass


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

    if ally == "cyan":
        config["CYN_H_MIN"] = A_H_min
        config["CYN_H_MAX"] = A_H_max
        config["CYN_S_MIN"] = A_S_min
        config["CYN_S_MAX"] = A_S_max
        config["CYN_V_MIN"] = A_V_min
        config["CYN_V_MAX"] = A_V_max
    else:
        config["MAG_H_MIN"] = A_H_min
        config["MAG_H_MAX"] = A_H_max
        config["MAG_S_MIN"] = A_S_min
        config["MAG_S_MAX"] = A_S_max
        config["MAG_V_MIN"] = A_V_min
        config["MAG_V_MAX"] = A_V_max

    save_config()
    pass


def ally_detection(target_frame):
    a_hsv = cv2.cvtColor(target_frame, cv2.COLOR_BGR2HSV)
    lower = np.array([A_H_min, A_S_min, A_V_min])
    higher = np.array([A_H_max, A_S_max, A_V_max])

    a_mask = cv2.inRange(a_hsv, lower, higher)
    a_mask = cv2.erode(a_mask, None, iterations=2)
    a_mask = cv2.dilate(a_mask, None, iterations=3)

    a_contours, a_hierarchy = cv2.findContours(a_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    a_angle = -1
    if len(a_contours) > 0:
        a_contours = max(a_contours, key=cv2.contourArea)
        a_x, a_y, a_w, a_h = cv2.boundingRect(a_contours)

        a_center_x = int(a_x + (a_w * 0.5))
        a_center_y = int(a_y + (a_h * 0.5))

        cv2.rectangle(target_frame, (a_x, a_y), (a_x + a_w, a_y + a_h), (0, 255, 0), 2)

        center_point = (target_frame.shape[1] // 2, target_frame.shape[0])
        cv2.line(target_frame, center_point, (a_center_x, a_center_y), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(target_frame, (a_center_x, a_center_y), 5, (0, 0, 255), -1)

        a_angle = int(math.atan2(center_point[1] - a_center_y, center_point[0] - a_center_x) * 180 / math.pi)
        a_angle = a_angle - 90 if a_angle > 90 else a_angle + 270

    if show_result:
        cv2.imshow("Ally", target_frame)

    return a_angle


def creatTrackbar():
    cv2.createTrackbar('H_MIN', 'Trackbars', H_min, 180, on_trackbar)
    cv2.createTrackbar('H_MAX', 'Trackbars', H_max, 180, on_trackbar)
    cv2.createTrackbar('S_MIN', 'Trackbars', S_min, 255, on_trackbar)
    cv2.createTrackbar('S_MAX', 'Trackbars', S_max, 255, on_trackbar)
    cv2.createTrackbar('V_MIN', 'Trackbars', V_min, 255, on_trackbar)
    cv2.createTrackbar('V_MAX', 'Trackbars', V_max, 255, on_trackbar)
    cv2.createTrackbar('ALLY_H_MIN', 'Trackbars', A_H_min, 180, on_trackbar)
    cv2.createTrackbar('ALLY_H_MAX', 'Trackbars', A_H_max, 180, on_trackbar)
    cv2.createTrackbar('ALLY_S_MIN', 'Trackbars', A_S_min, 255, on_trackbar)
    cv2.createTrackbar('ALLY_S_MAX', 'Trackbars', A_S_max, 255, on_trackbar)
    cv2.createTrackbar('ALLY_V_MIN', 'Trackbars', A_V_min, 255, on_trackbar)
    cv2.createTrackbar('ALLY_V_MAX', 'Trackbars', A_V_max, 255, on_trackbar)
    cv2.createTrackbar('p1', 'Trackbars', param1, 200, on_trackbar)
    cv2.createTrackbar('p2', 'Trackbars', param2, 200, on_trackbar)
    cv2.createTrackbar('minrad', 'Trackbars', minrad, 200, on_trackbar)
    cv2.createTrackbar('maxrad', 'Trackbars', maxrad, 200, on_trackbar)
    cv2.createTrackbar('offsetx', 'Trackbars', offsetx, 640, on_trackbar)


def buffering(frame):
    global k_buffer
    global angle
    if angle == -1:
        return

    if k_buffer > 0:
        k_buffer -= 1
        # put text buffer
        cv2.putText(frame, "Buffer: {}".format(k_buffer), (int(frame.shape[1] / 2) - 60, int(frame.shape[0] - 60)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # progress bar
        cv2.rectangle(frame, (int(frame.shape[1] / 2) - 60, int(frame.shape[0] - 50)),
                      (int(frame.shape[1] / 2) + 60, int(frame.shape[0] - 40)), (0, 0, 0), -1)
        cv2.rectangle(frame, (int(frame.shape[1] / 2) - 60, int(frame.shape[0] - 50)),
                      (int((frame.shape[1] / 2 - 60) + 120 * k_buffer / BUFFER), int(frame.shape[0] - 40)), (0, 255, 0),
                      -1)

    if k_buffer <= 0:
        angle = -1
        k_buffer = BUFFER


try:
    with open("config.json", "r+b") as f:
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
show_result = config["show_result"]
ally = config["ally"]

if ally == "cyan":
    A_H_min = config["CYN_H_MIN"]
    A_H_max = config["CYN_H_MAX"]
    A_S_min = config["CYN_S_MIN"]
    A_S_max = config["CYN_S_MAX"]
    A_V_min = config["CYN_V_MIN"]
    A_V_max = config["CYN_V_MAX"]
else:
    A_H_min = config["MAG_H_MIN"]
    A_H_max = config["MAG_H_MAX"]
    A_S_min = config["MAG_S_MIN"]
    A_S_max = config["MAG_S_MAX"]
    A_V_min = config["MAG_V_MIN"]
    A_V_max = config["MAG_V_MAX"]

if PORT_SERIAL != '':
    ser = serial.Serial(PORT_SERIAL, 9600, timeout=0, parity=serial.PARITY_NONE, rtscts=1)

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cam.set(cv2.CAP_PROP_EXPOSURE, -5.2)
cam.set(cv2.CAP_PROP_FOCUS, 0)

if show_result:
    cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Ball', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Ally', cv2.WINDOW_NORMAL)

    cv2.resizeWindow("Trackbars", 400, 800)
    cv2.resizeWindow("Ball", 640, 480)
    cv2.resizeWindow("Ally", 640, 480)

if show_result:
    creatTrackbar()

while True:
    ret, frame = cam.read()
    img = frame

    ally_detection(img)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if show_result:
        H_min = cv2.getTrackbarPos('H_MIN', 'Trackbars')
        H_max = cv2.getTrackbarPos('H_MAX', 'Trackbars')
        S_min = cv2.getTrackbarPos('S_MIN', 'Trackbars')
        S_max = cv2.getTrackbarPos('S_MAX', 'Trackbars')
        V_min = cv2.getTrackbarPos('V_MIN', 'Trackbars')
        V_max = cv2.getTrackbarPos('V_MAX', 'Trackbars')
        A_H_min = cv2.getTrackbarPos('ALLY_H_MIN', 'Trackbars')
        A_H_max = cv2.getTrackbarPos('ALLY_H_MAX', 'Trackbars')
        A_S_min = cv2.getTrackbarPos('ALLY_S_MIN', 'Trackbars')
        A_S_max = cv2.getTrackbarPos('ALLY_S_MAX', 'Trackbars')
        A_V_min = cv2.getTrackbarPos('ALLY_V_MIN', 'Trackbars')
        A_V_max = cv2.getTrackbarPos('ALLY_V_MAX', 'Trackbars')
        param1 = cv2.getTrackbarPos('p1', 'Trackbars')
        param2 = cv2.getTrackbarPos('p2', 'Trackbars')
        minrad = cv2.getTrackbarPos('minrad', 'Trackbars')
        maxrad = cv2.getTrackbarPos('maxrad', 'Trackbars')
        offsetx = cv2.getTrackbarPos('offsetx', 'Trackbars')

    lower_hsv = np.array([H_min, S_min, V_min])
    higher_hsv = np.array([H_max, S_max, V_max])

    mask = cv2.inRange(hsv, lower_hsv, higher_hsv)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=3)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        # Crop the image with padding 5px then validate using hough circle
        crop_img = frame[y - 40 if y > 40 else y:y + h + 40, x - 40 if x > 40 else x:x + w + 40]

        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        try:
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                1,
                100,
                param1=param1,
                param2=param2,
                minRadius=minrad,
                maxRadius=maxrad
            )
        except:
            pass

        if circles is not None:
            circles = np.uint16(np.around(circles))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            center_x = int(x + (w * 0.5))
            center_y = int(y + (h * 0.5))
            if w > 5 and h > 5:
                titik_tengah = (int(offsetx), int(frame.shape[0]))
                cv2.line(frame, titik_tengah, (center_x, center_y), (0, 255, 0), 2, cv2.LINE_AA)
                angle = int(math.atan2(titik_tengah[1] - center_y, titik_tengah[0] - center_x) * 180 / math.pi)
                angle = angle - 90 if angle > 90 else angle + 270
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                k_buffer = BUFFER
                pass
        else:
            buffering(frame)
    else:
        buffering(frame)

    cv2.line(frame, (offsetx - 100, frame.shape[0]), (offsetx - 150, 0), (255, 0, 0), 2, cv2.LINE_AA)
    cv2.line(frame, (offsetx + 100, frame.shape[0]), (offsetx + 150, 0), (255, 0, 0), 2, cv2.LINE_AA)

    if angle != -1:
        point = Point(center_x, center_y)
        polygon = Polygon([
            (offsetx - 100, frame.shape[0]),
            (offsetx - 150, 0),
            (offsetx + 150, 0),
            (offsetx + 100, frame.shape[0])
        ])
        if polygon.contains(point):
            angle = 0

    print(angle)
    # show angle to frame
    cv2.putText(
        frame,
        str(angle),
        (int(frame.shape[1] / 2) - 20, int(frame.shape[0] - 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    if PORT_SERIAL != '':
        ser.write(str(angle).encode() + b"\n")
        ser.flush()
        ser.flushOutput()

    if show_result:
        cv2.imshow('Ball', frame)

    key = cv2.waitKey(1)
    if key == 27:  # exit on ESC
        break

cv2.destroyAllWindows()
