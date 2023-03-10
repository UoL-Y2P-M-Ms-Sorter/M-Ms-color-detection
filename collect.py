import json
import os

import cv2
import serial.tools.list_ports

from com import send

with open('./class_indices.json', "r") as f:
    class_indict = json.load(f)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

device_exist = 0
port_list = list(serial.tools.list_ports.comports())

port_name = "STM32"

for i in range(0, len(port_list)):
    if port_name in port_list[i].description:
        serial = serial.Serial(port=port_list[i].device, write_timeout=2)
        device_exist = 1

mode = 2
color = 0

img_index = [200, 200, 200, 200, 200, 200, 200]

while 1:

    if device_exist:
        send(serial, 0, 0)

    ret, frame = cap.read()
    cv2.imshow("capture", frame)

    key = cv2.waitKey(0) & 0xFF
    color = key - 48
    cv2.imwrite(os.path.join('data/train', class_indict[str(color)], str(img_index[color] + 1) + '.png'), frame)
    img_index[color] += 1

    if key == ord('q'):
        break

serial.close()
