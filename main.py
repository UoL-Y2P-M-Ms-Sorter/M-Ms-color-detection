import cv2
import serial.tools.list_ports
from PIL import Image
import torch
from torchvision import transforms
from model import resnet18
import time
from com import send
from pynput import keyboard
from pynput.keyboard import Key, Controller
import json
import threading

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

global frame


def cap0():
    while 1:
        global frame
        _, frame = cap.read()
        cv2.imshow("capture", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break


capture = threading.Thread(target=cap0)

keyboard0 = Controller()

isEnd = False

with open('./class_indices.json', "r") as f:
    class_indict = json.load(f)


def keyboard_on_release(key):
    global isEnd
    if key == keyboard.Key.esc:
        isEnd = True
        return False


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("{} is in use".format(device))

data_transform = transforms.Compose(
    [transforms.CenterCrop([200, 200]),
     transforms.Resize(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

net = resnet18(num_classes=7).to(device)
net.load_state_dict(torch.load("weight/mms.pth", map_location=device))
net.eval()

device_exist = 0

capture.start()

stopper = keyboard.Listener(on_release=keyboard_on_release)
while 1:

    with keyboard.Listener(
            on_release=keyboard_on_release) as starter:
        starter.join()

    stopper = keyboard.Listener(on_release=keyboard_on_release)
    stopper.start()
    isEnd = False

    start_time = time.time()
    x = 1  # displays the frame rate every 1 second
    counter = 0

    while 1:
        if device_exist == 0:
            import serial.tools.list_ports

            port_list = list(serial.tools.list_ports.comports())
            port_name = "COM5"
            for i in range(0, len(port_list)):
                if port_name in port_list[i].description:
                    try:
                        serial = serial.Serial(port_list[i].device, write_timeout=1)
                        device_exist = 1
                    except Exception:
                        pass

        frame_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = torch.unsqueeze(data_transform(frame_PIL), dim=0)

        with torch.no_grad():
            output = torch.squeeze(net(image.to(device))).cpu()
            predict = torch.argmax(output).numpy()

            device_exist = send(serial, 0, predict)

        counter += 1
        if (time.time() - start_time) >= x:
            print(device_exist)
            print("FPS: %.3f" % (counter / (time.time() - start_time)))
            print(class_indict[str(predict)])
            print()
            counter = 0
            start_time = time.time()

        if isEnd:
            if device_exist:
                send(serial, 1, predict)
            break
