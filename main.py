import cv2
import serial.tools.list_ports
from PIL import Image
import torch
from torchvision import transforms
from model import resnet18
import json
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

port_list = list(serial.tools.list_ports.comports())
port_name = "STM32"
for i in range(0, len(port_list)):
    if port_name in port_list[i].description:
        serial = serial.Serial(port_list[i].device, 115200, timeout=2)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
print(cap.get(3))
print(cap.get(4))
print(cap.get(5))

with open('./class_indices.json', "r") as f:
    class_indict = json.load(f)

net = resnet18(num_classes=6).to(device)
net.load_state_dict(torch.load("weight/mms.pth", map_location=device))
net.eval()

start_time = time.time()
x = 1 # displays the frame rate every 1 second
counter = 0

while(1):


    ret, frame = cap.read(0)
    frame_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = torch.unsqueeze(data_transform(frame_PIL), dim=0)

    with torch.no_grad():
        output = torch.squeeze(net(image.to(device)))
        predict = torch.argmax(output).numpy()


    print(class_indict[str(predict)])


    cv2.imshow("capture", frame)

    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

    counter += 1
    if (time.time() - start_time) > x:
        '''print("FPS: ", counter / (time.time() - start_time))'''
        counter = 0
        start_time = time.time()