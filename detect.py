import cv2
import numpy
import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from model import rcnn_model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has five classes - 4 class (A/B1/B2/B3) + background
num_classes = 5

model = rcnn_model(k=4)
model.to(device)

model_path = os.path.join("./models", 'model_epoch-22-focal-adam-top4.pt')
net_state_dict = torch.load(model_path)
model.load_state_dict(net_state_dict)

test_dir = "./NBI放大验证"
patient_name_list = os.listdir(test_dir)
patient_result = {}
for patient_name in patient_name_list:
    patient_result[patient_name] = {}
    for _, _, filenames in os.walk(os.path.join(test_dir, patient_name)):
        for filename in filenames:
            if filename[-3:] == "JPG" or filename[-3:] == "jpg":
                patient_result[patient_name][filename] = {"A":0, "B1":0, "B2":0, "B3":0}

# print(patient_result)

model.eval()
classes = ['gt', 'A', 'B1', 'B2', 'B3']
colors = [(0, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (255, 255, 255)]

for patient_name, img_names in patient_result.items():
    for img_name in img_names:
        image_path = os.path.join(test_dir, patient_name, img_name)
        image = Image.open(image_path).convert("RGB")
        image_tensor = torchvision.transforms.ToTensor()(image)

        result_img = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR)

        images = [image_tensor.to(device)]
        out = model(images)
        boxes = out[0]['boxes']
        labels = out[0]['labels']
        scores = out[0]['scores']

        for cls in labels:
            patient_result[patient_name][img_name][classes[cls]] += 1

print(patient_result)
        # for idx in range(boxes.shape[0]):
        #     x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
        #     name = classes[labels[idx].item()]
        #     cv2.rectangle(result_img, (x1, y1), (x2, y2), colors[labels[idx].item()], thickness=2)
        #     cv2.putText(result_img, text=name + " %.3f" % scores[idx], org=(x1, y1 + 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #                 fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=colors[labels[idx].item()])
        #
        # save_path = image_path[:-4] + "_result.png"
        # save_path = save_path.replace("\\", "/")
        # cv2.imencode(".png", result_img)[1].tofile(save_path)
        # cv2.imshow("1", result_img)
        # cv2.waitKey(0)
        # print(save_path)