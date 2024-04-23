import cv2
import torch
from ultralytics import YOLO


SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
            [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]]
event = ["fire", "smoke"]
CONFIDENCE_SMALL = 0.7
CONFIDENCE_MIDDLE = 0.8
CONFIDENCE_LARGE = 0.9

img = cv2.imread('image/smoking2.png')
# 必须转化为(640, 640)
img = cv2.resize(img, (640, 640))
# 转化为tensor
tensor = torch.from_numpy(img).float()/255.0
# [640, 640, 3] -> [640, 3, 640] -> [3, 640, 640]
tensor = tensor.transpose(1, 2).transpose(0, 1)
# 增加维度
tensor = torch.unsqueeze(tensor, 0)


# 加载预训练的模型和参数
model1 = YOLO("yolov8s-pose.pt", task="pose")
model2 = YOLO("detect.pt", task="detect")
print("模型加载成功！")
# 进行预测
results1 = model1(tensor)
results2 = model2(tensor)
print("预测完成！")


boxes, keypoints = results1[0].boxes, results1[0].keypoints
# 遍历每个人
for i in range(len(boxes)):
    x1, y1, x2, y2 = boxes.xyxy.cpu()[i].numpy().astype(int)
    kpt = keypoints.data[i].cpu().numpy().reshape(-1)
    # 至少是个人吧
    if boxes.conf[i] > CONFIDENCE_SMALL:
        # 画出框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)

        for j in range(19):
            xi, yi = SKELETON[j]

            pos1_s = kpt[(xi - 1) * 3 + 2]
            pos2_s = kpt[(yi - 1) * 3 + 2]
            # 至少长得像个关节点吧
            if pos1_s > CONFIDENCE_SMALL and pos2_s > CONFIDENCE_SMALL:
                # 画出关节点
                if j < 17:
                    pos_x, pos_y = int(round(kpt[j * 3])), int(round(kpt[j * 3 + 1]))
                    cv2.circle(img, (pos_x, pos_y), 5, (0, 255, 0), -1)
                # 按一定顺序连接关节点
                pos1_x, pos1_y = int(round(kpt[(xi - 1) * 3])), int(round(kpt[(xi - 1) * 3 + 1]))
                pos2_x, pos2_y = int(round(kpt[(yi - 1) * 3])), int(round(kpt[(yi - 1) * 3 + 1]))
                cv2.line(img, (pos1_x, pos1_y), (pos2_x, pos2_y), (0, 0, 255), 4)

boxes = results2[0].boxes
for i in range(len(boxes)):
    if boxes.conf[i] > CONFIDENCE_SMALL:
        x1, y1, x2, y2 = boxes.xyxy.cpu()[i].numpy().astype(int)
        cls = boxes.cls[i].cpu().numpy().astype(int)
        # 属于那两个事件
        if cls < len(event):
            eve = event[cls]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
            # 绘制类别在图像上
            cv2.putText(img, eve, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

cv2.imshow('img', img)
cv2.waitKey(0)


