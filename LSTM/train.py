import os

import torch
import torchvision
from torch.utils.data import DataLoader

from LSTM.model import lstm
from dataset import SkeletonDataset

import config.config as config

BATCH_SIZE = config.BATCH_SIZE  # batch

print("Loading data...")
skdata = SkeletonDataset(data_dir='data/')
val = SkeletonDataset(data_dir='val/')
dataloader = DataLoader(skdata, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
print("Loading data done!")

model = lstm()

path = 'models/lstm.pth'
# 如果有已经训练好的模型，可以直接加载
# if os.path.exists(path):
#     model.load('models/lstm.pth')

# model.save('models/lstm.pth')

print("Start training...")
for epoch in range(1000):
    for batch, target, name in dataloader:
        model.train_step(batch, target)
print("Training done!")
model.save('models/lstm.pth')
print("Model saved!")
testloader = DataLoader(val, batch_size=1, shuffle=True, drop_last=True)
count = 0
all = 0
for epoch in range(10):
    for batch, target, name in testloader:
        res = model.predict(batch)
        all += 1
        if res == target:
            count += 1
            print("True")
        else:
            print("False")
print("Accuracy: ", count / all)