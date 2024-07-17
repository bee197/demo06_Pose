import os

from torch.utils.data import DataLoader
from LSTM.dataset import SkeletonDataset
from LSTM.model import lstm

model = lstm()
path = 'models/lstm.pth'
# 如果有已经训练好的模型，可以直接加载
if os.path.exists(path):
    model.load('models/lstm.pth')
val = SkeletonDataset(data_dir='val/')
testloader = DataLoader(val, batch_size=1, shuffle=True, drop_last=True)
count = 0
all = 0
for epoch in range(10):
    for batch, target, _ in testloader:
        res = model.predict(batch)
        all += 1
        if res == target:
            count += 1
            print("True")
        else:
            print("False")
print("Accuracy: ", count / all)