import torch
import torch.nn as nn
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from datetime import datetime  # 用于计算时间
import torch.nn.functional as F

import config.config as config

FEATURE = config.FEATURE

# 定义常量
INPUT_SIZE = FEATURE  # 定义输入的特征数
HIDDEN_SIZE = 32  # 定义一个LSTM单元有多少个神经元
BATCH_SIZE = config.BATCH_SIZE  # batch
DROP_RATE = 0.2  # drop out概率
LAYERS = 2  # 有多少隐层，一个隐层一般放一个LSTM单元


# 定义一些常用函数
# 保存日志
# fname是要保存的位置，s是要保存的内容
def log(fname, s):
    f = open(fname, 'a')
    f.write(str(datetime.now()) + ': ' + s + '\n')
    f.close()


# 定义LSTM的结构
class lstm(nn.Module):
    def __init__(self):
        super(lstm, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=LAYERS,
            dropout=DROP_RATE,
            batch_first=True  # 如果为True，输入输出数据格式是(batch, seq_len, feature)
            # 为False，输入输出数据格式是(seq_len, batch, feature)，
        )
        self.hidden_out = nn.Linear(HIDDEN_SIZE, 3)  # 最后一个时序的输出接一个全连接层
        self.h_s = None
        self.h_c = None

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, x):  # x是输入数据集
        r_out, (h_s, h_c) = self.rnn(x)  # 如果不导入h_s和h_c，默认每次都进行0初始化
        #  h_s和h_c表示每一个隐层的上一时间点输出值和输入细胞状态
        # h_s和h_c的格式均是(num_layers * num_directions, batch, HIDDEN_SIZE)
        # 如果是双向LSTM，num_directions是2，单向是1
        output = self.hidden_out(r_out[:, -1, :])
        return output

    def predict(self, x):
        x = x.to(torch.float32)
        # self.eval()
        # with torch.no_grad():
        output = self.forward(x)
        pred = output.max(1)[1]
        conf = F.softmax(output)
        return pred, conf

    def train_step(self, x, y):
        x = x.to(torch.float32)
        # self.train()
        output = self.forward(x)

        probs = F.softmax(output, dim=1)

        # probs = torch.sum(probs * target_onehot, dim=1)

        loss = self.criterion(probs, y)
        print("loss: ", loss.item())

        # 反向传播
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def test(self, x, y):
        x = x.to(torch.float32)
        output = self.forward(x)
        print("output: ", output)
        probs = F.softmax(output, dim=1)
        loss = self.criterion(probs, y)
        print("loss: ", loss.item())
        # 反向传播
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
