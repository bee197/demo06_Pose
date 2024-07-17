import os
import re
import numpy as np
import torch
from torch import arctan2
from torch.utils.data import Dataset
from LSTM.util import read_xy, read_xyz

category2label = {
    'A023': 0,  # 挥手
    'A043': 1,  # 跌倒
    'A050': 2,  # 打/拍打
}

import config.config as config

LENGTH = config.LENGTH
FEATURE = config.FEATURE
INDEX = config.INDEX


class SkeletonDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []
        self.targets = []
        self.name = []
        txt_files = os.listdir(data_dir)
        for fname in txt_files:
            points = read_xy(data_dir + fname)
            # 使用正则表达式解析类别
            category = re.search(r'A\d+', fname).group()
            # 将类别映射成数字标签
            target = category2label[category]

            self.data.append(points)
            self.targets.append(target)
            self.name.append(fname)

    def __getitem__(self, idx):
        length = LENGTH
        start = np.random.randint(0, self.data[idx].shape[1] - length)
        end = start + length
        # [2, LENGTH, 13, 2]
        data = self.data[idx][:, start:end, ...]
        # [2, LENGTH, 13, 2]
        data = data[:, :, INDEX, :]
        # 将numpy.ndarray转换为Tensor
        data = torch.from_numpy(data)
        # 交换第1个维度和第2个维度
        # [LENGTH, 2, 13, 2]
        data = torch.transpose(data, 0, 1)
        # 使用索引获取形状为(LENGTH, 3, 13, 1)的数据
        # [LENGTH, 2, 13]
        data = data[:, 0:2, :, 0]
        # 取得x,y坐标
        # [LENGTH, 13]
        x_data = data[:, 0]
        y_data = data[:, 1]
        # 计算肘/膝关节角度
        # angle1 = self.get_angle(x_data[:, 1], x_data[:, 2], x_data[:, 3], y_data[:, 1], y_data[:, 2], y_data[:, 3])
        # angle2 = self.get_angle(x_data[:, 4], x_data[:, 5], x_data[:, 6], y_data[:, 4], y_data[:, 5], y_data[:, 6])
        # angle3 = self.get_angle(x_data[:, 7], x_data[:, 8], x_data[:, 9], y_data[:, 7], y_data[:, 8], y_data[:, 9])
        # angle4 = self.get_angle(x_data[:, 10], x_data[:, 11], x_data[:, 12], y_data[:, 10], y_data[:, 11],
        #                         y_data[:, 12])
        # # [LENGTH, 13]
        # angle = torch.cat((angle1.unsqueeze(1), angle2.unsqueeze(1), angle3.unsqueeze(1), angle4.unsqueeze(1)),
        #                   1)
        # 计算给定两个坐标的距离
        # [LENGTH, 13]
        dist = torch.sqrt(x_data ** 2 + y_data ** 2)
        # [LENGTH, 26]
        data = data.reshape(LENGTH, 13 * 2)
        # concat
        # [LENGTH, 39]
        data = torch.cat((data, dist), 1)

        return data, self.targets[idx], self.name[idx]

    def __len__(self):
        return len(self.data)

    def get_angle(self, x1, x2, x3, y1, y2, y3):
        v1 = (x1 - x2, y1 - y2)  # 上臂骨骼端点向量
        v2 = (x3 - x2, y3 - y2)  # 下臂骨骼端点向量

        dot = v1[0] * v2[0] + v1[1] * v2[1]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        angle = arctan2(cross, dot) / np.pi * 180
        return angle
