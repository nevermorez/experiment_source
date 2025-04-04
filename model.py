
import torch.nn as nn
import torch.optim as optim
import torch
class PairwiseCNN(nn.Module):
    # def __init__(self):
    #     super(PairwiseCNN, self).__init__()
    #     self.conv1 = nn.Conv2d(in_channels=2, out_channels=12, kernel_size=3, padding=1)
    #     self.bn1 = nn.BatchNorm2d(12)
    #     self.pool1 = nn.MaxPool2d(kernel_size=2)
    #     # 可以添加更多的卷积和池化层
    #     self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1)
    #     self.bn2 = nn.BatchNorm2d(24) # batch_normalization，把输入归一化到均值0，方差1的分布
    #     self.pool2 = nn.MaxPool2d(kernel_size=2)
    #     # self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    #     # self.pool3 = nn.MaxPool1d(kernel_size=2)
    #     self.fc1 = nn.Linear(144, 144)  # 假设经过池化后得到30个特征
    #     # self.fc2 = nn.Linear(360,360)
    #     self.fc3 = nn.Linear(144, 1)  # 输出层，预测一个值
    def __init__(self):
        super(PairwiseCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(8)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        # 可以添加更多的卷积和池化层
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(16) # batch_normalization，把输入归一化到均值0，方差1的分布
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        # self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # self.bn3 = nn.BatchNorm1d(32) # batch_normalization，把输入归一化到均值0，方差1的分布
        # self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(240, 120)  # 假设经过池化后得到30个特征
        # self.fc2 = nn.Linear(480,240)
        self.fc3 = nn.Linear(120, 1)  # 输出层，预测一个值


    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.relu(self.bn2(x))
        x = self.pool2(x)
        # x = self.conv3(x)
        # x = torch.relu(self.bn3(x))
        # x = self.pool3(x)
        x = x.view(x.size(0), -1)  # 展平特征图
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x