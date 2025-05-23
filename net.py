from torch import nn

class MyAlexNet(nn.Module):
    def __init__(self):
        super(MyAlexNet, self).__init__()
        self.c1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.c2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.c3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.c4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.c5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)

        # 激活函数
        self.relu = nn.ReLU(inplace=True)

        # 池化层
        self.s1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.s2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.s3 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 展平
        self.flatten = nn.Flatten()

        # Dropout
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

        # 全连接层
        self.f6 = nn.Linear(256 * 6 * 6, 4096)
        self.f7 = nn.Linear(4096, 4096)
        self.f8 = nn.Linear(4096, 2)  # 猫狗分类是2类

    def forward(self, x):
        x = self.relu(self.c1(x))
        x = self.s1(x)
        x = self.relu(self.c2(x))
        x = self.s2(x)
        x = self.relu(self.c3(x))
        x = self.relu(self.c4(x))
        x = self.relu(self.c5(x))
        x = self.s3(x)
        x = self.flatten(x)
        x = self.dropout1(self.relu(self.f6(x)))
        x = self.dropout2(self.relu(self.f7(x)))
        x = self.f8(x)
        return x

if __name__ == '__main__':
    import torch

    # 创建一个随机图像张量（batch size = 1, 3 通道, 224x224 尺寸）
    x = torch.rand([1, 3, 224, 224])

    # 实例化模型
    model = MyAlexNet()

    # 将模型设置为评估模式（不启用 dropout）
    model.eval()

    # 前向传播
    y = model(x)

    # 打印输出形状
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")  # 应该是 [1, 2]
