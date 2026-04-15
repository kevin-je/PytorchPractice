import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

epochs = 10
batch_size = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
accuracy_max = 0
accuracy_lst = []

#加载数据
train_dataset = datasets.MNIST(
    root='C:/Users/asus/PycharmProjects/PytorchPractice/MnistData',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_dataset = datasets.MNIST(
    root='C:/Users/asus/PycharmProjects/PytorchPractice/MnistData',
    train=False,
    transform=transforms.ToTensor(),
    download=False
)

#定义模块
class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = F.relu(out+x, inplace=True)
        return out

#定义网络
class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.rb1 = ResNetBlock(16)

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.rb2 = ResNetBlock(32)

        self.fc = nn.Linear(32 * 4 * 4, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.rb1(out)
        out = self.conv2(out)
        out = self.rb2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

model = ResNet().to(device)

#建立损失函数和优化器
criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


if __name__ == "__main__":
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # 建立训练循化
    for epoch in range(epochs):
        model.train()
        print("=" * 6 + f"当前轮数：{epoch + 1}" + "=" * 6)
        loss_sum = 0
        for batch_idx, (input_data, target) in enumerate(train_loader):

            input_data: torch.Tensor
            target: torch.Tensor
            input_data = input_data.to(device)
            target = target.to(device)

            p = model(input_data)

            loss = criterion(p, target)
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 99:
                print(f"批次：{batch_idx + 1}，损失：{loss:.2f}")

        loss_avg = loss_sum / len(train_loader)
        print(f"---当前批次的训练集平均损失：{loss_avg:.2f}---")

        # 测试
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for input_data, target in test_loader:
                input_data = input_data.to(device)
                target = target.to(device)

                p = model(input_data)
                y_hat = torch.argmax(p, dim=1)
                correct += (y_hat == target).sum().item()
                total += target.size(0)
            accuracy = 100 * correct / total
            accuracy_lst.append(accuracy)
            print(f"---当前模型的测试集准确率：{accuracy:.2f}%---" + "\n")

        # 存储最佳模型参数
        if accuracy > accuracy_max:
            accuracy_max = accuracy
            torch.save(model.state_dict(), "ResNet_mnist.pth")

    # 结果可视化
    plt.rcParams['font.sans-serif'] = ['SimHei']
    x_date = np.arange(epochs) + 1
    y_data = np.array(accuracy_lst)
    plt.xlabel("训练轮次")
    plt.ylabel(r"准确率（$\text{%}$）")
    plt.ylim(93, 100)
    plt.title("ResNet在MNIST数据集上的准确率")
    plt.grid(True)
    plt.plot(x_date, y_data, marker="o", color="r")
    plt.savefig("ResNet_mnist.png")
    plt.show()