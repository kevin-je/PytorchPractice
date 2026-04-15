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
class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1x1_A = nn.Conv2d(in_channels=in_channels, out_channels=24, kernel_size=1)

        self.conv1x1_B = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1)

        self.conv1x1_C = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1)
        self.conv5x5 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=5, padding=2)

        self.conv1x1_D = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1)
        self.conv3x3_1 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, padding=1)
        self.conv3x3_2 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, padding=1)

    def forward(self, x):
        branchA = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branchA = self.conv1x1_A(branchA)
        branchA = F.relu(branchA)

        branchB = self.conv1x1_B(x)
        branchB = F.relu(branchB)

        branchC = self.conv1x1_C(x)
        branchC = self.conv5x5(branchC)
        branchC = F.relu(branchC)

        branchD = self.conv1x1_D(x)
        branchD = self.conv3x3_1(branchD)
        branchD = self.conv3x3_2(branchD)
        branchD = F.relu(branchD)

        out = torch.cat((branchA, branchB, branchC, branchD), dim=1)
        return out

#定义模型
class GoogleNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=88, out_channels=20, kernel_size=5)

        self.inception1 = InceptionA(in_channels=10)
        self.inception2 = InceptionA(in_channels=20)

        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(in_features=1408, out_features=10)

    def forward(self, x):
        x = F.relu(self.mp(self.conv1(x)))
        x = self.inception1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.inception2(x)

        x = torch.flatten(x, start_dim=1)
        logits = self.fc(x)
        return logits

model = GoogleNet().to(device)

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
            torch.save(model.state_dict(), "GoogleNet_mnist.pth")

    # 结果可视化
    plt.rcParams['font.sans-serif'] = ['SimHei']
    x_date = np.arange(epochs) + 1
    y_data = np.array(accuracy_lst)
    plt.xlabel("训练轮次")
    plt.ylabel(r"准确率（$\text{%}$）")
    plt.ylim(93, 100)
    plt.title("MLP在MNIST数据集上的准确率")
    plt.grid(True)
    plt.plot(x_date, y_data, marker="o", color="r")
    plt.savefig("GoogleNet_mnist.png")
    plt.show()