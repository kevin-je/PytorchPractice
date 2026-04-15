import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

batch_size = 64
epochs = 50
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#准备数据集
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

#定义模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, padding=0, bias=True)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, padding=0, bias=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.pool(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.pool(x))
        x = torch.flatten(x, start_dim=1)
        logits = self.fc(x)
        return logits

#建立损失函数和优化器
model = CNN().to(device)
criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

if __name__ == '__main__':
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    accuracy_lst = []

    #训练模型
    for epoch in range(epochs):
        print("=" * 6 + f"当前轮数：{epoch + 1}" + "=" * 6)
        loss_sum = 0
        for batch_idx, (input_data, target) in enumerate(train_loader):
            input_data: torch.Tensor
            target: torch.Tensor
            input_data = input_data.to(device)
            target = target.to(device)
            p = model(input_data)
            loss = criterion(p, target)
            if batch_idx % 100 == 99:
                print(f"批次：{batch_idx + 1}，损失：{loss:.2f}")
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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

    torch.save(model.state_dict(), "cnn_mnist.pth")

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
    plt.savefig("cnn_mnist.png")
    plt.show()