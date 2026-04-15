import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

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



#设计模型
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        logits = self.linear4(x)
        return logits

image, label = train_dataset[0]
H = image.shape[1]
W = image.shape[2]
input_size = H * W
batch_size = 64
epochs = 50
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#建立损失函数和优化器
model = MLP(input_size).to(device)
criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

if __name__ == '__main__':

    # 建立训练循环
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    accuracy_lst = []
    accuracy_max = 0

    for epoch in range(epochs):
        print("="*6 + f"当前轮数：{epoch+1}" + "="*6)
        loss_sum = 0
        for batch_idx, (input_data, target) in enumerate(train_loader):
            input_data: torch.Tensor
            target: torch.Tensor
            input_data = input_data.to(device)
            target = target.to(device)
            p = model(torch.flatten(input_data, start_dim=1))
            loss = criterion(p, target)
            if batch_idx % 100 == 99:
                print(f"批次：{batch_idx+1}，损失：{loss:.2f}")
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
                p = model(torch.flatten(input_data, start_dim=1))
                y_hat = torch.argmax(p, dim=1)
                correct += (y_hat == target).sum().item()
                total += target.size(0)
            accuracy = 100 * correct / total
            accuracy_lst.append(accuracy)
            print(f"---当前模型的测试集准确率：{accuracy:.2f}%---" + "\n")

        #存储最佳模型参数
        if accuracy > accuracy_max:
            accuracy_max = accuracy
            torch.save(model.state_dict(), "mlp_mnist.pth")

    #结果可视化
    plt.rcParams['font.sans-serif'] = ['SimHei']
    x_date = np.arange(epochs) + 1
    y_data = np.array(accuracy_lst)
    plt.xlabel("训练轮次")
    plt.ylabel(r"准确率（$\text{%}$）")
    plt.ylim(93, 100)
    plt.title("MLP在MNIST数据集上的准确率")
    plt.grid(True)
    plt.plot(x_date, y_data, marker="o", color="r")
    plt.savefig("mlp_mnist.png")
    plt.show()