import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

epochs = 100
batch_size = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#构建数据集
letter = ['h', 'e', 'l', 'o']
input_seq = torch.LongTensor([0, 1, 2, 2, 3]).to(device)
target_seq = torch.LongTensor([1, 2, 0, 3, 2]).to(device)

input_seq_hot = F.one_hot(input_seq, num_classes=4).float()

#构建网络
class RNNCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn_cell = nn.RNNCell(4, 4)

    def forward(self, _x, _h):
        h1 = self.rnn_cell(_x, _h)
        return h1

#构建损失函数和优化器
model = RNNCell().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

if __name__ == "__main__":

    #构建训练循环
    for epoch in range(epochs):
        print("=" * 6 + f"当前轮数：{epoch + 1}" + "=" * 6)
        model.train()

        h = torch.zeros(4, device=device)
        loss_sum = torch.zeros(1, device=device)
        for x, y in zip(input_seq_hot, target_seq):
            h = model(x, h)
            loss_sum += criterion(h, y)
        print(f"总损失：, {loss_sum.item():.2f}")

        optimizer.zero_grad()
        loss_sum: torch.Tensor
        loss_sum.backward()
        optimizer.step()

        #评估
        model.eval()
        with torch.no_grad():
            h = torch.zeros(4, device=device)
            out = []
            for x in input_seq_hot:
                h = model(x, h)
                _, idx = h.max(0)
                y_hat = letter[idx]
                out.append(y_hat)
            print("输出：", ''.join(out))