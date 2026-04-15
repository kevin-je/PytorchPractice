import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

epochs = 100
batch_size = 1
num_layers = 3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#构建数据集
letter = ['h', 'e', 'l', 'o']
input_seq = torch.LongTensor([0, 1, 2, 2, 3]).to(device)
target_seq = torch.LongTensor([1, 2, 0, 3, 2]).to(device)

input_seq_hot = F.one_hot(input_seq, num_classes=4).float()

#构建网络
class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=4, hidden_size=4, num_layers=num_layers)

    def forward(self, _x, _h):
        _output, _ = self.rnn(_x, _h)
        return _output

#构建损失函数和优化器
model = RNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

if __name__ == "__main__":

    #构建训练循环
    for epoch in range(epochs):
        print("=" * 6 + f"当前轮数：{epoch + 1}" + "=" * 6)
        model.train()

        h0 = torch.zeros(num_layers, 4, device=device)
        output = model(input_seq_hot, h0)

        loss = criterion(output, target_seq)
        print(f"总损失：, {loss.item():.2f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #评估
        model.eval()
        with torch.no_grad():
            out = []
            h = model(input_seq_hot, h0)
            _, idx = h.max(1)
            y_hat = [letter[i] for i in idx]
            print("输出：", ''.join(y_hat))