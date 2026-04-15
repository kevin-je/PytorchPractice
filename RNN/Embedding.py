import torch
import torch.nn as nn
import torch.optim as optim

epochs = 100
batch_size = 1
num_layers = 2
num_class = 4
seq_len = 5
vocab_size = 4
embedding_dim = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#构建数据集
letter = ['h', 'e', 'l', 'o']
input_seq = torch.LongTensor([0, 1, 2, 2, 3]).to(device)
target_seq = torch.LongTensor([1, 2, 0, 3, 2]).to(device)

#构建网络
class RNN_embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=num_layers,
        )
        self.linear = nn.Linear(in_features=embedding_dim, out_features=num_class)

    def forward(self, _x, _h):
        embedded = self.embedding(_x)
        _output, _ = self.rnn(embedded, _h)
        logits = self.linear(_output)
        return logits

#构建损失函数和优化器
model = RNN_embedding().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

if __name__ == "__main__":

    #构建训练循环
    for epoch in range(epochs):
        print("=" * 6 + f"当前轮数：{epoch + 1}" + "=" * 6)
        model.train()

        h0 = torch.zeros(num_layers, embedding_dim, device=device)
        output = model(input_seq, h0)

        loss = criterion(output, target_seq)
        print(f"总损失：, {loss.item():.2f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #评估
        model.eval()
        with torch.no_grad():
            out = []
            h = model(input_seq, h0)
            _, idx = h.max(1)
            y_hat = [letter[i] for i in idx]
            print("输出：", ''.join(y_hat))