import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence as pack

import numpy as np
import matplotlib.pyplot as plt

import gzip, csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 100
batch_size = 256
num_layers = 2
num_embeddings = 128
embedding_dim = 100
hidden_size = 100
num_class = None
num_workers = 8

#构建数据集
class NameDataset(Dataset):
    def __init__(self, path):
        super(NameDataset, self).__init__()

        with gzip.open(path, 'rt') as f:
            csv_reader = csv.reader(f, delimiter=',')
            data_rows = list(csv_reader)

            self.names = [row[0] for row in data_rows]
            self.countries = [row[1] for row in data_rows]
            self.max_seq_len = max(len(name) for name in self.names)

            self.countries_lst = sorted(set(self.countries))
            self.num_class = len(self.countries_lst)

            self.target = torch.LongTensor([
                self.make_countries_dict().get(key)
                for key in self.countries
            ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        return self.names[idx], self.target[idx]

    def make_countries_dict(self):
        keys = self.countries_lst
        values = [i for i in range(self.num_class)]
        countries_dict = {k: v for k, v in zip(keys, values)}
        return countries_dict

#设计网络
class GRU(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, num_layers, num_class):
        super(GRU, self).__init__()
        # self.num_embeddings = num_embeddings
        # self.embedding_dim = embedding_dim
        self.hidden_size:int = hidden_size
        self.num_layers:int = num_layers

        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_class)

    def generate_h0(self, _batch_size):
        h0 = torch.zeros(self.num_layers*2, _batch_size, self.hidden_size)
        return h0.to(device)

    def forward(self, _x, seq_lens):
        _batch_size = _x.size(1)
        h0 = self.generate_h0(_batch_size)
        emb = self.emb(_x)
        gru_input = pack(emb, seq_lens.cpu(), enforce_sorted=False)
        output, hn = self.gru(gru_input, h0)
        h_total = torch.cat((hn[-1], hn[-2]), dim=1)
        output = self.fc(h_total)
        return output

def get_tensors(names, max_name_len):
    _batch_size = len(names)
    tensors = torch.zeros(max_name_len, _batch_size).long()
    for idx, name in enumerate(names):
        ascii_codes = [ord(c) for c in name]
        tensors[:len(name), idx] = torch.LongTensor(ascii_codes)
    return tensors


if __name__ == "__main__":

    train_dataset = NameDataset(
        'C:/Users/asus/PycharmProjects/PytorchPractice/NameDataset/names_train.csv.gz'
    )
    max_name_len1 = train_dataset.max_seq_len
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_dataset = NameDataset(
        'C:/Users/asus/PycharmProjects/PytorchPractice/NameDataset/names_test.csv.gz'
    )
    max_name_len2 = test_dataset.max_seq_len
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    num_class = train_dataset.num_class
    max_name_len = max(max_name_len1, max_name_len2)

    # 构建损失函数和优化器
    model = GRU(num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_class=num_class).to(device)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    accuracy_lst = []
    accuracy_max = 0

    # 构建训练循环
    for epoch in range(epochs):
        print("\n" + "=" * 6 + f"当前轮数：{epoch + 1}" + "=" * 6)

        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data_tensors = get_tensors(data, max_name_len=max_name_len)
            data_tensors, target = data_tensors.to(device), target.to(device)

            _batch_size = data_tensors.shape[1]
            seq_lens = (data_tensors != 0).sum(dim=0)

            output = model(data_tensors, seq_lens=seq_lens)

            loss = criterion(output, target)
            if batch_idx % 10 == 9:
                print(
                    f"训练集第{epoch+1}轮第{batch_idx+1}批的平均损失：{loss.item():.2f}"
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 评估
        model.eval()
        with torch.no_grad():

            corrects = 0

            for batch_idx, (data, target) in enumerate(test_loader):

                data_tensors = get_tensors(data, max_name_len=max_name_len)
                data_tensors, target = data_tensors.to(device), target.to(device)

                _batch_size = data_tensors.shape[1]
                seq_lens = (data_tensors != 0).sum(dim=0)

                output = model(data_tensors, seq_lens=seq_lens)

                output_idx = output.argmax(dim=1)
                corrects += (output_idx == target).sum().item()

            accuracy = corrects / len(test_dataset) * 100
            accuracy_lst.append(accuracy)
            print(f"测试集的准确率：{accuracy:.2f}%")

        # 存储最佳模型参数
        if accuracy > accuracy_max:
            accuracy_max = accuracy
            torch.save(model.state_dict(), "GRU_names.pth")

    # 结果可视化
    plt.rcParams['font.sans-serif'] = ['SimHei']
    x_date = np.arange(epochs) + 1
    y_data = np.array(accuracy_lst)
    plt.xlabel("训练轮次")
    plt.ylabel(r"准确率（$\text{%}$）")
    plt.ylim(70, accuracy_max+5 if accuracy_max <= 95 else accuracy_max)
    plt.title("GRU在NAMES数据集上的准确率")
    plt.grid(True)
    plt.plot(x_date, y_data, color="r")
    plt.savefig("GRU_names.png")
    plt.show()