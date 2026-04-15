import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

import numpy as np
import matplotlib.pyplot as plt

import gzip
import csv

epochs = 100
batch_size = 32
num_embeddings = 128
embedding_dim = 32
hidden_size = 64
num_layers = 2
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 8

train_path = "D:\\PycharmProject\\PytorchPractice\\NameDataset\\names_train.csv.gz"
test_path = "D:\\PycharmProject\\PytorchPractice\\NameDataset\\names_test.csv.gz"

class NameDataset(Dataset):
    def __init__(self, filename):
        self.num_countries = None
        self.countries_dir = None
        self.names_len_max = None
        self.countries = None
        self.names = None
        self.filename = filename

        self.open_file()
        self.make_countries_dir()

    def open_file(self):
        names, countries = [], []
        with gzip.open(self.filename, 'rt') as f:
            reader = csv.reader(f)
            for row in reader:
                names.append(row[0])
                countries.append(row[1])

        self.names = names
        self.countries = countries

        names_len = [len(name) for name in names]
        self.names_len_max = max(names_len)

        # return self.names, self.countries

    def make_countries_dir(self):
        countries_lst = list(set(self.countries))
        num_countries = len(countries_lst)
        dir_countries = {}
        for i in range(num_countries):
            dir_countries[countries_lst[i]] = i

        self.countries_dir = dir_countries
        self.num_countries = num_countries

    @staticmethod
    def name_encoding(name):
        name_encoded = [ord(c) for c in name]
        return name_encoded

    def __getitem__(self, index):
        feature = self.name_encoding(self.names[index])
        label = self.countries_dir[self.countries[index]]

        return torch.LongTensor(feature), label

    def __len__(self):
        return len(self.names)

def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded_seqs = pad_sequence(sequences, padding_value=0)
    lengths = torch.tensor([len(seq) for seq in sequences])

    return padded_seqs, lengths, torch.LongTensor(labels)

class LSTM(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, num_layers, num_countries):
        super(LSTM, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers

        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            bidirectional=True,
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_size*2, num_countries)

    def forward(self, x, lengths):
        h0 = torch.zeros(2*self.num_layers, x.size(1), self.hidden_size).to(device)
        c0 = torch.zeros(2*self.num_layers, x.size(1), self.hidden_size).to(device)

        embedded = self.embedding(x)

        packed = pack_padded_sequence(embedded, lengths, enforce_sorted=False)

        _, (hn, cn) = self.lstm(packed, (h0, c0))
        forward, backward = hn[-2], hn[-1]
        y = torch.cat([backward, forward], 1)
        logits = self.fc(y)

        return logits

if __name__ == "__main__":


    train_dataset = NameDataset(train_path)
    test_dataset = NameDataset(test_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    model = LSTM(
        num_embeddings,
        embedding_dim,
        hidden_size,
        num_layers,
        train_dataset.num_countries
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    accuracy_lst = []
    best_accuracy = 0

    for epoch in range(epochs):
        correct = 0
        print("\n" + "=" * 6 + f"当前轮数：{epoch + 1}" + "=" * 6)

        model.train()

        for batch_idx, (features, lengths, labels) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features, lengths)
            loss = criterion(logits, labels)

            if batch_idx % 10 == 9:
                print(
                    f"训练集第{epoch+1}轮第{batch_idx+1}批的平均损失：{loss.item():.2f}"
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():

            for batch_idx, (features, lengths, labels) in enumerate(test_loader):
                features = features.to(device)
                labels = labels.to(device)

                logits = model(features, lengths)
                countries_index = logits.argmax(-1)
                correct += torch.sum(torch.eq(labels, countries_index))

            accuracy = (100 * correct / len(test_dataset.names)).item()
            accuracy_lst.append(accuracy)

            print(f"测试集的准确率：{accuracy:.2f}%")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), "LSTM_names.pth")

    plt.rcParams['font.sans-serif'] = ['SimHei']
    x_date = np.arange(epochs) + 1
    y_data = np.array(accuracy_lst)
    plt.xlabel("训练轮次")
    plt.ylabel(r"准确率（$\text{%}$）")
    plt.title("LSTM在NAMES数据集上的准确率")
    plt.grid(True)
    plt.plot(x_date, y_data, color="r")
    plt.savefig("LSTM_names.png")
    plt.show()