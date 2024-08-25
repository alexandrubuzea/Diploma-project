import torch
import torch.nn as nn
from torch.nn import LSTM
import matplotlib.pyplot as plt
from gen_data.data_generation import load_graphs
from torch.autograd import Variable

class ColoringNN(nn.LSTM):
    def __init__(self, in_size : int, out_size : int,
                  hidden_size : int, num_layers : int, num_nodes : int):
        super(ColoringNN, self).__init__(input_size=in_size, hidden_size=hidden_size)

        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_nodes = num_nodes

        self.cells = nn.ModuleList([
            LSTM(
                input_size = in_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
            for _ in range(num_nodes)
        ])

        self.linear = nn.Linear(in_features=hidden_size, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        outputs = []

        for i, lstm in enumerate(self.cells):
            out, _ = lstm(inputs[:, [i], :])

            out = self.linear(out)
            out = torch.flatten(input=out, start_dim=0)
            out = self.relu(out)

            outputs.append(out.unsqueeze(-1))

        res = torch.cat(outputs, dim=1)
        return res
        
TRAIN_SIZE = 10000
TEST_SIZE = 2000

RANDOM = False # if choosing the random dataset, false otherwise
RANDOM_SHUFFLE = True # if shuffling the nodes at loading, False for not shuffling

BATCH_SIZE = 128
LEARNING_RATE = 0.000001
EPOCHS = 200
IN_SIZE = 100
OUT_SIZE = 100
HIDDEN_SIZE = 256
NUM_NODES = 100

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            sample = self.transform(sample)

        return sample, label

import numpy as np

def custom_random_shuffle(train_data, train_colorings):
    n = len(train_colorings[0])

    for index, (example, coloring) in enumerate(zip(train_data, train_colorings)):
        permutation = list(np.random.permutation(n))

        new_example = [[example[permutation[i]][permutation[j]] for j in range(n)] for i in range(n)]
        new_coloring = [coloring[permutation[i]] for i in range(n)]

        train_data[index] = new_example
        train_colorings[index] = new_coloring

    return train_data, train_colorings

train_data, train_colorings = load_graphs("train_random" if RANDOM else "train_clique", TRAIN_SIZE)

# train_data, train_colorings = custom_random_shuffle(train_data, train_colorings)

train_loader = torch.utils.data.DataLoader(dataset=CustomDataset(data=train_data, labels=train_colorings), batch_size=BATCH_SIZE, shuffle=True)

print("Loaded training set")

test_data, test_colorings = load_graphs("test_random" if RANDOM else "test_clique", TEST_SIZE)
test_loader = torch.utils.data.DataLoader(dataset=CustomDataset(data=test_data, labels=test_colorings), batch_size=BATCH_SIZE, shuffle=False)

print("Loaded test set")

model = ColoringNN(in_size=IN_SIZE, out_size=OUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=3, num_nodes=NUM_NODES)
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

print("Created model, optimizer, criterion")

use_cuda = torch.cuda.is_available()

if use_cuda:
    model = model.cuda()

train_losses = []
test_losses = []

print("Beginning of training")

for epoch in range(EPOCHS):
    model.train()

    avg_loss = 0

    for index, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()

        if use_cuda:
            x = x.cuda()
            target = target.cuda()
        
        x = Variable(x)
        target = Variable(target)

        outputs = model(x)

        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()

        avg_loss = 0.9 * avg_loss + 0.1 * loss.item()

    train_losses.append(avg_loss)

    print(f"Finished training for epoch {epoch}")
    print(f"Training loss: {avg_loss}")

    model.eval()

    avg_loss = 0

    for index, (x, target) in enumerate(test_loader):
        if use_cuda:
            x = x.cuda()
            target = target.cuda()
        
        x = Variable(x)
        target = Variable(target)

        outputs = model(x)

        loss = criterion(outputs, target)

        avg_loss = 0.9 * avg_loss + 0.1 * loss.item()

    test_losses.append(avg_loss)

    print(f"Finished test for epoch {epoch}")
    print(f"Test loss: {avg_loss}")

torch.save(model, 'model_base2.pth')