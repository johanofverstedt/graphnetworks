
# Needed packages: torch, torch_geometric, torch-sparse, torch_scatter, torch_cluster

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataLoader

import numpy as np

import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
import torch_geometric.nn

def main():
    dataset = MNISTSuperpixels(root='~/MNISTSuperpixels', train=True)
    dataset_test = MNISTSuperpixels(root='~/MNISTSuperpixels', train=False)
    
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = GCNConv(dataset.num_node_features, 32)
            self.conv2 = GCNConv(32, 16)
            self.conv3 = GCNConv(16, dataset.num_classes)
            #self.conv1 = SAGEConv(dataset.num_node_features, 32)
            #self.conv2 = SAGEConv(32, 16)
            #self.conv3 = SAGEConv(16, dataset.num_classes)

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch

            x = self.conv1(x, edge_index)
            x = F.relu(x)
            #x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = self.conv3(x, edge_index)
            x = torch_geometric.nn.global_max_pool(x, batch)

            return F.log_softmax(x, dim=1)

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    for batch in loader:
        batch
        print(batch)
        print(batch.y[0])
        print(np.array(batch.y))
        break

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    #data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    model.train()
    n_train_correct = 0
    n_train_count = 0
    for epoch in range(200):
        for batch in loader:
            dat = batch.to(device)
            optimizer.zero_grad()
            out = model(dat)
            loss = F.nll_loss(out, dat.y)
            loss.backward()
            optimizer.step()
            n_train_correct += (torch.max(out, 1)[1].view(batch.y.size()) == batch.y).sum().item()
            n_train_count += 32#batch.y.size()
        dev_acc = 100. * n_train_correct / n_train_count
        print('Train accuracy: ', dev_acc)

         # calculate accuracy on validation set
        n_dev_correct = 0
        n_count = 0
        with torch.no_grad():
            for batch in loader_test:
                data = batch.to(device)
                answer = model(data)
                n_dev_correct += (torch.max(answer, 1)[1].view(batch.y.size()) == batch.y).sum().item()
                n_count += 1
        dev_acc = 100. * n_dev_correct / n_count
        print('Test accuracy: ', dev_acc)

main()

