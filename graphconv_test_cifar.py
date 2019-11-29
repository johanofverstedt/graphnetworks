
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

import pickle

batch_size = 4
test_batch_size = 128
learning_rate = 0.001
weight_decay = 5e-5

def test_load_cifar_graph(path):
    with open(path, 'rb') as handle:
        graphs = pickle.load(handle)
    print(graphs[1].y)
    print(len(graphs))
    return graphs

def make_conv(f_in, f_out):
    #return GCNConv(f_in, f_out)
    return SAGEConv(f_in, f_out)

def make_pool(f_in):
    return torch_geometric.nn.TopKPooling(f_in, ratio=0.5)

def main():
    #dataset = MNISTSuperpixels(root='~/MNISTSuperpixels', train=True)
    #dataset_test = MNISTSuperpixels(root='~/MNISTSuperpixels', train=False)

    dataset = test_load_cifar_graph('/home/johan/cifar.pickle')
    dataset_test = test_load_cifar_graph('/home/johan/cifar_test.pickle')
    num_node_features = 3
    num_classes = 10
    
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            #self.conv1 = GCNConv(num_node_features, 32)
            #self.conv2 = GCNConv(32, 16)
            #self.conv3 = GCNConv(16, num_classes)
            self.conv1 = make_conv(num_node_features, 32)
            self.pool1 = make_pool(32)#torch_geometric.nn.TopKPooling(32, ratio=0.5)
            self.conv2 = make_conv(32, 16)
            self.pool2 = make_pool(16)#torch_geometric.nn.TopKPooling(16, ratio=0.5)
            self.conv3 = make_conv(16, num_classes)
            self.pool3 = make_pool(num_classes)
            #self.pool3 = torch_geometric.nn.TopKPooling(num_classes, ratio=0.5)

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch

            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x, edge_index, _, batch, _, _ = self.pool1(x=x, edge_index=edge_index, batch=batch)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x, edge_index, _, batch, _, _ = self.pool2(x=x, edge_index=edge_index, batch=batch)
            x = self.conv3(x, edge_index)
            x, edge_index, _, batch, _, _ = self.pool3(x=x, edge_index=edge_index, batch=batch)
            x = torch_geometric.nn.global_mean_pool(x, batch)

            return F.log_softmax(x, dim=1)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=test_batch_size, shuffle=False)

    for batch in loader:
        batch
        print(batch)
        print(batch.y[0])
        print(np.array(batch.y))
        break

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    #data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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
            n_train_count += batch_size#batch.y.size()
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
                n_count += test_batch_size
        dev_acc = 100. * n_dev_correct / n_count
        print('Test accuracy: ', dev_acc)

main()

