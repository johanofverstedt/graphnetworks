
# Code for preprocessing an image dataset into a graph dataset
# Author: Nadezhda Koriakina

# Imports

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import cKDTree as KDTree
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

import torch
import torch_geometric.data
from torchvision.datasets import CIFAR10
#from tqdm import tqdm

import pickle

# Parameters
dmax = 1.5

n_segments = 100
compactness = 20

def process_image_nn_based_on_radius(img, img_class):
    img = np.asarray(img)
    img_height, img_width, ch = img.shape
    col, row = np.meshgrid(np.arange(img_height), np.arange(img_width))
    coord = np.stack((col, row), axis=2).reshape(-1, 2) 

    #dmax: 8 neighbours; 1: 4 neighbours (with Euclidean distance)
    kdT = KDTree(coord)
    res = kdT.query_pairs(dmax)
    res = [(x[0],x[1]) for x in list(res)]
    
    res = np.transpose(res)

    ### Create a graph
    #G = nx.Graph()
    #for i in range(coord.shape[0]):
    #    G.add_node(i, intensity=img[coord[i,0], coord[i,1]], test=False, val=False, label=0)
    #G.add_edges_from(res)

    ### Add nodes
    x = torch.Tensor(img.reshape(img_height*img_width, ch))
    #G.edges()

    edge_index = torch.LongTensor(res)

    D = torch_geometric.data.Data(x = x, edge_index = edge_index, y=img_class)
    return D

# test on CIFAR

def export_cifar(path, tmp_path, train=True, max_images=10000):
    dataset = CIFAR10(tmp_path, train, download=True)
    print(dataset[1][1])

    graphs = []

    cnt = 0
    for (data, yy) in dataset:
        #print(data)
        #print(yy)
        img = data
        y = yy

        graph = process_image_nn_based_on_radius(img, y)
        graphs.append(graph)
        cnt += 1
        if cnt == max_images:
            break

    with open(path, 'wb') as handle:
        pickle.dump(graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)

def test_load_cifar_graph(path):
    with open(path, 'rb') as handle:
        graphs = pickle.load(handle)
    print(graphs[1].y)
    print(len(graphs))
    return graphs

export_cifar('/home/johan/cifar.pickle', '~/tmp_cifar/', train=True, max_images=70000)
export_cifar('/home/johan/cifar_test.pickle', '~/tmp_cifar/', train=False, max_images=70000)
graphs = test_load_cifar_graph('/home/johan/cifar.pickle')



def preprocess_image_superpixel(img, img_class):
    segments = slic(img, n_segments=100, compactness=20)
    ### Find neighbours of each segment (edges); based on https://peekaboo-vision.blogspot.com/2011/08/region-connectivity-graphs-in-python.html
    down = np.c_[segments[:-1, :].ravel(), segments[1:, :].ravel()]
    right = np.c_[segments[:, :-1].ravel(), segments[:, 1:].ravel()]

    all_edges = np.vstack([right, down])
    # Remove edges with the same labels
    all_edges = all_edges[all_edges[:, 0] != all_edges[:, 1], :]
    edges = np.unique(all_edges, axis=0)

    ### Create a graph
    G_superp = nx.Graph()
    G_superp['y'] = img_class
    segments_ids = np.unique(segments)

    ### Compute average color values for each superpixel
    def mean_image(image,label): # based on https://stackoverflow.com/questions/41578473/how-to-calculate-average-color-of-a-superpixel-in-scikit-image
        mean_intensity=[]
        im_rp=image.reshape((image.shape[0]*image.shape[1],image.shape[2]))
        sli_1d=np.reshape(label,-1)    
        uni=np.unique(sli_1d)
        uu=np.zeros(im_rp.shape)
        for i in uni:
            loc=np.where(sli_1d==i)[0]
            #print(loc)
            mm=np.mean(im_rp[loc,:],axis=0)
            mean_intensity.append(mm)
            uu[loc,:]=mm
        oo=np.reshape(uu,[image.shape[0],image.shape[1],image.shape[2]]).astype('uint8')
        return mean_intensity
    mean_intensity = mean_image(np.asarray(img),segments)

    ### Add nodes
    for i in range(len(segments_ids)):
        G_superp.add_node(segments_ids[i], intensity=mean_intensity[i], test=False, val=False, label=0)

    ### Add edges
    G_superp.add_edges_from(list(map(tuple, edges)), neighbouring_superpixels=True)
    return  G_superp

"""
def test1():
    img = Image.open('sample.jpg')
    plt.imshow(img)
    plt.show()
    img = img.resize((28,28))
    plt.imshow(img)
    plt.show()

    process_image_nn(img, 1.5)
"""
def test2():
    pass

#test1()