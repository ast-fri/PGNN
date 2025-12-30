import os
from random import seed
import sys
import gc
import math
import time
import copy
import statistics

import numpy as np
import scipy
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import differential_evolution, Bounds

import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.nn import MessagePassing
from torch_geometric.datasets import Planetoid, Reddit
import torch_geometric.datasets as tg
from torch_geometric.utils import to_undirected

from ogb.nodeproppred import PygNodePropPredDataset



def compute_masks(datasetname, dataset):
    data = dataset[0]
    if datasetname.startswith("ogbn-"):
        split_index = dataset.get_idx_split()
        train_mask = torch.full((data.num_nodes,), False, dtype=torch.bool)
        val_mask = torch.full((data.num_nodes,), False, dtype=torch.bool)
        test_mask = torch.full((data.num_nodes,), False, dtype=torch.bool)
        train_mask[split_index["train"]] = True
        test_mask[split_index["test"]] = True
        val_mask[split_index["valid"]] = True
    else:
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
    return data, train_mask, val_mask, test_mask

def load_dataset(datasetname):
    if datasetname == "reddit":
        dataset = Reddit(root='./dataset/reddit/')
    elif datasetname == "ogbn-arxiv":
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor(remove_edge_index=False), root="../dataset")
    elif datasetname == "ogbn-products":
        dataset = PygNodePropPredDataset(name='ogbn-products', transform=T.ToSparseTensor(remove_edge_index=False), root="./dataset")
    else:
        dataset = Planetoid(root='./dataset', name=datasetname, split='public')
    return dataset
def runner_SGC(datasetname, learn_rate, l2_param, ratio, use_rank_approx, file_name, device, num_runs, seed, num_epochs=500):
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = load_dataset(datasetname)
    data, train_mask, val_mask, test_mask = compute_masks(datasetname, dataset)
    print(data)
    # device = torch.device('cpu')
    device = torch.device("cuda:" + str(device)) if torch.cuda.is_available() else torch.device("cpu")
    print("device: ",device)
    class MyMessagePassing(MessagePassing):
        def __init__(self, in_channels: int, out_channels: int, bias: bool = True, **kwargs):
            super().__init__(aggr='add')  # "Add" aggregation (Step 5).
            self.lin = Linear(in_channels, out_channels, bias=False)
            self.batch_norm = torch.nn.BatchNorm1d(out_channels)
            self.bias = Parameter(torch.empty(out_channels))

            self.init()

        def init(self):

            stdv = 1. / math.sqrt(self.lin.weight.size(1))
            self.lin.weight.data.uniform_(-stdv, stdv)
            self.bias.data.zero_()

        def reset_parameters(self):
            self.lin.reset_parameters()
            self.bias.data.zero_()

        def forward(self, x, Sc, m_fact, training=True, do_batch_norm = False):
            if training:
                out = self.lin(torch.matmul(Sc, x) + m_fact)
                out += self.bias
                
            else:
                x_cpu = x.to('cpu')
                lin_weights_cpu = self.lin.weight.to('cpu')
                lin_bias_cpu = self.bias.to('cpu')
                
                out_cpu = torch.matmul(C, torch.matmul(x_cpu, lin_weights_cpu.t())) + lin_bias_cpu
                out = out_cpu.to(device)
                
                
            if do_batch_norm:
                out = self.batch_norm(out)
            return out

        def message(self, x_j, norm):
            return norm.view(-1, 1) * x_j

    class SGC(torch.nn.Module):
        def __init__(self):
            super().__init__()
            n_neurons =  128
            n_features =  data.x.shape[1]
            n_classes = max(data.y) + 1
            self.conv1 = MyMessagePassing(n_features, n_neurons)
            self.conv2 = MyMessagePassing(n_neurons, n_classes)
            self.reset_parameters()
        def reset_parameters(self):
            self.conv1.reset_parameters()
            self.conv2.reset_parameters()

        def forward(self, S_x, S_c, mfact, mfact1, rho_t, m, x, training=True):
            if training:
                if use_rank_approx == "True":                
                    x1 = self.conv1(S_x, S_c, mfact, training, do_batch_norm = False) + mfact1 
                    x1 = self.conv2(x1, S_c, mfact, training, do_batch_norm = False) + mfact1
                    x1 = F.dropout(x1, p = 0)

                    x1 = rho_t @ (v1 @ x1) + m
                else:
                    x1 = torch.add(self.conv1(S_x, S_c, mfact, training, do_batch_norm = False), mfact1)
                    x1 = torch.add(self.conv2(x1, S_c, mfact, training, do_batch_norm = False), mfact1)
                    x1 = F.dropout(x1, p = 0)
                    x1 = rho_t @ x1 + m
                return x1

            else:
                x1 = self.conv1(x, S_c, mfact, training = False, do_batch_norm = False)
                x1 = self.conv2(x1, S_c, mfact, training = False, do_batch_norm = False)
                return x1

    n = data.x.shape[0]
    d = data.x.shape[1]
    old_num_features = d
    
    print("sketch-ratio is ",ratio)
    k1 = int(ratio*n) - d
    if k1 > 0:
        def generate_pairs(n):
            a = torch.tensor([0]*n)
            tup = torch.stack((a, torch.tensor(list(range(1,n+1)))))
            return tup
        a = torch.randint(0, d, size=(k1,))
        b = torch.randint(0, d, size=(k1,))
        random_pairs = torch.stack((a, b))
        data.x = torch.cat((data.x, torch.mul(data.x[:, random_pairs[0]] ,data.x[:,random_pairs[1]])), dim = 1)
    
    if datasetname == "Pubmed" :
        data.x = F.batch_norm(data.x, running_mean=None, running_var=None, training=True)
    
    n, d = data.x.shape[0], data.x.shape[1]
    print("preprocessing started ==========================================================================================")
    s1 = time.time()
    ps = (data.x - torch.mean(data.x, 1, True))
    #==============================================================================
    ps = (ps / (np.sqrt(int(ratio*n) - 1)))


    print("eigen value computation started")
    [_,_,rho_t] = torch.svd_lowrank((ps.T), q=int(ratio*n), niter=2, M=None)
    data.x = data.x[:, :old_num_features]
    model = SGC().to(device)
    model_memory_usage = torch.cuda.memory_allocated(device)
    S_x = torch.matmul(rho_t.T, data.x - torch.mean(data.x, 1, True))
    mval = torch.mean(data.x, 1, True).to(device)
    S_x = S_x.to(device)
    gcn_norm = torch_geometric.nn.conv.gcn_conv.gcn_norm
    C1, C2 = gcn_norm(data.edge_index, add_self_loops = True)
    C = (torch.sparse_coo_tensor(C1,C2)).to(torch.float32)
    mfact =  rho_t.T @ (C @ torch.mean(data.x, 1, True))
    mfact1 = -1*rho_t.T @ torch.mean(data.x, 1, True)
    S_c =  rho_t.T @ (torch.matmul(C, rho_t))
    if use_rank_approx == "True":
        Beta = min(int(ratio*n), int(0.1*n), int(12000))
        # Beta = min(int(ratio*n), int(0.1*n), int(1500))
        res = torch.svd_lowrank(rho_t, q = Beta, niter = 2)
        row_indices = torch.arange(Beta)
        col_indices = torch.arange(Beta)
        v1 =  res[2].T
        rho_t = res[0] @ torch.sparse_coo_tensor(torch.stack([row_indices, col_indices]), res[1], size=(Beta, Beta)).to_sparse() 
        v1 = v1.to(device)
            
    data.y = data.y.to(device)  
    rho_t = rho_t.to(device)
    S_c = S_c.to(device)
    mfact = mfact.to(device)
    mfact1 = mfact1.to(device)
    
    
    print("preprocessing ended================================================================================================")
    print("time taken is ", time.time() - s1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=l2_param)
    print("Graph Convolutional Network (GCN):")
    SGC()
    def compute_accuracy(pred_y, y):
        return (pred_y == y).sum()
    model.train()
    losses = []
    accuracies = []
    val_losses = []
    val_accuracies = []
    # Initialize variables for early stopping
    best_val_acc = 0.0
    best_epoch = 0
    best_model = None
    best_val_acc = 0.0
    train_time = []
    all_mem = []
    for run in range(num_runs):
        training = True
        model.train()
        for epoch in range(num_epochs):
            se = time.time()
            optimizer.zero_grad()
            out = model(S_x, S_c, mfact, mfact1, rho_t, mval, data.x, training = training)
            loss = F.cross_entropy(out[train_mask], data.y[train_mask])
            # train_loses.append(loss.item())
            correct = compute_accuracy(out.argmax(dim=1)[train_mask], data.y[train_mask])
            acc = int(correct) / int(train_mask.sum())
            losses.append(loss.item())
            val_loss = F.cross_entropy(out[val_mask], data.y[val_mask])
            # val_loses.append(val_loss.item())
            val_correct = compute_accuracy(out.argmax(dim=1)[val_mask], data.y[val_mask])
            val_acc = int(val_correct) / int(val_mask.sum())
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = copy.deepcopy(model)
            loss.backward()
            optimizer.step()
            train_memory_usage = torch.cuda.memory_allocated(device) -   model_memory_usage
            all_mem.append(train_memory_usage)
            t1_stop = time.time() - se
            if run == 0:
                train_time.append(t1_stop)
            if (epoch+1) % 100 == 0:
                print('Epoch: {}, Loss: {:.4f}, Training Acc: {:.4f}, Validation Accuracy: {:.4f}'.format(epoch+1, loss.item(), acc, val_acc))

        del model
        gc.collect()
        model = SGC().to(device)
        # model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=l2_param)
        training = False
        # model.eval()

        # Use the best model for test prediction
        if best_model is not None:
            best_model.eval()

            pred = best_model(S_x, S_c, mfact, mfact1, rho_t, mval, data.x, training=False).argmax(dim=1)
            correct = compute_accuracy(pred[test_mask], data.y[test_mask])
            acc = int(correct) / int(test_mask.sum())
            # runs with less accuracy are discarded
            if acc >= 0.5:
                accuracies.append(acc)
            print(f'Best Model Test Accuracy: {acc:.4f}')
        else:
            print('No best model found.')
    print(accuracies)
    # try:
    print(statistics.mean(accuracies),"+-", statistics.stdev(accuracies))
    print("average training time per epoch is ", statistics.mean(train_time), len(train_time))
    print("peak memory usage (MB) is ", max(all_mem)/1048576.0)

    my_float_list = [statistics.mean(accuracies),"+-", statistics.stdev(accuracies), learn_rate, l2_param]

    float_list_as_string = ' '.join(map(str, my_float_list))

    #  Save output to a .txt file
    # output_filename = "output_results.txt"
    with open(file_name, "w") as file:
        file.write("Accuracies: " + str(accuracies) + "\n")
        file.write("Mean Accuracy: " + str(statistics.mean(accuracies)) + " +- " + str(statistics.stdev(accuracies)) + "\n")
        file.write("Average Training Time per Epoch: " + str(statistics.mean(train_time)) + " (Total epochs: " + str(len(train_time)) + ")\n")
        file.write("Peak Memory Usage (MB): " + str(max(all_mem)/1048576.0) + "\n")
        file.write("Summary: " + float_list_as_string + "\n")

    print(f"Results saved to {file_name}")
   