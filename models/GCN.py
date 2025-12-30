import os
from random import seed
import sys
import gc
import math
import time
import copy
import statistics
from torch_sparse import SparseTensor
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
import torch.nn as nn
from ogb.nodeproppred import PygNodePropPredDataset
from torch.nn import LayerNorm
from sketch import CountSketch, TensorSketch
from sketch import DEFAULT_COUNT_SKETCH_CONFIG
from sketch import DEFAULT_TENSOR_SKETCH_CONFIG
def runner_GCN(datasetname, learn_rate, l2_param, ratio, use_rank_approx, num_layers, hidden_dim, dropout, layer_norm, ratio2, order, device,lr_update_weight, random_sample_size,epochs, runs, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = datasetname
    dataset = Planetoid(root='../Project_sketch/dataset', name=dataset, split = 'public')
    data = dataset[0]
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    print(data)
    gcn_norm = torch_geometric.nn.conv.gcn_conv.gcn_norm
    C1, C2 = gcn_norm(data.edge_index, add_self_loops = True)
    C = (torch.sparse_coo_tensor(C1,C2)).to(torch.float32)
    device = torch.device('cuda:'+str(device))
    n = data.x.shape[0]
    class MyMessagePassing(MessagePassing):
        def __init__(self, in_channels: int, out_channels: int, order : int, bias: bool = True, **kwargs):
            super().__init__(aggr='add')  # "Add" aggregation (Step 5).
            self.order = order
            self.weight = Parameter(torch.empty((in_channels, out_channels), dtype=torch.float32), requires_grad=True)
            self.bias = Parameter(torch.empty(out_channels, dtype=torch.float32), requires_grad=True)
            self.coeffs = Parameter(torch.empty(order, dtype=torch.float32), requires_grad=True)
            # self.ln = torch.nn.LayerNorm((out_channels,))
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            # mean = 0.0  # Centered at zero
            # std = 0.2
            # torch.nn.init.normal_(self.weight, mean=mean, std=std)

            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            # self.weight = torch.empty(data.x.shape[1], int(ratio*n))  # Example tensor of shape (3, 5)
            
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            self.coeffs.data.fill_(0)
            self.coeffs.data[0] = 1.0
        def forward(self, nf_mats, conv_mats, rho_meanpair = None, use_rank_approx = True, weight2 = None, cslist = None, training = None, layer_return = None, grad_turnon = False):
            
            if training == True:
                if use_rank_approx == "False":
                    zs = conv_mats @ (nf_mats @ self.weight + self.bias)
                    zs = (sum([self.coeffs[degree - 1] * torch.pow(zs, degree)        for degree in range(1, self.order + 1)])) 
                    
                    
                    
                    zs = weight2.T @ (rho_t.T @ (zs - mval))
                    if layer_return == True:
                        zl = zs
                    zs = rho_t @ (weight2 @ (zs)) + mval

                    if layer_return == True:
                        return zs, zl
                    else:
                        return zs

                else:
                    zs = [torch.fft.rfft((rho_meanpair[0][0] @ (weight2 @ nf_mats) + rho_meanpair[0][1]) @ self.weight + self.bias, dim=0)]

                    for degree in range(1, self.order):
                        zs.append(torch.fft.rfft((rho_meanpair[degree][0] @ (weight2 @ nf_mats) + rho_meanpair[degree][1]) @ self.weight + self.bias, dim=0))
                        zs[-1] = zs[-1] * zs[-2]
                    zs = list(map(lambda _: torch.fft.irfft(_, dim=0), zs))
                    sumv = 0
                    # for cs in conv_mats: 
                    for degree in (range(1, self.order+1)):
                        sumv += self.coeffs[degree - 1] * (weight2.T @ (conv_mats[degree-1] @ zs[degree-1]))
                    sumv = sumv + weight2 @ mfact1
                    
                    if layer_return == True:
                        return sumv, sumv
                    else:
                        return sumv
                
            elif training == "Baseline":
                    zs = conv_mats @ (nf_mats @ self.weight + self.bias)
                    return F.relu(zs)    
                
            else:
                zs = conv_mats @ (nf_mats @ self.weight + self.bias)
                return sum([self.coeffs[degree - 1] * torch.pow(zs, degree)        for degree in range(1, self.order + 1)])
                    
            
        def message(self, x_j, norm):
            return norm.view(-1, 1) * x_j
    class GCN(torch.nn.Module):
        def __init__(self, hidden_dim, layers, layernorm, dropoutval, order):
            super().__init__()
            n_neurons =  hidden_dim
            n_features =  data.x.shape[1]
            n_classes = max(data.y) + 1
            num_layers = layers
            self.num_layers = num_layers
            batchnorm = layernorm
            self.dropout = dropoutval
            self.batchnorm = batchnorm
            self.norm = LayerNorm(n_neurons, elementwise_affine=True)
            self.ln = False
            self.weight2 = Parameter(torch.empty((rho_t.shape[1], int(ratio*n)), dtype=torch.float32), requires_grad=False)
            self.weight2_optimizer = torch.optim.Adam([self.weight2], lr=0.01)

            if self.batchnorm:
                self.bns = torch.nn.ModuleList()
                self.bns.append(torch.nn.BatchNorm1d(n_neurons))
            self.convs = torch.nn.ModuleList()

            self.convs.append(MyMessagePassing(n_features, n_neurons, order))
            
            for j in range(num_layers-2):
                self.convs.append(MyMessagePassing(n_neurons, n_neurons, order))
                
                if self.batchnorm:
                    self.bns.append(torch.nn.BatchNorm1d(n_neurons))
            
            self.convs.append(MyMessagePassing(n_neurons, n_classes, order))
            # self.ln2 = torch.nn.LayerNorm(n_classes.item())
            self.reset_parameters()
        def reset_parameters(self):
            for conv in self.convs:
                conv.reset_parameters()
            if self.batchnorm:
                for bn in self.bns:
                    bn.reset_parameters()
            self.weight2.data = v1
        def update_weight3(self, target_value, predval, sx, targetx, cmat, learning_rate=lr_update_weight, unsketch_mat = None):
            
            # self.weight2_optimizer.zero_grad()            
        
            weight2_temp = self.weight2.detach().clone().requires_grad_(True)
            # weight2_optimizer = torch.optim.Adam([weight2_temp], lr=learning_rate)
            # weight2_temp.zero_grad()            
        
            
            sx = (unsketch_mat @ weight2_temp @ sx + mval[random_sample])
            
            # c1 = 1/(targetx.shape[0]*targetx.shape[1])
            c1 = 1
            term2 = c1*torch.linalg.norm(sx - targetx, 'fro')    
            loss2 =    term2
            
            loss2.backward()
            # weight2_optimizer.step()
            torch.nn.utils.clip_grad_norm_(weight2_temp, max_norm=1)
            with torch.no_grad():
                self.weight2 -= learning_rate * weight2_temp.grad
            # self.weight2 = torch.nn.Parameter(weight2_temp.detach().clone())
        def forward(self, nf_mats, conv_mats, rho_meanpair = None, training=None, use_rank_approx = True, cslist = None):        
            # exit()
            if training == True:  
                z = []  
                nf_mats, zl = self.convs[0](nf_mats, conv_mats, rho_meanpair, use_rank_approx = use_rank_approx, weight2 = self.weight2, training = training, cslist  = cs, layer_return = True)
                if self.batchnorm:
                    nf_mats = self.bns[0](nf_mats)
                if self.ln:
                    nf_mats = self.norm(nf_mats)
                z.append(zl.detach())
                for i, conv in enumerate(self.convs[1:-1]):
                    nf_mats, zl = conv(nf_mats, conv_mats, rho_meanpair, use_rank_approx = use_rank_approx, weight2 = self.weight2,training = training, cslist = cs, layer_return = True)
                    
                    if self.batchnorm:
                        nf_mats = self.bns[i+1](nf_mats)
                    if self.ln:
                        nf_mats = self.norm(nf_mats)
                    z.append(zl.detach())
                    nf_mats = F.dropout(nf_mats, p=self.dropout, training= training)
                
                nf_mats, zl = self.convs[-1](nf_mats, conv_mats, rho_meanpair, use_rank_approx = use_rank_approx, weight2 = self.weight2, training = training, cslist = cs, layer_return = True, grad_turnon = True)
                
                z.append(zl.detach())
                sumv = 0
                if use_rank_approx == "True":
                    #=====================================================================
                    nf_mats  = rho_t @ (self.weight2 @ nf_mats) + mval
                    #=====================================================================
                return F.log_softmax(nf_mats, dim = 1), z
            
            elif training == "Baseline":
                nf_mats = self.convs[0](nf_mats, conv_mats, training = training)

                if self.batchnorm:
                    nf_mats = self.bns[0](nf_mats)
                for i, conv in enumerate(self.convs[1:-1]):
                    nf_mats = nf_mats + conv(nf_mats, conv_mats, training = training)
                    if self.batchnorm:
                        nf_mats = self.bns[i+1](nf_mats)
                    nf_mats = F.dropout(nf_mats, p=self.dropout, training=self.training)
                nf_mats = self.convs[-1](nf_mats, conv_mats, training)
                return F.log_softmax(nf_mats, dim = 1)
            else:
                nf_mats = self.convs[0](nf_mats, conv_mats, training = training)

                if self.batchnorm:
                    nf_mats = self.bns[0](nf_mats)
                for i, conv in enumerate(self.convs[1:-1]):
                    nf_mats = nf_mats + conv(nf_mats, conv_mats, training = training)
                    if self.batchnorm:
                        nf_mats = self.bns[i+1](nf_mats)
                    nf_mats = F.dropout(nf_mats, p=self.dropout, training=self.training)
                nf_mats = self.convs[-1](nf_mats, conv_mats, training)
                
                return F.log_softmax(nf_mats, dim = 1)

    n = data.x.shape[0]  # Replace with your desired value of n
    d = data.x.shape[1]
    old_num_features = d
    
    print("sketch-ratio is ",ratio)
    k1 = int(ratio*n) - d  
    if k1 > 0:
        Wp = torch.empty(data.x.shape[1], int(ratio*n) - d, dtype=torch.float32, requires_grad=False)
        nn.init.kaiming_uniform_(Wp, a=math.sqrt(5))
        data.x = torch.cat((data.x, data.x @ Wp), dim = 1)
    
    n, d = data.x.shape[0], data.x.shape[1]

    print("preprocessing started ==========================================================================================")
    s1 = time.time()  
    Ctmp = C.to_sparse()
    
    if datasetname == "ogbn-products":
        data.x =  F.batch_norm(data.x, running_mean=None, running_var=None, training=True)



    tmp = (Ctmp @ (data.x))
    mval = torch.mean(tmp, 1, True)
    ps =   F.relu(tmp - torch.mean(tmp, 1, True))
    # ps = (F.relu(ps))
    # mval = torch.mean(ps,1, True)
    
    #==============================================================================
    ps = (ps / (np.sqrt(int(ratio*n) - 1)))
    print("eigen value computation started")
    [_,_,rho_t] = torch.svd_lowrank((ps.T), q=int(ratio*n), niter=2, M=None)
    v1 = torch.eye(int(ratio*n))
    sample_sketch = v1.T @ rho_t.T @ ((data.x[:, :old_num_features]) - mval)
    sample_sketch = sample_sketch.to(device)
    train_indices = np.where(np.array(train_mask))[0]
    val_indices = np.where(np.array(val_mask))[0]
    test_indices = np.where(np.array(test_mask))[0]
  
    train_indices = (list(train_indices))
  
    data.x = data.x[:, :old_num_features]
    model = GCN(hidden_dim = hidden_dim, layers = num_layers, layernorm= layer_norm, dropoutval = dropout, order = order)
    model_memory_usage = torch.cuda.memory_allocated(device)
    if v1.device != rho_t.device:
        v1 = v1.to('cpu')

    S_x = torch.matmul(v1.T @ rho_t.T, data.x - torch.mean(data.x, 1, True))
    
    
    if use_rank_approx == "False":
        S_x = rho_t @ v1 @ S_x + mval

        S_x = S_x.to(device)
    else:
        S_x = S_x.to(device)

    Ctmp = C.clone()
    row_indices = Ctmp.coalesce().indices()[0]
    col_indices = Ctmp.coalesce().indices()[1]
    values = Ctmp.coalesce().values()

    Ctmp = SparseTensor(
        row=row_indices,
        col=col_indices,
        value=values,
        sparse_sizes=C.size()
    )

    if use_rank_approx == "True":
        num_sketches = order
        # sketch_list = []
        n = data.x.shape[0]
        in_dim = int(n)
        ratio2 = ratio2
        out_dim = int(ratio2*n)
        # cs = CountSketch(in_dim, out_dim).to(device)
        
        cs = []
        cslist = []
        for _ in range(num_sketches):
            cs.append(CountSketch(in_dim, out_dim))
            # cslist[0].ht_pytorch_sparse, cslist[0].s
            cslist.append([cs[-1].ht_pytorch_sparse.to(device), (cs[-1].s).to(device)])
        ts = TensorSketch(cs, config = DEFAULT_TENSOR_SKETCH_CONFIG)
        
        Sc = ts.sketch_mat(Ctmp)
        for i in range(len(Sc)):
            Sc[i] = rho_t.T @ Sc[i].T
            Sc[i] = Sc[i].to(device)
        del Ctmp
    else:
        cs = []
        cslist = []    
    mfact =  rho_t.T @ (C @ torch.mean(data.x, 1, True))
    mfact1 = -rho_t.T @ torch.mean(data.x, 1, True)    
    data.y = data.y.to(device)  
    # S_c = S_c.to(device)
    mfact = mfact.to(device)
    mfact1 = mfact1.to(device)
    print("preprocessing ended================================================================================================")
    print("time taken is ", time.time() - s1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=l2_param)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    print("Graph Convolutional Network (GCN):")
    def compute_accuracy(pred_y, y):
        return (pred_y == y).sum()
    model.train()
    losses = []
    accuracies = []
    if n < 1e5:
        num_epochs = epochs
    else:
        num_epochs = epochs

    # data.x = data.x.to(device)
    val_losses = []
    val_accuracies = []
    # Initialize variables for early stopping
    best_val_acc = 0.0
    best_epoch = 0

    num_runs = runs

    if use_rank_approx == "False":
        rho_meanpair  = []

    # Note ### The degree is kept the same as the num of sketches
    if use_rank_approx == "True":
        rho_meanpair  = []
        degree = num_sketches
        for j in range(degree):
            sketch_tmp = cs[j].sketch_mat(rho_t)
            mtransform = cs[j].sketch_mat(mval)
            rho_meanpair.append([sketch_tmp.to(device), mtransform.to(device)])
        #=======================================================================
        rho_t = rho_t.to(device) 
    else:
        rho_t = rho_t.to(device)
    mval = mval.to(device)
    v1 = v1.to(device)

    if datasetname == "ogbn-arxiv" or datasetname == "ogbn-products":
        Ctmp = C.to_sparse_coo().clone()
        row_indices = Ctmp.indices()[0]
        col_indices = Ctmp.indices()[1]
        values = Ctmp.values()
    else:
        Ctmp = C.clone()
        row_indices = Ctmp.coalesce().indices()[0]
        col_indices = Ctmp.coalesce().indices()[1]
        values = Ctmp.coalesce().values()

    Ctmp = SparseTensor(
        row=row_indices,
        col=col_indices,
        value=values,
        sparse_sizes=C.size()
    )
    
    model = model.to(device)

    #====================================================================================================
    best_model = None
    best_val_acc = 0.0
    train_time = []
    all_mem = []
    C = C.to(device)
    data.x = data.x.to(device)
    best_list = []
    # test_x = data.x[test_mask].to(device)
    for run in range(num_runs):
        best_test = 0
        epoch_best = 0
        training = True
        model.train()
        for epoch in range(num_epochs):
            se = time.time()
            
            optimizer.zero_grad()
            # out = model(S_x, S_c, mfact, mfact1, rho_t, mval, data.x, training = training)
            if use_rank_approx == "False":
                
                out, z2 = model(S_x, C, rho_meanpair, training = training, use_rank_approx = use_rank_approx, cslist = cslist)
                # out = model(S_x, C, rho_meanpair, training = training, use_rank_approx = use_rank_approx)
            else:
                out, z2 = model(S_x, Sc, rho_meanpair, training = training, use_rank_approx = use_rank_approx, cslist = cslist)  
            alpha = 1
            # print("out is ", out)
            # exit()
            loss = F.cross_entropy(out[train_mask], data.y[train_mask])  
      
            random_sample = np.random.choice(train_indices, size=min(random_sample_size, len(train_indices)), replace=False)
            # sample2  = np.random.choice(train_indices, size=min(50, len(train_indices)), replace=False)
            # zl = zl.detach()
            cmat = Ctmp[random_sample, random_sample].to_dense()
            targetx = (data.x[random_sample,:]).to(device)

            
            correct = compute_accuracy(out.argmax(dim=1)[train_mask], data.y[train_mask])
            acc = int(correct) / int(train_mask.sum())
            losses.append(loss.item())
            val_loss = F.cross_entropy(out[val_mask], data.y[val_mask])
            val_correct = compute_accuracy(out.argmax(dim=1)[val_mask], data.y[val_mask])
            val_acc = int(val_correct) / int(val_mask.sum())
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = copy.deepcopy(model)
            loss.backward()
            model.update_weight3(data.y[random_sample], z2, sample_sketch, targetx, cmat, unsketch_mat = rho_t[random_sample,:].to(device))           
            nn.utils.clip_grad_norm_(model.parameters(), 1)  # Clip gradients
            optimizer.step()
            
            train_memory_usage = torch.cuda.memory_allocated(device) -   model_memory_usage
            all_mem.append(train_memory_usage)
            t1_stop = time.time() - se
            if run == 0:
                train_time.append(t1_stop)
            if (epoch+1) % 10== 0:
                pred = best_model(data.x, C, training = False).argmax(dim=1)
                correct = compute_accuracy(pred[test_mask], data.y[test_mask])
                test_acc = int(correct) / int(test_mask.sum())
                if best_test < test_acc:
                    best_test = test_acc
                    epoch_best = epoch
                print('Epoch: {}, Loss: {:.4f}, Train Acc: {:.4f}, Valid Acc: {:.4f}, Test Acc: {:.4f}'.format(epoch+1, loss.item(), acc, val_acc, test_acc))
                # with open(result_save_dir+'/results_all.txt', 'a') as f:
                #     f.write(f"{run}\t{epoch}\t{loss.item():.4f}\t{acc:.4f}\t{val_acc:.4f}\t{test_acc:.4f}\n")
        best_list.append([best_test, epoch_best])
        
        del model
        best_val_acc = 0.0
        
        gc.collect()
        model = GCN(hidden_dim = hidden_dim, layers = num_layers, layernorm= layer_norm, dropoutval = dropout, order = order).to(device)
        
        # model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=l2_param)
        training = False
        # model.eval()

        # Use the best model for test prediction
        if best_model is not None:
            best_model.eval()
            best_model = best_model
            pred = best_model(data.x, C, training = False).argmax(dim=1)
            correct = compute_accuracy(pred[test_mask], data.y[test_mask])
            # else:    
            acc = int(correct) / int(test_mask.sum())
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
    
    my_float_list = [statistics.mean(accuracies),"+-", statistics.stdev(accuracies), ratio, learn_rate, l2_param, num_layers, hidden_dim, dropout, layer_norm, ratio2, order, lr_update_weight, random_sample_size, epochs, runs, best_list]

    float_list_as_string = ' '.join(map(str, my_float_list))
    print(my_float_list)
    with open('./output/pgcn_citeseer.txt', 'a') as file:
        file.write(float_list_as_string + '\n')
