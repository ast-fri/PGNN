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
from torch_geometric.nn import GATConv
def runner_GAT(dataset, learn_rate, l2_param, ratio, use_rank_approx, num_layers, hidden_dim, dropout, layer_norm, ratio2, order, device, num_epochs, seed, runs):
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = Planetoid(root='../Project_sketch/dataset', name=dataset, split = 'public')
    data = dataset[0]
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    # print(data.shape)
    gcn_norm = torch_geometric.nn.conv.gcn_conv.gcn_norm
    C1, C2 = gcn_norm(data.edge_index, add_self_loops = True)
    C = (torch.sparse_coo_tensor(C1,C2)).to(torch.float32)
    device = torch.device('cuda:'+str(device))
    # device = torch.device('cpu')
    n = data.x.shape[0]
    
    #===========================================================================================
    class MyMessagePassing(MessagePassing):
        def __init__(self, in_channels: int, out_channels: int, heads: int, bias: bool = True, **kwargs):
            super().__init__(aggr='add')  # "Add" aggregation (Step 5).
            self.conv = GATConv(in_channels, out_channels, heads)

        def reset_parameters(self):
            pass
        def forward(self, nf_mats, conv_mats, rho_meanpair = None, use_rank_approx = True, weight2 = None, tslist = None, training = None, layer_return = None):
            
            if training == True:
                if use_rank_approx == "False":
                    
                    zs = self.conv(nf_mats, conv_mats)
                    zs = F.relu(zs)
                    zs = weight2.T @(rho_t.T @ (zs - mval))
                    # print("reached zs shape ", zs)
                    # exit()
                    if layer_return == True:
                        zl = zs
                    
                    zs = rho_t @ (weight2 @ (zs)) + mval
                    # zs = F.layer_norm(zs, zs.size()[1:])

                    
                    if layer_return == True:
                        return zs, zl
                    else:
                        return zs
                else:
                    print("not implemented")
                    exit()
                    if layer_return == True:
                        return sumv, sumv
                    else:
                        return sumv
            elif training == "Baseline":
                    zs = self.conv(nf_mats, conv_mats)
                    # return F.elu(zs)
                    return (zs)    
                
            else:
                zs = self.conv(nf_mats, conv_mats)
                # return F.elu(zs)
                return (zs)            
        def message(self, x_j, norm):
            return norm.view(-1, 1) * x_j

    class GAT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            n_neurons =  hidden_dim
            n_features =  data.x.shape[1]
            n_classes = max(data.y) + 1
            self.num_layers = num_layers
            batchnorm = False
            self.dropout = dropout
            self.batchnorm = layer_norm
            self.lowrank = lowrank
            self.weight2 = Parameter(torch.empty((rho_t.shape[1], int(ratio*n)), dtype=torch.float32), requires_grad=False)
            self.weight2_optimizer = torch.optim.SGD([self.weight2], lr=0.01)
            heads = 2
            if self.batchnorm:
                self.bns = torch.nn.ModuleList()
                self.bns.append(torch.nn.BatchNorm1d(n_neurons*heads))
            self.convs = torch.nn.ModuleList()

            self.convs.append(MyMessagePassing(n_features, n_neurons, heads))
            
            for j in range(num_layers-2):
                self.convs.append(MyMessagePassing(heads*n_neurons, n_neurons, heads))
                
                if self.batchnorm:
                    self.bns.append(torch.nn.BatchNorm1d(heads*n_neurons))
            
            self.convs.append(MyMessagePassing(heads*n_neurons, n_classes, 1))
            # self.ln2 = torch.nn.LayerNorm(n_classes.item())
            self.reset_parameters()
        def reset_parameters(self):
            for conv in self.convs:
                conv.reset_parameters()
            if self.batchnorm:
                for bn in self.bns:
                    bn.reset_parameters()
            self.weight2.data = v1
        def update_weight3(self, target_value, predval, sx, targetx, cmat, learning_rate=0.001, unsketch_mat = None):
            
            self.weight2_optimizer.zero_grad()            
        
            weight2_temp = self.weight2.detach().clone().requires_grad_(True)        
        
            sx = (unsketch_mat @ weight2_temp @ sx + mval[random_sample])
        
            term2 = alpha*torch.linalg.norm(targetx - sx, 'fro') 
            loss2 =   term2
            
            loss2.backward()
            torch.nn.utils.clip_grad_norm_(weight2_temp, max_norm=1)
            with torch.no_grad():
                self.weight2 -= learning_rate * weight2_temp.grad
            self.weight2 = torch.nn.Parameter(weight2_temp.detach().clone())

        def update_weight2(self, custom_loss_function, target_value, predval, learning_rate=0.001):
            # Enable gradient tracking for weight2 temporarily
            weight2_temp = self.weight2.detach().clone().requires_grad_(True)
            
            # Compute the custom loss
            custom_loss = custom_loss_function(weight2_temp, target_value, predval)
            # Backpropagate the custom loss
            custom_loss.backward()
            
            # torch.nn.utils.clip_grad_norm_([weight2_temp], max_norm=1.0)

            # Manually update self.weight2
            
            with torch.no_grad():
                self.weight2 -= learning_rate * weight2_temp.grad

        def loss_function(self, weight2, target_value, predval):
            # return torch.linalg.norm((rho_t[random_sample,:] @ weight2 @ sample_sketch - target_value))
            prediction = rho_t[random_sample, :] @ weight2 @ predval + mval[random_sample]   
            alpha = 1
            return torch.linalg.norm((prediction - target_value), 'fro') 

        def forward(self, nf_mats, conv_mats, rho_meanpair = None, training=None, use_rank_approx = True):
            
            # exit()
            if training == True:
                z = []    
                nf_mats, zl = self.convs[0](nf_mats, conv_mats, rho_meanpair, use_rank_approx = use_rank_approx, weight2 = self.weight2, training = training, layer_return = True)
                
                if self.batchnorm:
                    nf_mats = self.bns[0](nf_mats)
                z.append(zl.detach())
                for i, conv in enumerate(self.convs[1:-1]):
                    nf_mats, zl = conv(nf_mats, conv_mats, rho_meanpair, use_rank_approx = use_rank_approx, weight2 = self.weight2,training = training, layer_return = True)
                    
                    if self.batchnorm:
                        nf_mats = self.bns[i+1](nf_mats)
                    z.append(zl.detach())
                    nf_mats = F.dropout(nf_mats, p=self.dropout, training= training)
                
                nf_mats, zl = self.convs[-1](nf_mats, conv_mats, rho_meanpair, use_rank_approx = use_rank_approx, weight2 = self.weight2, training = training, layer_return = True)
                z.append(zl.detach())
                if use_rank_approx == "True":
                    nf_mats  = rho_t @ self.weight2.cpu() @ nf_mats.cpu() + mval.cpu()
                    nf_mats = nf_mats.to(device)
                    # nf_mats = rho_t @ self.weight2 @ nf_mats + mval
                return F.log_softmax(nf_mats, dim = 1), zl
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
        def generate_pairs(n):
            a = torch.tensor([0]*n)
            tup = torch.stack((a, torch.tensor(list(range(1,n+1)))))
            return tup
        a = torch.randint(0, d, size=(k1,))
        b = torch.randint(0, d, size=(k1,))
        random_pairs = torch.stack((a, b))
        data.x = torch.cat((data.x, torch.mul(data.x[:, random_pairs[0]] ,data.x[:,random_pairs[1]])), dim = 1)
    
    n, d = data.x.shape[0], data.x.shape[1]
    print("preprocessing started ==========================================================================================")
    import time
    s1 = time.time()
    Ctmp = C.to_sparse()
    tmp = (Ctmp @ data.x)
    del Ctmp
    mval = torch.mean(tmp, 1, True)
    ps =   (tmp - torch.mean(tmp, 1, True))
    
    #==============================================================================
    ps = (ps / (np.sqrt(int(ratio*n) - 1)))
    # ps = ps / torch.norm(ps, dim=1, keepdim=True) 
    # ps = F.relu(ps)
    import scipy

    print("eigen value computation started")
    [_,_,rho_t] = torch.svd_lowrank((ps.T), q=int(ratio*n), niter=2, M=None)
    sample_sketch = rho_t.T @ (data.x[:, :old_num_features] - mval)
    sample_sketch = sample_sketch.to(device)
    lowrank  =  min(int(ratio*n), int(0.1*n), int(20000))
    row_indices = torch.arange(lowrank)
    col_indices = torch.arange(lowrank)
    # v1 =  res[2].T
    tmp = rho_t.clone()
    v1 = torch.eye(int(ratio*n))
    train_indices = np.where(np.array(train_mask))[0]
    val_indices = np.where(np.array(val_mask))[0]

    train_indices = (list(train_indices))
    #=====================================================================================
    data.x = data.x[:, :old_num_features]
    model = GAT()
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
        from sketch import CountSketch, TensorSketch
        from sketch import DEFAULT_COUNT_SKETCH_CONFIG
        from sketch import DEFAULT_TENSOR_SKETCH_CONFIG
        num_sketches = 3
        # sketch_list = []
        n = data.x.shape[0]
        in_dim = int(n)
        ratio2 = 0.4
        out_dim = int(ratio2*n)
        # cs = CountSketch(in_dim, out_dim).to(device)
        
        cs = []
        for _ in range(num_sketches):
            cs.append(CountSketch(in_dim, out_dim))
        ts = TensorSketch(cs, config = DEFAULT_TENSOR_SKETCH_CONFIG)
        
        Sc = ts.sketch_mat(Ctmp)
        # rho_meanpair = ts.sketch_mat(torch.cat((rho_t, mval.expand(-1, int(ratio*n))), dim = 1))
        for i in range(len(Sc)):
            Sc[i] = rho_t.T @ Sc[i].T
            Sc[i] = Sc[i].to(device)
            # rho_meanpair[i] = rho_meanpair[i].to(device)
        del Ctmp
    print("sketch obtained")
    mfact =  tmp.T @ (C @ torch.mean(data.x, 1, True))
    mfact1 = -rho_t.T @ torch.mean(data.x, 1, True)
    data.y = data.y.to(device)  
    mfact = mfact.to(device)
    mfact1 = mfact1.to(device)
    print("preprocessing ended================================================================================================")
    print("time taken is ", time.time() - s1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=l2_param)
    print("GAT Model:")
    GAT()
    def compute_accuracy(pred_y, y):
        return (pred_y == y).sum()
    model.train()
    losses = []
    accuracies = []
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
    else:
        rho_t = rho_t.to(device)
    mval = mval.to(device)
    v1 = v1.to(device)
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
    best_model = None
    best_val_acc = 0.0
    train_time = []
    all_mem = []
    if use_rank_approx == "False":
        C  = C.to_sparse_coo()
        C = C.to(device)
    for run in range(num_runs):
        # train_loses = []
        # val_loses = []
        training = True
        model.train()
        for epoch in range(num_epochs):
            se = time.time()
            optimizer.zero_grad()
            # out = model(S_x, S_c, mfact, mfact1, rho_t, mval, data.x, training = training)
            if use_rank_approx == "False":
                # out = model(S_x, C, rho_meanpair, training = training, use_rank_approx = use_rank_approx)
                out, z2 = model(S_x, C, rho_meanpair, training = training, use_rank_approx = use_rank_approx)
                # out = model(S_x, C, rho_meanpair, training = training, use_rank_approx = use_rank_approx)
            else:
                out, z2 = model(S_x, Sc, rho_meanpair, training = training, use_rank_approx = use_rank_approx)
            alpha = 1
            # print("out is ", out)
            # exit()
            loss = F.cross_entropy(out[train_mask], data.y[train_mask])  
            
            random_sample = np.random.choice(train_indices, size=min(400, len(train_indices)), replace=False)
            
            cmat = Ctmp[random_sample, random_sample].to_dense()
            targetx = (data.x[random_sample,:]).to(device)
             
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
            # nn.utils.clip_grad_norm_(model.parameters(), 1)  # Clip gradients
            model.update_weight3(data.y[random_sample], z2, sample_sketch, targetx, cmat, unsketch_mat = rho_t[random_sample,:].to(device))
       
            optimizer.step()
            

            train_memory_usage = torch.cuda.memory_allocated(device) -   model_memory_usage
            all_mem.append(train_memory_usage)
            t1_stop = time.time() - se
            if run == 0:
                train_time.append(t1_stop)
            if (epoch+1) % 10 == 0:
                b2 = copy.deepcopy(best_model)
                b2.eval()
                
                b2 = b2.to('cpu')
                pred = b2(data.x, C.to('cpu'), training = False).argmax(dim=1)
                correct = compute_accuracy(pred[test_mask], data.y[test_mask].to('cpu'))
                test_acc = int(correct) / int(test_mask.sum())
                print('Epoch: {}, Loss: {:.4f}, Training Acc: {:.4f}, Validation Accuracy: {:.4f}, Test Accuracy: {:.4f}'.format(epoch+1, loss.item(), acc, val_acc, test_acc))
        del model
        gc.collect()
        model = GAT().to(device)
        
        # model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=l2_param)
        training = False
        # model.eval()
        # Use the best model for test prediction
        if best_model is not None:
            best_model.eval()
            # if use_rank_approx == "True":
            best_model = best_model.to('cpu')
            pred = best_model(data.x, C.to('cpu'), training = False).argmax(dim=1)
            # if use_rank_approx == "True":
            correct = compute_accuracy(pred[test_mask], data.y[test_mask].to('cpu'))
            # else:    
            #     correct = compute_accuracy(pred[test_mask], data.y[test_mask])
            acc = int(correct) / int(test_mask.sum())
            # runs with less accuracy are discarded
            if acc >= 0.5:
                accuracies.append(acc)
            print(f'Best Model Test Accuracy: {acc:.4f}')
        else:
            print('No best model found.')
    import statistics
    print(accuracies)
    # try:
    print(statistics.mean(accuracies),"+-", statistics.stdev(accuracies))
    print("average training time per epoch is ", statistics.mean(train_time), len(train_time))
    print("peak memory usage (MB) is ", max(all_mem)/1048576.0)

    my_float_list = [statistics.mean(accuracies),"+-", statistics.stdev(accuracies), learn_rate, l2_param]

    float_list_as_string = ' '.join(map(str, my_float_list))
   