import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'models')))
from GCN import runner_GCN
from GAT import runner_GAT
from SGC import runner_SGC
from SAGE import runner_SAGE
def train(datasetname, arch, learn_rate, l2_param, rate, use_rank_approx, num_layers, hidden_dim, dropout, layer_norm, ratio2, order, device,lr_update_weight, random_sample,epochs, runs, seed):
    if arch == "GCN":
        runner_GCN(datasetname, learn_rate, l2_param, ratio, use_rank_approx, num_layers, hidden_dim, dropout, layer_norm, ratio2, order, device,lr_update_weight, random_sample,epochs, runs, seed)
    elif arch == "SGC":
        file_name = f"./output/{arch}_lr_{learn_rate}_wd_{l2_param}_ratio_{ratio}.txt"
        runner_SGC(datasetname, learn_rate, l2_param, ratio, use_rank_approx, file_name=file_name, device=device, num_runs=runs, seed=seed, num_epochs=epochs)
    elif arch == "GAT":
        runner_GAT(datasetname, learn_rate, l2_param, ratio, use_rank_approx, num_layers, hidden_dim, dropout, layer_norm, ratio2, order, device, epochs, seed, runs)
    elif arch == "SAGE":
         runner_SAGE(datasetname, learn_rate, l2_param, ratio, use_rank_approx, num_layers, hidden_dim, dropout, layer_norm, ratio2, order, device, epochs, seed, runs)
    else:
        print("invalid architecture input, available architectures are : GCN, SAGE, GAT, SGC")
if __name__ == '__main__':
    datasetname = sys.argv[1]
    arch = sys.argv[2]
    learn_rate = float(sys.argv[3])
    l2_param = float(sys.argv[4])
    ratio = float(sys.argv[5])
    use_rank_approx = (sys.argv[6])
    num_layers = int(sys.argv[7])
    hidden_dim = int(sys.argv[8])
    dropout = float(sys.argv[9])
    layer_norm = int(sys.argv[10])
    if layer_norm == 0:
        layer_norm = False
    else:
        layer_norm = True   
    ratio2 = float(sys.argv[11])
    order = int(sys.argv[12])
    device = int(sys.argv[13])
    lr_update_weight = float(sys.argv[14])
    random_sample = int(sys.argv[15])
    epochs = int(sys.argv[16])
    runs = int(sys.argv[17])
    seed = int(sys.argv[18])
    train(datasetname, arch, learn_rate, l2_param, learn_rate, use_rank_approx, num_layers, hidden_dim, dropout, layer_norm, ratio2, order, device, lr_update_weight, random_sample,epochs, runs, seed)

