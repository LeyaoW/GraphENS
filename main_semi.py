import os.path as osp
import random
import torch
import torch.nn.functional as F
from nets import *
from data_utils import *
from args import parse_args
from models import *
from losses import *
from sklearn.metrics import balanced_accuracy_score, f1_score
import statistics
import numpy as np
from dataset import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import warnings

warnings.filterwarnings("ignore", message="Using a non-full backward hook when the forward contains multiple autograd Nodes")
warnings.filterwarnings("ignore", category=UserWarning)


## Arg Parser ##
args = parse_args()

## Handling exception from arguments ##
assert not (args.warmup < 1 and args.ens)
# assert args.imb_ratio > 1


args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

int_list=[]
for char in args.imb_class:
    int_list.append(int(char))
args.imb_class=int_list


# # Load Dataset ##
# dataset = "Cora"
# path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
# dataset = get_dataset0(dataset, path, split_type='public')
# # data = dataset[0]

#Our processed data
data, split_edge, args = get_dataset(args.dataset, args)
n_cls = data.y.max().item() + 1
data = data.to(device)

def backward_hook(module, grad_input, grad_output):
    global saliency
    saliency = grad_input[0].data


def train(data):
    global class_num_list, idx_info, prev_out, aggregator
    global data_train_mask, data_val_mask, data_test_mask

    

    model.train()
    optimizer.zero_grad()

    if args.ens:
        # Hook saliency map of input features
        model.conv1[0].temp_weight.register_backward_hook(backward_hook)
        
        # Sampling source and destination nodes
        sampling_src_idx, sampling_dst_idx = sampling_idx_individual_dst(class_num_list, idx_info, device)
        beta = torch.distributions.beta.Beta(2, 2)
        lam = beta.sample((len(sampling_src_idx),) ).unsqueeze(1)
        ori_saliency = saliency[:data.x.shape[0]] if (saliency != None) else None

        # Augment nodes
        if epoch > args.warmup:
            with torch.no_grad():
                prev_out = aggregator(prev_out, data.edge_index)
                prev_out = F.softmax(prev_out / args.pred_temp, dim=1).detach().clone()
            new_edge_index, dist_kl = neighbor_sampling(data.x.size(0), data.edge_index, sampling_src_idx, sampling_dst_idx,
                                        neighbor_dist_list, prev_out)
            new_x = saliency_mixup(data.x, sampling_src_idx, sampling_dst_idx, lam, ori_saliency, dist_kl = dist_kl, keep_prob=args.keep_prob)
        else:
            new_edge_index = duplicate_neighbor(data.x.size(0), data.edge_index, sampling_src_idx)
            dist_kl, ori_saliency = None, None
            new_x = saliency_mixup(data.x, sampling_src_idx, sampling_dst_idx, lam, ori_saliency, dist_kl = dist_kl)
        new_x.requires_grad = True           

        # Get predictions
        output = model(new_x, new_edge_index, None)
        prev_out = (output[:data.x.size(0)]).detach().clone() # logit propagation

        ## Train_mask modification ##
        add_num = output.shape[0] - data_train_mask.shape[0]
        new_train_mask = torch.ones(add_num, dtype=torch.bool, device= data.x.device)
        new_train_mask = torch.cat((data_train_mask, new_train_mask), dim =0)

        ## Label modification ##
        new_y = data.y[sampling_src_idx].clone()
        new_y = torch.cat((data.y[data_train_mask], new_y),dim =0)

        ## Compute Loss ##
        criterion(output[new_train_mask], new_y).backward()

    else: ## Vanilla Train ##
        output = model(data.x, data.edge_index, None)
        criterion(output[data_train_mask], data.y[data_train_mask]).backward()

    with torch.no_grad():
        model.eval()
        output = model(data.x, data.edge_index, None)
        val_loss= F.cross_entropy(output[data_val_mask], data.y[data_val_mask])

    optimizer.step()
    scheduler.step(val_loss)


@torch.no_grad()
def test(args,data):
    model.eval()
    logits = model(data.x, data.edge_index, None,)
    accs, baccs, f1s = [], [], []

    for i, mask in enumerate([data_train_mask, data_val_mask, data_test_mask]):
        pred = logits[mask].max(1)[1]
        y_pred = pred.cpu().numpy()
        y_true = data.y[mask].cpu().numpy()
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        bacc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')

        accs.append(acc)
        baccs.append(bacc)
        f1s.append(f1)
    

    min_mask= torch.zeros(size=data.y.shape).to(device)
    #print(data_test_mask.shape)
    for i in args.imb_class:
        min_mask= ((data.y == i).bool() | min_mask.bool()).to(device)
    
    maj_mask= ~min_mask
    
    min_mask=min_mask & data_test_mask
    maj_mask=maj_mask & data_test_mask
    #print(maj_mask.shape)
        
    maj_pred = logits[maj_mask].max(1)[1]
    maj_acc =maj_pred.eq(data.y[maj_mask]).sum().item() / maj_mask.sum().item()
    
    min_pred = logits[min_mask].max(1)[1]
    min_acc = min_pred.eq(data.y[min_mask]).sum().item() / min_mask.sum().item()
    
        
    
    return accs, baccs, f1s, maj_acc, min_acc


## Log for Experiment Setting ##
setting_log = "Dataset: {}, ratio: {}, net: {}, n_layer: {}, feat_dim: {}, ens: {}".format(
    args.dataset, str(args.imb_ratio), args.net, str(args.n_layer), str(args.feat_dim), str(args.ens))

repeatition = 5
# seed = 100
seed=1033
avg_test_acc, avg_val_acc, avg_val_f1, avg_test_bacc, avg_test_f1, avg_maj_acc, avg_min_acc = [], [], [], [], [],[],[]
for r in range(repeatition):

    ## Fix seed ##
    torch.cuda.empty_cache()
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    seed += 1

    data_train_mask, data_val_mask, data_test_mask = data.train_mask.clone(), data.val_mask.clone(), data.test_mask.clone()

    ## Data statistic ##
    stats = data.y[data_train_mask]
    n_data = [] # number of data in each class
    for i in range(n_cls):
        data_num = (stats == i).sum()
        n_data.append(int(data_num.item()))

    # # Load data
    # if args.dataset == 'cora_raw':
    #     class_sample_num = 20
    #     imb_class_num = 5
    # elif args.dataset == 'CiteSeer':
    #     class_sample_num = 20
    #     imb_class_num = 3
    # elif args.dataset == 'pubmed_raw':
    #     class_sample_num = 20
    #     imb_class_num = 2
    # else:
    #     print("no this dataset: {args.dataset}")

    idx_info = get_idx_info(data.y, n_cls, data_train_mask)
    #print(class_num_list) [4, 4, 4, 4, 4, 20, 20]
    #print( idx_info)

    # #for artificial imbalanced setting: only the last imb_class_num classes are imbalanced
    # class_num_list = []
    class_num_list = data.class_num
    # for i in range(n_cls):
    #     #if args.imb_ratio > 1 and i > n_cls-1-imb_class_num: #only imbalance the last classes
    #     if args.imb_ratio > 1 and i < imb_class_num:  #only imbalance the first 'imb_class_num' classes
    #         class_num_list.append(int(class_sample_num*(1./args.imb_ratio)))
    #     else:
    #         class_num_list.append(class_sample_num)

    # if args.imb_ratio > 1:
    #     data_train_mask, idx_info = split_semi_dataset(len(data.x), n_data, n_cls, class_num_list, idx_info, data.x.device)

    ## Adjacent node distribution ##
    if args.ens:
        neighbor_dist_list = get_ins_neighbor_dist(data.y.size(0), data.edge_index, data_train_mask, device)
    else:
        neighbor_dist_list = None

    ## Model Selection ##
    # if args.net == 'GCN':
    #     model = GCN(args.n_layer, dataset.num_features, args.feat_dim, n_cls, normalize=True, is_add_self_loops=True)
    # elif args.net == 'GAT':
    #     model = GAT(args.n_layer, dataset.num_features, args.feat_dim, n_cls, args.n_head, is_add_self_loops=True)
    # elif args.net == "SAGE":
    #     model = SAGE(args.n_layer, dataset.num_features, args.feat_dim, n_cls)
    #print(args.num_features,dataset.num_features) # 1433 1433
    if args.net == 'GCN':
         model = GCN(args.n_layer, args.num_features, args.feat_dim, n_cls, normalize=True, is_add_self_loops=True)
    elif args.net == 'GAT':
        model = GAT(args.n_layer, args.num_features, args.feat_dim, n_cls, args.n_head, is_add_self_loops=True)
    elif args.net == "SAGE":
        model = SAGE(args.n_layer, args.num_features, args.feat_dim, n_cls)
    else:
        raise NotImplementedError("Not Implemented Architecture!")

    ## Criterion Selection ##
    if args.loss_type == 'ce': # CE
        criterion = CrossEntropy()
    else:
        raise NotImplementedError("Not Implemented Loss!")

    model = model.to(device)
    criterion = criterion.to(device)

    # Set optimizer
    optimizer = torch.optim.Adam([
        dict(params=model.reg_params, weight_decay=5e-4),
        dict(params=model.non_reg_params, weight_decay=0),], lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor = 0.5,
                                                           patience = 100,
                                                           verbose=False)

    # Train models
    best_val_acc = test_acc = best_val_f1 = 0
    saliency = None
    prev_out = None
    aggregator = MeanAggregation()
    for epoch in range(1, 1001):
        if epoch%100==0:
            print(epoch)
        train(data)
        accs, bacc, f1s, maj_acc, min_acc = test(args,data)
        train_acc, val_acc, tmp_test_acc = accs
        train_f1, tmp_val_f1, tmp_test_f1 = f1s
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            val_f1 = tmp_val_f1
            test_acc = tmp_test_acc
            test_bacc = bacc[2]
            test_f1 = f1s[2]
            test_min_acc=min_acc
            test_maj_acc= maj_acc
            
            

    avg_val_acc.append(best_val_acc)
    avg_val_f1.append(val_f1)
    avg_test_acc.append(test_acc)
    avg_test_bacc.append(test_bacc)
    avg_test_f1.append(test_f1)
    
    avg_maj_acc.append(test_maj_acc)
    avg_min_acc.append(test_min_acc)

## Calculate statistics ##
acc_CI =  (statistics.stdev(avg_test_acc) / (repeatition ** (1/2)))
bacc_CI =  (statistics.stdev(avg_test_bacc) / (repeatition ** (1/2)))
f1_CI =  (statistics.stdev(avg_test_f1) / (repeatition ** (1/2)))
avg_acc = statistics.mean(avg_test_acc)
avg_val_acc = statistics.mean(avg_val_acc)
avg_val_f1 = statistics.mean(avg_val_f1)
avg_bacc = statistics.mean(avg_test_bacc)
avg_f1 = statistics.mean(avg_test_f1)
avg_maj_acc=statistics.mean(avg_maj_acc)
avg_min_acc=statistics.mean(avg_min_acc)

avg_log = 'Test Acc: {:.4f} +- {:.4f}, BAcc: {:.4f} +- {:.4f}, F1: {:.4f} +- {:.4f}, Val Acc: {:.4f}, Val F1: {:.4f}, Maj-Min: {:.4f}'
avg_log = avg_log.format(avg_acc ,acc_CI ,avg_bacc, bacc_CI, avg_f1, f1_CI, avg_val_acc, avg_val_f1, avg_maj_acc-avg_min_acc)
log = "{}\n{}".format(setting_log, avg_log)
print(log)