import os.path as osp
import torch_geometric.transforms as T
import torch
from torch_geometric.utils import add_remaining_self_loops, degree
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected, add_self_loops
import numpy as np
import warnings
import os
import secrets

warnings.filterwarnings("ignore", message="`resume_download` is deprecated")

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def normalize_edge(edge_index, n_node):
    edge_index = to_undirected(edge_index.t(), num_nodes=n_node)

    edge_index, _ = add_self_loops(edge_index, num_nodes=n_node)

    row, col = edge_index
    deg = degree(col)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.

    edge_weight_gcn = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    edge_weight_sage = deg_inv_sqrt[row] * deg_inv_sqrt[row]

    return edge_index, edge_weight_gcn, edge_weight_sage
   
def seed_everything(seed=1033):
    secrets.SystemRandom().seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataset(name,  args):
    data = torch.load(f'./dataset/{name}.pt')
    #print(data)
    seed_everything(args.seed) 


    if name in ['cora_raw', 'pubmed_raw']:
        data = random_planetoid_splits(data, args.imb_class, args.imb_ratio)


    if args.dataset=="cora_raw":
        args.categories=['Rule Learning', 'Neural Networks', 'Case Based', 'Genetic Algorithms', 'Theory', 'Reinforce- ment Learning', 'Probabilistic Methods']
    elif args.dataset=="pubmed_raw":
        args.categories=['Diabetes Mellitus, Ex- perimental', 'Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2']
    
    else:
        data.raw_text = data.raw_texts
        my_num_val=500
        if name == 'Arxiv':
            args.imb_class=list(range(30))
        elif name=="Photo":
            args.imb_class=list(range(8))
            my_num_val=200
        elif name=="Computer":
            args.imb_class=list(range(6))
            my_num_val=100
        elif name=="History":
            args.imb_class=list(range(4,12))
            
        print(args.imb_class)
            
        data = random_planetoid_splits(data, args.imb_class, args.imb_ratio,num_val=my_num_val)
        
    data.class_num = []
    for i in data.y.unique():
        data.class_num.append(torch.sum((data.y == i) & data.train_mask).item())
        
    print( data.class_num) #[4, 4, 4, 4, 4, 20, 20]
    
   



    data.x = torch.tensor(data.bow_x, dtype = torch.float)

    data.x = data.x/ data.x.sum(1, keepdim=True)
    data.x[torch.isnan(data.x)] = 0      
    
    args.num_features = data.x.shape[1]
    args.num_nodes = data.x.shape[0]
    args.num_classes = data.y.max().item() + 1
    
    data.edge_index, _ = add_remaining_self_loops(
        data.edge_index, num_nodes=data.num_nodes) # add remaining self-loops (i,i) for every node i in  the graoh
    data.edge_index = to_undirected(data.edge_index, data.num_nodes) # if (i,j) in edge_index, generate (j,i) in edge_index
    

    # calculate the degree normalize term: the product of the inverse square roots of the degrees of the source and destination nodes
    row, col = data.edge_index # source and destination nodes
    deg = degree(col, data.num_nodes)
    deg_inv_sqrt = deg.pow(-0.5) 

    
    data.edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    data.adj_t = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], value = data.edge_weight, is_sorted=False)
    

    #for link prediction
    transform = T.RandomLinkSplit(is_undirected=True, neg_sampling_ratio = 1.0, num_val = 0.1, num_test = 0.2) # negative sampling
    train_data, val_data, test_data = transform(data)
    train_edge, val_edge, test_edge = train_data.edge_label_index, val_data.edge_label_index, test_data.edge_label_index
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = train_edge[:, :train_edge.shape[1]//2].t()
    split_edge['train']['edge_neg'] = train_edge[:, train_edge.shape[1]//2:].t()
    split_edge['valid']['edge'] = val_edge[:, :val_edge.shape[1]//2].t()
    split_edge['valid']['edge_neg'] = val_edge[:, val_edge.shape[1]//2:].t()
    split_edge['test']['edge'] = test_edge[:, :test_edge.shape[1]//2].t()
    split_edge['test']['edge_neg'] = test_edge[:, test_edge.shape[1]//2:].t()


    data.train_edge_index, data.train_edge_weight_gcn, data.train_edge_weight_sage = normalize_edge(split_edge['train']['edge'], data.x.shape[0])
    data.train_adj_gcn = SparseTensor(row=data.train_edge_index[0], col=data.train_edge_index[1], value = data.train_edge_weight_gcn, is_sorted=False)
    
    
    return data, split_edge, args



def random_planetoid_splits(data, imb_class, imb_ratio, num_train_node=20,num_val=500):
    indices = []
    
    num_classes = data.y.max().item() + 1
    for i in range(num_classes):
        index = torch.nonzero(data.y == i).view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    #print(indices)
    train_index = []
    res_index = []
    for i, _ in enumerate(indices):
        #print(i,_)
        if i not in imb_class:
            train_index.append(_[:num_train_node])
            res_index.append(_[num_train_node:])
        else:
            train_index.append(_[:int(num_train_node*imb_ratio)])
            res_index.append(_[int(num_train_node*imb_ratio):])
    
    train_index = torch.cat(train_index, dim=0)
    res_index = torch.cat(res_index, dim=0)
    
    res_index = res_index[torch.randperm(res_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(res_index[:num_val], size=data.num_nodes)
    data.test_mask = index_to_mask(res_index[num_val:], size=data.num_nodes)

    data.train_id = train_index
    data.val_id = res_index[:num_val]
    data.test_id = res_index[num_val:]

    return data
