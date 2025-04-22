# Implementation of HeteroFL publised in ICLR21

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import pickle
import copy
import sys
import os
import torch.multiprocessing as mp
import queue
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import arguments
from dataLoader.dataLoaders import getNodeIndicies
from dataLoader.dataset import get_dataset
from nn_models import resnet_sbn, resnet_sbn_tiny, vggnet_sbn

def select_level(id, args):
    if args.randomBranch == 0:
        i = id / args.nodes
    else:
        i = np.random.random_sample()
    if i < args.level1:
        return 0
    elif i < (args.level1+args.level2):
        return 1
    elif i < (args.level1 + args.level2 + args.level3):
        return 2
    else:
        return 3

class Node:
    def __init__(self, nodeID, node_indices, args):
        self.id = nodeID
        self.node_indices = node_indices
        self.__args = args

    def train(self, device, lr, model, train_dataset):
        train_loader = DataLoader(Subset(train_dataset, self.node_indices), \
            batch_size=self.__args.batch_size, shuffle=True)
        model.train()
        model.to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(self.__args.local_epoch):
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

        model.to(torch.device('cpu'))
        return copy.deepcopy(model.state_dict())

def train_worker(device, trainq, resultq, args):

    processDone = False
    train_dataset, _ = get_dataset(args)

    while not processDone:

        msg = trainq.get()
        node = msg['node']

        if node == 'done':
            processDone = True
        
        else:
            lr = msg['lr']
            model = msg['model']
            model_weight = node.train(device, lr, model, train_dataset)
            resultq.put({'weight': copy.deepcopy(model_weight), 'branch_num': msg['branch_num']})
    
        del node
        del msg

def prune(state_dict, param_idx):
    ret_dict = {}
    for k in state_dict.keys():
        if 'num' not in k:
            ret_dict[k] = state_dict[k][torch.meshgrid(param_idx[k])]
        else:
            ret_dict[k] = state_dict[k]
    return copy.deepcopy(ret_dict)

        
def create_model(num_classes, ratio, track=False, scale=True):
    if args.dataset == 'cifar10':
        model = vggnet_sbn.make_VGG(num_classes, ratio, track, scale)

    elif args.dataset == 'cifar100':
        model = resnet_sbn.resnet18(num_classes, ratio, track, scale)

    elif args.dataset == 'tiny-imagenet':
        model = resnet_sbn_tiny.resnet34_tiny(num_classes, ratio, track, scale)

    return model


if __name__ == "__main__":

    mp.set_start_method('spawn')
    os.environ["OMP_NUM_THREADS"] = "1"

    if torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
        devices = [torch.device("cuda:{}".format(i)) for i in range(n_devices)]
    else:
        n_devices = 1
        devices = [torch.device('cpu')]

    args = arguments.parser()

    if args.full == 1:
        args.level1 = 0.0
        args.level2 = 0.0
        args.level3 = 0.0
        args.level4 = 1.0

    num_nodes = args.nodes
    num_round = args.round
    num_local_epoch = args.local_epoch
    
    print("> Setting:", args)

    # load data
    nodeIDs = [i for i in range(num_nodes)]
    manager = mp.Manager()
    nodes = list()
    train_jobs = mp.Queue()
    resultQs = mp.Queue()
    preQ = queue.Queue()
    nodeindices = getNodeIndicies(nodeIDs, num_nodes, args)
    train_dataset, test_dataset = get_dataset(args)
    test_loader = DataLoader(test_dataset, 256, shuffle=False, num_workers=4)

    if args.dataset == "cifar10":
        num_classes = 10

    elif args.dataset == "cifar100":
        num_classes = 100
    
    elif args.dataset == "tiny-imagenet":
        num_classes = 200

    hetero_models = []
    hetero_model_param_idx = []

    pruning_ratios = [1/4, 2/4, 3/4, 4/4]

    if args.base != -1:
        args.scale = 0
    
    for ratio in pruning_ratios:

        model = create_model(num_classes, ratio, track=False, scale=args.scale)
        hetero_models.append(model)
        state_dict = model.state_dict()
        param_idx = {}
        for k in state_dict.keys():
            param_idx[k] = [torch.arange(size) for size in state_dict[k].shape]
        hetero_model_param_idx.append(param_idx)

    for nodeID in nodeIDs:
        nodes.append((Node(nodeID, nodeindices[nodeID], args)))

    # round
    num_nodes_p = max(int(args.fraction * num_nodes), 1)
    global_model_parameters = copy.deepcopy(hetero_models[-1].state_dict())
    train_processes = []

    for i in range(n_devices * args.n_procs):
        p = mp.Process(target=train_worker, args=(devices[i%n_devices], train_jobs, resultQs, args))
        p.start()
        train_processes.append(p)

    lr = args.lr
    np.random.seed(21)
    
    for r in range(1, num_round+1):
        if r % 5 == 0:
            print(f"Round {r}", end=', ')

        cur_time = time.time()
        lr *= args.lr_decay
    
        index_nodes_p = np.random.choice(range(num_nodes),
                                         num_nodes_p, replace=False)
        
        for model, param_idx in zip(hetero_models, hetero_model_param_idx):
            pruned_dict = prune(global_model_parameters, param_idx)
            model.load_state_dict(pruned_dict)

        # train
        count = 0
        for num, i in enumerate(index_nodes_p):
            node = nodes[i]
            branch_num = select_level(node.id, args)
            
            if args.base != -1:
                if branch_num < args.base:
                    continue
                else:
                    branch_num = args.base

            model = hetero_models[branch_num]
            count += 1
            train_jobs.put({'node': copy.deepcopy(node), 'lr':lr, 'model': copy.deepcopy(model), 'branch_num': branch_num})

        peer_weights = [[] for _ in range(len(hetero_models))]

        for _ in range(count):
            result = resultQs.get()
            weight = result['weight']
            branch_num = result['branch_num']
            peer_weights[branch_num].append(weight)

        ## model avg
        
        with torch.no_grad():

            for k, v in global_model_parameters.items():
                count = torch.zeros(v.shape, dtype=torch.float32)
                tmp_v = torch.zeros(v.shape, dtype=torch.float32)
                if 'num' not in k:
                    for i, weights in enumerate(peer_weights):
                        param_idx = hetero_model_param_idx[i]
                        for weight in weights:
                            tmp_v[torch.meshgrid(param_idx[k])] += weight[k]
                            count[torch.meshgrid(param_idx[k])] += 1
                    tmp_v[count > 0] = tmp_v[count > 0].div_(count[count > 0])
                    v[count > 0] = tmp_v[count > 0]

                else:
                    for weights in peer_weights:
                        for weight in weights:
                            tmp_v += weight[k]
                            count += 1
                    tmp_v = tmp_v.div_(count)
                    v = tmp_v
                
                
        if r % 5 == 0:
            preQ.put({'round': r, 'model_weight': copy.deepcopy(global_model_parameters)})
            print(f"Elapsed Time : {time.time()-cur_time:.1f}")

    for i in range(n_devices * args.n_procs):
        train_jobs.put({'node' : 'done'})

    for p in train_processes:
        p.join()

    # Train Finished
    time.sleep(5)
    # Test Start

    acc_list = list()

    if args.base != -1:
        # for naive baseline's global model
        hetero_models = [hetero_models[args.base]] 
        hetero_model_param_idx = [hetero_model_param_idx[args.base]] 
        
    for _ in range(preQ.qsize()):
        msg = preQ.get()
        global_model_parameters = msg['model_weight']
        for model, param_idx in zip(hetero_models, hetero_model_param_idx):
            pruned_dict = prune(global_model_parameters, param_idx)
            model.load_state_dict(pruned_dict)
        
        # global model
        model = create_model(num_classes, pruning_ratios[args.base], track=True, scale=args.scale)
        model.load_state_dict(hetero_models[-1].state_dict(), strict=False)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = nn.DataParallel(model)
        model.to(device)

        # Track global statistics (sBN)

        model.train()
        
        # This procedure takes too much time, so when round < 890, only track 10% of global statistics.
        # Although 10% og global statistics are enough, only round >=900 test results are in FedSD paper. 

        if msg['round'] < 890:
            track_nodes = nodes[:10]
        else:
            track_nodes = nodes

        for node in track_nodes:
            train_loader = DataLoader(Subset(train_dataset, node.node_indices), \
                batch_size=args.batch_size, shuffle=True)
            
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(train_loader):
                    images, labels = images.to(device), labels.to(device)
                    output = model(images)

        # Let's evaluate

        model.eval()

        correct, total = 0, 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                _, pred_labels = torch.max(output, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

            acc= correct/total
    
        print("Round: {} / acc: {}".format(msg['round'], acc))
        acc_list.append(acc)


    file_name = '../../save/{}/N[{}]_R[{}]_E[{}]_iid[{}]_F[{}]_LR[{}]_LD[{}]_B[{}]_RB[{}]_S[{}]_full[{}]_hetero.pkl'.\
		format(args.dataset, args.nodes, args.round, args.local_epoch, args.iid, \
            args.FedDyn, args.lr, args.lr_decay, args.base, args.randomBranch, args.scale, args.full)


    with open(file_name, 'wb') as f:
        pickle.dump(acc_list, f)
