# from multiprocessing import Pool
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import pickle
import copy
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import arguments
from dataLoader import dataset
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms, models
from nn_models.resnet_sbn import resnet101, resnet18
from nn_models.resnet import multi_resnet18_kd
from nn_models.resnet_224 import multi_resnet18_224
from torch.optim.lr_scheduler import ReduceLROnPlateau

def evaluate(model, test_loader, args, device):
    model.to(device)
    model.eval()

    loss, total, correct_multi= 0.0, 0.0, 0.0
    accuracy_single_list = list()
    
    for i in range(args.num_branch):
        accuracy_single_list.append(0)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            output_list, _ = model(images)

            ensemble_output = torch.stack(output_list, dim=2)
            ensemble_output = torch.sum(ensemble_output, dim=2) / len(output_list)
            
            _, pred_labels_multi = torch.max(ensemble_output, 1)
            pred_labels_multi = pred_labels_multi.view(-1)
            correct_multi += torch.sum(torch.eq(pred_labels_multi, labels)).item()

            for i, single in enumerate(output_list):  
                _, pred_labels_single = torch.max(single, 1)
                pred_labels_single = pred_labels_single.view(-1)
                accuracy_single_list[i] += torch.sum(torch.eq(pred_labels_single, labels)).item()
                
            total += len(labels)

        accuracy_multi = correct_multi/total

        for i in range(len(accuracy_single_list)):
            accuracy_single_list[i] /= total
        
    model.to(torch.device('cpu'))
    
    return accuracy_multi, accuracy_single_list, loss

# def evaluate_simple(model, test_loader, args, device):
#     model.to(device)
#     model.eval()

#     loss, total, correct = 0.0, 0.0, 0.0
    
#     with torch.no_grad():
#         for batch_idx, (images, labels) in enumerate(test_loader):
#             images, labels = images.to(device), labels.to(device)
#             output = model(images)

#             _, pred_labels = torch.max(output, 1)
#             pred_labels = pred_labels.view(-1)
#             correct += torch.sum(torch.eq(pred_labels, labels)).item()

#             total += len(labels)

#         accuracy_multi = correct/total

    
#     model.to(torch.device('cpu'))
    
#     return accuracy_multi, None, loss


def create_model(args, blocks):
    if args.dataset == 'cifar100':
        model = multi_resnet18_kd(n_blocks=blocks, num_classes=100)
        # model = resnet18(100, 1.0, track=True, scale=False)
        return nn.DataParallel(model)
     

def get_dataset(args):

    if args.dataset == 'cifar100':
        apply_transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408),
                                  (0.2675, 0.2565, 0.2761)),
                                  ]
        )
        apply_transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408),
                                  (0.2675, 0.2565, 0.2761)),
                                  ]
        )
        dir = '~/FedSD/data/cifar100'
        train_dataset = datasets.CIFAR100(dir, train=True, download=True,
                                         transform=apply_transform_train)
        test_dataset = datasets.CIFAR100(dir, train=False, download=True,
                                        transform=apply_transform_test)

        return train_dataset, test_dataset


class KLLoss(nn.Module):
    def __init__(self, args):
        self.args = args
        super(KLLoss, self).__init__()

    def forward(self, pred, label):
        T=self.args.temperature
        
        predict = F.log_softmax(pred/T,dim=1)
        target_data = F.softmax(label/T,dim=1)
        target_data =target_data+10**(-7)
        with torch.no_grad():
            target = target_data.detach().clone()

        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss



if __name__ == "__main__":

    import torch.nn as nn
    import queue

    """main"""
    args = arguments.parser()
    device = torch.device("cuda:0")

    print("> Setting:", args)

    # load data
    
    train_dataset, test_dataset = get_dataset(args)

    acc_list = list()

    net = create_model(args, blocks=2)
    net.to(device)


    train_loader = DataLoader(train_dataset, batch_size = 256, shuffle = True, num_workers = 16)
    test_loader = DataLoader(test_dataset, batch_size = 256, shuffle = False, num_workers = 16)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    criterion_kl = KLLoss(args).cuda()
    
    preq = queue.Queue()
    args.round = 21
    
    with torch.no_grad():

        for roundIdx in range(args.round+1)[1:]:
            
            if roundIdx == 2:
                begin_time = time.time()

            start_time = time.time()

            print("Current Round : {}".format(roundIdx), end=', ')
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                
                loss = 0.
                optimizer.zero_grad()
                images, labels = images.to(device), labels.to(device)
            
                output_list, _ = net(images)
                # loss += criterion(output_list[-1], labels)
                
                # for i, branch_output in enumerate(output_list):
                #     loss += criterion(branch_output, labels)

                #     if args.kd == 1:
                #         for j in range(len(output_list)):
                #             if j == i:
                #                 continue
                #             else:
                #                 loss += criterion_kl(branch_output, output_list[j].detach()) / (len(output_list) - 1)

                # loss.backward()
                # optimizer.step()


            print(f"Elapsed Time : {time.time()-start_time:.1f}")

    avg_time = (time.time() - begin_time) / 20
    print(avg_time)
    print(avg_time * 100 / 60)

