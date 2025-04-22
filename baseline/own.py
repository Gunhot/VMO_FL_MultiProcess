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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from nn_models.resnet import ResNet20, ResNet18

def evaluate_simple(model, test_loader, args, device):
    model.to(device)
    model.eval()

    loss, total, correct = 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)

            _, pred_labels = torch.max(output, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()

            total += len(labels)

        accuracy_multi = correct/total

    
    model.to(torch.device('cpu'))
    
    return accuracy_multi, loss


def create_model(args, teacher=False):

    if args.dataset == 'imagenet':
        model = ResNet20(1000)

        return nn.DataParallel(model)

def get_dataset(args):

    # if args.dataset == 'cifar100':
    #     apply_transform_train = transforms.Compose(
    #         [transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #          transforms.ToTensor(),
    #          transforms.Normalize((0.5071, 0.4867, 0.4408),
    #                               (0.2675, 0.2565, 0.2761)),
    #                               ]
    #     )
    #     apply_transform_test = transforms.Compose(
    #         [transforms.ToTensor(),
    #          transforms.Normalize((0.5071, 0.4867, 0.4408),
    #                               (0.2675, 0.2565, 0.2761)),
    #                               ]
    #     )
    #     dir = '~/FedSD/data/cifar100'
    #     train_dataset = datasets.CIFAR100(dir, train=True, download=True,
    #                                      transform=apply_transform_train)
    #     test_dataset = datasets.CIFAR100(dir, train=False, download=True,
    #                                     transform=apply_transform_test)

    #     return train_dataset, test_dataset

    # if args.dataset == 'tiny-imagenet':
    #     apply_transform_train = transforms.Compose(
    #         [transforms.RandomRotation(20),
    #         transforms.RandomHorizontalFlip(0.5),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
    #                             std=[0.2302, 0.2265, 0.2262])
    #                               ]
    #     )
    #     apply_transform_test = transforms.Compose(
    #         [transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
    #                             std=[0.2302, 0.2265, 0.2262])
    #                               ]
    #     )
    #     dir = '~/FedSD/data/tiny-imagenet/'
    #     train_dataset = datasets.ImageFolder(dir+'train', transform=apply_transform_train)
    #     test_dataset = datasets.ImageFolder(dir+'test', transform=apply_transform_test)

    #     return train_dataset, test_dataset

    if args.dataset == 'imagenet':
        apply_transform_train = transforms.Compose(
            [transforms.Resize((32,32)),
            transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                                  ]
        )
        apply_transform_test = transforms.Compose(
            [
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                                  ]
        )
        dir = '~/scheduler/data/imagenet'
        train_dataset = datasets.ImageNet(dir, split='train',
                                         transform=apply_transform_train)
        test_dataset = datasets.ImageNet(dir, split='val',
                                        transform=apply_transform_test)

        return train_dataset, test_dataset




if __name__ == "__main__":

    import torch.nn as nn
    import queue

    """main"""
    args = arguments.parser()
    device = torch.device("cuda:0")

    print("> Setting:", args)

    # load data
    
    train_dataset, test_dataset = get_dataset(args)

    # print(len(train_dataset))
    # print(len(test_dataset))

    acc_list = list()

    net = create_model(args)
    net.to(device)


    train_loader = DataLoader(train_dataset, batch_size = 768, shuffle = True, num_workers = 16)
    test_loader = DataLoader(test_dataset, batch_size = 768, shuffle = False, num_workers = 16)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    preq = queue.Queue()

    args.round = 5

    for roundIdx in range(args.round+1)[1:]:
        
        start_time = time.time()

        if roundIdx == 100 or roundIdx ==150 or roundIdx == 180:
            for g in optimizer.param_groups:
                g['lr'] = 0.2 * g['lr']


        print("Current Round : {}".format(roundIdx), end=', ')
        net.train()
        net.to(device)

        avg_loss = 0.
        for batch_idx, (images, labels) in enumerate(train_loader):
            
            loss = 0.
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
        
            output = net(images)
            loss += criterion(output, labels)
            loss.backward()
            optimizer.step()

        print(f"Elapsed Time : {time.time()-start_time:.1f}")

        if roundIdx % 1 == 0:
            preq.put({'round': roundIdx, 'model': copy.deepcopy(net.to('cpu'))})
        
   
    torch.save(net.module.state_dict(), "../../save/models/resnet20_imagenet.pt")

    preq.put('kill')

    while True:
        msg = preq.get()

        if msg == 'kill':
            break

        else:
            
            model = msg['model']
            round = msg['round']
            
            acc, loss = evaluate_simple(model, test_loader, args, device)
            print("Round: {} / Accuracy: {}".format(round, acc))
            

