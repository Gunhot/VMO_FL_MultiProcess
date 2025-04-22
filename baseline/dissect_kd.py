# 0: no kd
# 1 : self distillation
# 2 : kd
# 3 : LS
# 4 : gradient rescaling
# 5 : similarity
# 6 : 4 + 5
# 7 : poorly trained teacher (25% data)
# 8 : light teacher (100% data)

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
    
    return accuracy_multi, None, loss


def create_model(args, teacher=False):
    if args.dataset == 'cifar100':
        if args.student == 0 or teacher:
            model = resnet101(num_class=100)
            return nn.DataParallel(model)
        else:
            model = multi_resnet18_kd(n_blocks=4, num_classes=100)
            return nn.DataParallel(model)
    
    if args.dataset == 'domainnet':
        if args.student == 0 or teacher:
            model = resnet101(num_class=100)
            return nn.DataParallel(model)
        else:
            model = multi_resnet18_224(n_blocks=4, num_classes=345)
            return nn.DataParallel(model)

def create_model_light(args):
    if args.dataset == 'cifar100':
        model = multi_resnet18_kd(n_blocks=1, num_classes=100)
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

    if args.dataset == 'domainnet':
        apply_transform_train = transforms.Compose(
            [transforms.Resize([224,224]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.7490, 0.7390, 0.7179],
                                std=[0.2188, 0.2179, 0.2245])
                                  ]
        )
        apply_transform_test = transforms.Compose(
            [transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.7490, 0.7390, 0.7179],
                                std=[0.2188, 0.2179, 0.2245])
                                  ]
        )
        dir = os.path.dirname(os.path.realpath(__file__))
        dir_split = dir.split('/')
        dir = '/'.join(dir_split[:-2]) 
        
        domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        train_datasets = []
        test_datasets = []

        for domain in domains:
            cur_dir = dir + '/data/' + domain
            train_dataset = datasets.ImageFolder(cur_dir+'_train', transform = apply_transform_train)
            test_dataset = datasets.ImageFolder(cur_dir+'_test', transform = apply_transform_test)
            train_datasets.append(train_dataset)
            test_datasets.append(test_dataset)
        
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        test_dataset = torch.utils.data.ConcatDataset(test_datasets)
    
        #print(len(train_dataset))
        #print(len(test_dataset))
        
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

class Loss_gradient_rescaling(nn.Module):
    def __init__(self, args):
        self.args = args
        super(Loss_gradient_rescaling, self).__init__()

    def forward(self, pred, teacher, label):
        T=self.args.temperature
        K = pred.size(1)

        predict = F.log_softmax(pred/T,dim=1)
        teacher_prob = F.softmax(teacher/T,dim=1)
        target_data = torch.ones_like(pred)

        for i in range(pred.shape[0]):
            true_index = label[i]
            confidence = teacher_prob[i][true_index]
            target_data[i] = target_data[i] * (1 - confidence) / (K-1)
            target_data[i][true_index] = confidence

        with torch.no_grad():
            target = target_data.detach().clone()

        # print(torch.max(F.softmax(pred/T, dim=1)))
        # print(torch.max(target[0]))
        # exit()

        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss

class Loss_similarity(nn.Module):
    def __init__(self, args):
        self.args = args
        super(Loss_similarity, self).__init__()

    def forward(self, pred, weight, label):
        
        T=self.args.temperature

        predict = F.log_softmax(pred/T,dim=1)
        target_data = torch.zeros_like(pred)

        for i in range(pred.shape[0]):
            true_index = label[i]
            weight_true = weight[true_index]
            target_data[i] = torch.matmul(weight_true, weight.T)

        target_data = F.relu(target_data)
        target_data = torch.pow(target_data, 0.3)
        target_data = F.softmax(target_data / 0.3, dim=1)

        with torch.no_grad():
            target = target_data.detach().clone()

        # print(torch.max(F.softmax(pred/T, dim=1)))
        # print(torch.max(target[0]))
        # exit()

        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        

        return loss

class Loss_gr_sim(nn.Module):
    def __init__(self, args):
        self.args = args
        super(Loss_gr_sim, self).__init__()

    def forward(self, pred, teacher, weight, label):
        
        T=self.args.temperature
        K = pred.size(1)

        predict = F.log_softmax(pred/T,dim=1)

        # similarity
        target_data_sim = torch.zeros_like(pred)

        for i in range(pred.shape[0]):
            true_index = label[i]
            weight_true = weight[true_index]
            target_data_sim[i] = torch.matmul(weight_true, weight.T)

        target_data_sim = F.relu(target_data_sim)
        target_data_sim = torch.pow(target_data_sim, 0.3)
        target_data_sim = F.softmax(target_data_sim / 0.3, dim=1)

        # gradient rescaling
        teacher_prob = F.softmax(teacher/T,dim=1)
        target_data_gr = torch.ones_like(pred)

        for i in range(pred.shape[0]):
            true_index = label[i]
            confidence = teacher_prob[i][true_index]
            target_data_gr[i] = target_data_gr[i] * (1 - confidence) / (K-1)
            target_data_gr[i][true_index] = confidence

        target_data = target_data_gr * 0.5 + target_data_sim * 0.5

        with torch.no_grad():
            target = target_data.detach().clone()

        # print(torch.max(F.softmax(pred/T, dim=1)))
        # print(torch.max(target[0]))
        # exit()

        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        

        return loss

# class LSLoss(nn.Module):
#     def __init__(self, args):
#         self.args = args
#         super(LSLoss, self).__init__()

#     def forward(self, pred, label):
#         T=self.args.temperature
#         correct_prob = self.args.ls_prob
#         K = pred.size(1)
#         teacher_soft = torch.ones_like(pred)
#         teacher_soft = teacher_soft*(1-correct_prob)/(K-1)

#         for i in range(pred.shape[0]):
#             teacher_soft[i ,label[i]] = correct_prob

#         predict = F.log_softmax(pred/T,dim=1)

#         target_data = torch.pow(teacher_soft, 1/T)
#         target_data = torch.nn.functional.normalize(target_data, p=1.0)
#         #target_data = F.softmax(teacher_soft/T,dim=1)
        
#         target_data = target_data+10**(-7)
#         with torch.no_grad():
#             target = target_data.detach().clone()

#         loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
#         return loss


if __name__ == "__main__":

    import torch.nn as nn
    import queue
    import random

    """main"""
    args = arguments.parser()

    args.round = 200
    device = torch.device("cuda:0")

    print("> Setting:", args)

    # load data
    
    train_dataset, test_dataset = get_dataset(args)

    # print(len(train_dataset))
    # print(len(test_dataset))

    acc_list = list()

    net = create_model(args)
    net.to(device)

    indices = torch.randperm(len(train_dataset))[:12500]
    train_dataset = torch.utils.data.Subset(train_dataset, indices)

    train_loader = DataLoader(train_dataset, batch_size = 256, shuffle = True, num_workers = 16)
    test_loader = DataLoader(test_dataset, batch_size = 256, shuffle = False, num_workers = 16)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)

    if args.kd == 3:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss()
    
    criterion_kl = KLLoss(args).cuda()
    criterion_gradient_rescaling = Loss_gradient_rescaling(args).cuda()
    criterion_similarity = Loss_similarity(args).cuda()
    criterion_gr_sim = Loss_gr_sim(args).cuda()

    #criterion_ls = LSLoss(args).cuda()
    #scheduler = ReduceLROnPlateau(optimizer, verbose=True, patience=10)
    preq = queue.Queue()

    if args.kd == 2 or args.kd == 4 or args.kd == 6:
        teacher = create_model(args, teacher=True)
        teacher.module.load_state_dict(torch.load("../../save/Models/CIFAR100.pt"))
        teacher.to(device)
    
    if args.kd == 7:
        teacher = create_model(args, teacher=True)
        teacher.module.load_state_dict(torch.load("../../save/Models/CIFAR100_poor.pt"))
        teacher.to(device)
        

    if args.kd == 8:
        teacher = create_model_light(args)
        teacher.module.load_state_dict(torch.load("../../save/Models/CIFAR100_light.pt"))
        teacher.to(device)
        

    if args.kd == 5 or args.kd == 6:
        teacher = create_model(args, teacher=True)
        teacher.module.load_state_dict(torch.load("../../save/Models/CIFAR100.pt"))
        logit_weight = teacher.module.state_dict()['fc.weight'].detach().clone()
        logit_weight = torch.nn.functional.normalize(logit_weight, p=2.0)
        teacher.to(device)

    for roundIdx in range(args.round+1)[1:]:
        
        start_time = time.time()

        if roundIdx == 100 or roundIdx ==150 or roundIdx == 180:
            for g in optimizer.param_groups:
                g['lr'] = 0.2 * g['lr']


        args.consistency_rampup = int(args.round * 0.3)
        current = np.clip(roundIdx, 0.0, args.consistency_rampup)
        phase = 1.0 - current / args.consistency_rampup
        consistency_weight = float(np.exp(-5.0 * phase * phase))

        print("Current Round : {}".format(roundIdx), end=', ')
        net.train()
        net.to(device)

        avg_loss = 0.
        for batch_idx, (images, labels) in enumerate(train_loader):
            
            loss = 0.
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)

            if args.student == 0:
                #print("new batch")
                output = net(images)
                loss += criterion(output, labels) ## true label, branch output loss
                #avg_loss += loss.detach()
            
            else:
                output_list, _ = net(images)
                
                if args.kd == 2 or args.kd == 4 or args.kd == 6 or args.kd ==7:
                    with torch.no_grad():
                        teacher_output = teacher(images)
                
                elif args.kd == 8:
                    with torch.no_grad():
                        teacher_output, _ = teacher(images)
                
                for i, branch_output in enumerate(output_list):
                    
                    if i != len(output_list) -1:
                        continue

                    if args.kd == 0 or args.kd == 1 or args.kd == 3:
                        coefficient = 1.0
                    else:
                        coefficient = 0.7

                    loss += coefficient * criterion(branch_output, labels)

                    if args.kd == 1:
                        for j in range(len(output_list)):
                            if j == i:
                                continue
                            else:
                                loss += consistency_weight * criterion_kl(branch_output, output_list[j].detach()) / (len(output_list) - 1)

                    if args.kd == 2 or args.kd ==7:
                        loss += 0.3 * criterion_kl(branch_output, teacher_output.detach())

                    elif args.kd == 8:
                        loss += 0.3 * criterion_kl(branch_output, teacher_output[0].detach())
                    
                    elif args.kd == 4:
                        loss += 0.3 * criterion_gradient_rescaling(branch_output, teacher_output.detach(), labels.detach())
                    
                    elif args.kd == 5:
                        loss += 0.3 * criterion_similarity(branch_output, logit_weight, labels.detach())
                    
                    elif args.kd == 6:
                        loss += 0.3 * criterion_gr_sim(branch_output, teacher_output.detach(), logit_weight, labels.detach())

            loss.backward()
            optimizer.step()

        #print("Loss : {:.2f}".format(avg_loss / (batch_idx + 1)), end = ', ')
        print(f"Elapsed Time : {time.time()-start_time:.1f}")

        # scheduler.step(avg_loss)

        if roundIdx % 5 == 0:
            preq.put({'round': roundIdx, 'model': copy.deepcopy(net.to('cpu'))})
        
        # net.eval()
        # loss, total, correct = 0.0, 0.0, 0.0

        # with torch.no_grad():
        #     for batch_idx, (images, labels) in enumerate(test_loader):
        #         images, labels = images.to(device), labels.to(device)

        #         output = net(images)[0][-1]
        #         loss_batch = criterion(output, labels)
        #         loss += loss_batch.item()

        #         _, pred_labels = torch.max(output, 1)
        #         pred_labels = pred_labels.view(-1)
        #         correct += torch.sum(torch.eq(pred_labels, labels)).item()
        #         total += len(labels)

        #     accuracy = correct/total

        # print('Accuracy : {}'.format(accuracy))
        # acc_list.append(accuracy)

    
    preq.put('kill')
    global_acc = {'multi':list(), 'single':list()}

    while True:
        msg = preq.get()

        if msg == 'kill':
            break

        else:
            
            model = msg['model']
            round = msg['round']
            
            if args.student == 0:
                acc_multi, acc_single, loss = evaluate(model, test_loader, args, device)
            else:
                acc_multi, acc_single, loss = evaluate(model, test_loader, args, device)

            global_acc['multi'].append(acc_multi)
            global_acc['single'].append(acc_single)
        
            print("Round: {} / Accuracy: {}".format(round, acc_multi))


    file_name = '../../save/{}/K[{}]_ST[{}]_kd_25.pkl'.\
        format(args.dataset, args.kd, args.student)

    with open(file_name, 'wb') as f:
        pickle.dump([global_acc], f)

