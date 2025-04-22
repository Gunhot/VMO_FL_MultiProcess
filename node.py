import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.utils.data import DataLoader, Subset
import numpy as np
# from dataLoader.waterbirds import DRODataset

class Client:
    def __init__(self, nodeID, prev_grads, args, node_indices=None):
        self.nodeID = nodeID
        self.__node_indices = node_indices
        self.__args = args
        # if args.FedDyn == 1:
        #     self.__prev_grads = prev_grads
        self.__prev_grads = prev_grads
     
    def get_prev_grads(self):
        return copy.deepcopy(self.__prev_grads)

    def set_prev_grads(self, prev_grads):
        self.__prev_grads = prev_grads
    
    def train(self, device, lr, model, train_dataset, round):
        #gunhot
        if self.__args.dataset == 'viggo':
            print("Gunhot VIGGO CLIENT LEARNING")
            train_loader = DataLoader(train_dataset, batch_size=self.__args.batch_size, shuffle=True)

            model.train()
            model.to(device)

            lamb = 1.0
            if self.__args.step > 0 and round >= self.__args.step:
                self.__args.FedDyn = 0

            if self.__args.FedDyn == 1:
                with torch.no_grad():
                    server_params = {k: param.flatten().clone() for (k, param) in model.named_parameters()}
                for k, param in model.named_parameters():
                    self.__prev_grads[k] = self.__prev_grads[k].to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

            for _ in range(self.__args.local_epoch):
                for batch in train_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    optimizer.zero_grad()
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs[0]

                    # FedDyn 적용
                    if self.__args.FedDyn == 1:
                        for k, param in model.named_parameters():
                            curr_param = param.flatten()
                            lin_penalty = torch.dot(curr_param, self.__prev_grads[k])
                            loss -= lin_penalty * lamb
                            quad_penalty = self.__args.alpha / 2.0 * torch.sum(torch.square(curr_param - server_params[k]))
                            loss += quad_penalty * lamb

                    loss.backward()
                    optimizer.step()

            if self.__args.FedDyn == 1:
                with torch.no_grad():
                    for k, param in model.named_parameters():
                        curr_param = param.flatten().clone()
                        self.__prev_grads[k] = lamb * (self.__prev_grads[k] - self.__args.alpha * (curr_param - server_params[k]))
                        self.__prev_grads[k] = self.__prev_grads[k].to(torch.device('cpu'))
            print("Gunhot VIGGO CLIENT LEARNING DONE")
            model.to(torch.device('cpu'))
            return model.state_dict()
        #gunhot
        else:
            if self.__node_indices is None:
                train_loader = DataLoader(train_dataset, batch_size=self.__args.batch_size, shuffle=True)

            else:  
                train_loader = DataLoader(Subset(train_dataset, self.__node_indices), \
                    batch_size=self.__args.batch_size, shuffle=True)
            
            model.train()
            model.to(device)
            lamb = 1.0

            if self.__args.step > 0:
                if round >= self.__args.step:
                    self.__args.FedDyn = 0

            if self.__args.FedDyn == 1:
                with torch.no_grad():
                    server_params = {k: param.flatten().clone() for (k, param) in model.named_parameters()}

                for k, param in model.named_parameters():
                    self.__prev_grads[k] = self.__prev_grads[k].to(device)
            

            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-3)
            criterion = nn.CrossEntropyLoss()

            local_loss = 0.
            count = 0
            
            for _ in range(self.__args.local_epoch):
                for _, (images, labels) in enumerate(train_loader):
                    images, labels = images.to(device), labels.to(device)
                    loss = 0.
                    optimizer.zero_grad()
                    output = model(images)
                    loss += criterion(output, labels) ## true label, branch output loss
                    
                    local_loss += loss.detach().clone().to(torch.device('cpu'))
                    count += 1

                    if self.__args.FedDyn == 1:
                        for k, param in model.named_parameters():
    
                            curr_param = param.flatten()
                            ## linear penalty

                            lin_penalty = torch.dot(curr_param, self.__prev_grads[k])
                            loss -= lin_penalty * lamb

                            ## quadratic penalty
                            
                            quad_penalty = self.__args.alpha/2.0 * torch.sum(torch.square(curr_param - server_params[k]))
                            loss += quad_penalty * lamb

                    loss.backward()
                    optimizer.step()

            if self.__args.FedDyn == 1:
                # update prev_grads
                with torch.no_grad():
                    for k, param in model.named_parameters():
                        curr_param = param.flatten().clone()
                        # self.__prev_grads[k] = self.__args.lamb * (self.__prev_grads[k] - self.__args.alpha * (curr_param - server_params[k]))
                        self.__prev_grads[k] = lamb * (self.__prev_grads[k] - self.__args.alpha * (curr_param - server_params[k]))
                        self.__prev_grads[k] = self.__prev_grads[k].to(torch.device('cpu'))

            model.to(torch.device('cpu'))

            # if select_level(self.nodeID, self.__args) == 3:
            #     self.teacher.to(torch.device('cpu'))

            weight = model.state_dict()

            return copy.deepcopy(weight)
        
        
