import torch
from model_avg import model_avg
from nn_models.vgg import VGG9
from nn_models.resnet import ResNet20, ResNet18
from nn_models.convnet import convnet
import torchvision
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from nn_models.squeezenet import squeezenet
#gunhot
from nn_models.transformers.gpt2 import GPT2Medium
#gunhot

class Server:
    def __init__(self, args):
        self.model = None
        self.param_sum = dict()
        self.param_count = dict()
        self.args = args
        self.h = [dict() for i in range(args.nodes)]
        self.h_size = []
        self.h_size_avg = []
        self.h_client_0 = []
        self.pseudo_gradients = [dict() for i in range(args.nodes)]
        self.pseudo_gradients_size = []
        self.pseudo_gradients_size_avg = []
        self.lamb = 1.0
        self.active_clients = []

    def get_model(self):
        return self.model 

    def initialize_h(self):
        for item in self.h:
            for k, v in self.model.state_dict().items():
                item[k] = torch.zeros(v.shape)

        for item in self.pseudo_gradients:
            for k, v in self.model.state_dict().items():
                item[k] = torch.zeros(v.shape)

    def update_node_info(self, weight, node_id):
        
        origin = self.model.state_dict()
        self.active_clients.append(node_id)

        for k in weight.keys():
            if self.param_sum[k] is None:
                self.param_sum[k] = weight[k]
            else:
                self.param_sum[k] += weight[k]
            self.param_count[k] += 1

            self.h[node_id][k] = self.lamb * (self.h[node_id][k] - self.args.alpha * (weight[k] - \
                    origin[k]) / self.args.nodes)
            
            self.pseudo_gradients[node_id][k] = weight[k] - origin[k]
          
    def avg_parameters(self):
        origin = self.model.state_dict()
        h = dict()
        h_active = dict()
        server_h = []
        server_pseudo_gradients = []
        pseudo_gradients = dict()
        for k in origin.keys():

            if 'weight' not in k and 'bias' not in k:
                continue

            h[k] = torch.zeros(origin[k].shape)
            h_active[k] = torch.zeros(origin[k].shape)
            pseudo_gradients[k] = torch.zeros(origin[k].shape)
        
            for i in range(self.args.nodes):
                h[k] += self.h[i][k]
            
            for i in self.active_clients:
                h_active[k] += self.h[i][k]
                pseudo_gradients[k] += self.pseudo_gradients[i][k]
            
            server_h.append(h_active[k].flatten().clone())
            server_pseudo_gradients.append(pseudo_gradients[k].flatten().clone())
            
        server_h = torch.cat(server_h)
        server_pseudo_gradients = torch.cat(server_pseudo_gradients)
        self.h_size.append(torch.norm(server_h, p=2))
        self.pseudo_gradients_size.append(torch.norm(server_pseudo_gradients, p=2))

        h_size_sum = 0
        pseudo_gradients_size_sum = 0

        for i, indiv_h in enumerate(self.h):

            if i in self.active_clients:
                indiv_h_lst = []
                for k in origin.keys():
                    if 'weight' not in k and 'bias' not in k:
                        continue
                    
                    indiv_h_lst.append(indiv_h[k].flatten().clone())

                indiv_h_lst = torch.cat(indiv_h_lst)
                h_size_sum += torch.norm(indiv_h_lst, p=2)

        self.h_size_avg.append(h_size_sum / self.args.nodes)

        for i, indiv_pseudo_gradients in enumerate(self.pseudo_gradients):
            if i in self.active_clients:
                indiv_pseudo_gradients_lst = []
                for k in origin.keys():
                    if 'weight' not in k and 'bias' not in k:
                        continue
                    
                    indiv_pseudo_gradients_lst.append(indiv_pseudo_gradients[k].flatten().clone())

                indiv_pseudo_gradients_lst = torch.cat(indiv_pseudo_gradients_lst)
                pseudo_gradients_size_sum += torch.norm(indiv_pseudo_gradients_lst, p=2)

        self.pseudo_gradients_size_avg.append(pseudo_gradients_size_sum / self.args.nodes)

        avg_parameters = model_avg(self.param_sum, self.param_count, self.args, h, origin)
    
        state_dict = {k: avg_parameters[k] for k in self.model.state_dict().keys()}
        self.model.load_state_dict(state_dict)
        
        for k in self.model.state_dict().keys():
            self.param_sum[k] = None
            self.param_count[k] = 0

    def set_initial_model(self):
        
        if self.args.dataset == 'femnist':
            self.model = squeezenet(62)

        # if self.args.dataset == 'cifar10':
        #     if self.args.pretrained:
        #         self.model = ResNet20(1000)
        #         self.model.load_state_dict(torch.load("../save/models/imagenet.pt"))
        #         self.model.linear = nn.Linear(64, 10)

        #     else:
        #         self.model = ResNet20()
                
            # self.model = squeezenet()

        if self.args.dataset == 'cifar100':
            if self.args.pretrained:
                self.model = ResNet20(1000)
                self.model.load_state_dict(torch.load("../save/models/resnet20_imagenet.pt"))
                self.model.linear = nn.Linear(64,100)
            else:
                num_classes = 100
                self.model = ResNet20(num_classes)
        
        if self.args.dataset == 'tiny-imagenet':
            num_classes = 200
            self.model = ResNet18(num_classes)
          

        if self.args.dataset == 'waterbirds':
            num_classes = 2
            self.model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
            # self.model = torchvision.models.resnet18()
            d = self.model.fc.in_features
            self.model.fc = nn.Linear(d, num_classes)
        
        #gunhot
        elif self.args.dataset == 'viggo':
            print("Gunhot VIGGO OKAY")
            self.model = GPT2Medium()
            print("Gunhot VIGGO OKAY")
        #gunhot
        # elif self.args.dataset == 'tiny-imagenet':
        #     num_classes = 200
        #     if self.args.base == -1:
        #         for i in range(self.args.num_branch):
        #             self.models.append(multi_resnet34_tiny(n_blocks=i+1, num_classes=num_classes, norm=self.args.norm))
        
        #     else:
        #         for i in range(self.args.base+1):
        #             self.models.append(multi_resnet34_tiny(n_blocks=i+1, num_classes=num_classes, norm=self.args.norm))

        # elif self.args.dataset == 'domainnet':
        #     num_classes = 345
        #     if self.args.base == -1:
        #         for i in range(self.args.num_branch):
        #             self.models.append(multi_resnet18_224(n_blocks=i+1, num_classes=num_classes, norm=self.args.norm))
        
        #     else:
        #         for i in range(self.args.base+1):
        #             self.models.append(multi_resnet18_224(n_blocks=i+1, num_classes=num_classes, norm=self.args.norm))

        # elif self.args.dataset == 'wikitext-2':
        #     if self.args.base == -1:
        #         for i in range(self.args.num_branch):
        #             self.models.append(multi_transformer(n_blocks=i+1))
        #     else:
        #         for i in range(self.args.base+1):
        #             self.models.append(multi_transformer(n_blocks=i+1))

        #gunhot
        self.param_sum = {k: None for k in self.model.state_dict().keys()}
        self.param_count = {k: 0 for k in self.model.state_dict().keys()}
        #gunhot

        # for model in self.models:
        #     state_dict = {k: self.models[-1].state_dict()[k] for k in model.state_dict().keys()}
        #     model.load_state_dict(state_dict)




