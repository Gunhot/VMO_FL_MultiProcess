#!/bin/bash

python heteroFL.py --dataset='cifar100'; 
python heteroFL.py --dataset='cifar100' --base=0;
python heteroFL.py --dataset='cifar100' --base=1;
python heteroFL.py --dataset='cifar100' --base=2;
python heteroFL.py --dataset='cifar100' --base=3;
python heteroFL.py --dataset='tiny-imagenet'; 
python heteroFL.py --dataset='tiny-imagenet' --base=0;
python heteroFL.py --dataset='tiny-imagenet' --base=1;
python heteroFL.py --dataset='tiny-imagenet' --base=2;
python heteroFL.py --dataset='tiny-imagenet' --base=3;
