#!/bin/bash


python main.py --dataset='mnist' --kd=0 --FedDyn=0 --base=1;
python main.py --dataset='mnist' --kd=0 --FedDyn=0 --base=2;
python main.py --dataset='mnist' --kd=0 --FedDyn=0 --base=3;

