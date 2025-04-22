#! /bin/bash

python main.py --dataset='femnist' --nodes=540 --round=1000 --fraction=0.1 --FedDyn=0 --iid=1 --n_procs=4 --batch_size=50 --lr_decay=0.999 --lr=0.1;
python main.py --dataset='femnist' --nodes=540 --round=1000 --fraction=0.1 --FedDyn=1 --iid=1 --n_procs=4 --batch_size=50 --lr_decay=0.999 --lr=0.1 --step=750;
python main.py --dataset='femnist' --nodes=540 --round=1000 --fraction=0.1 --FedDyn=1 --iid=1 --n_procs=4 --batch_size=50 --lr_decay=0.999 --lr=0.1;

python main.py --dataset='femnist' --nodes=540 --round=1000 --fraction=0.1 --FedDyn=0 --iid=0 --n_procs=4 --batch_size=50 --lr_decay=0.999 --beta=0.1 --lr=0.1;
python main.py --dataset='femnist' --nodes=540 --round=1000 --fraction=0.1 --FedDyn=1 --iid=0 --n_procs=4 --batch_size=50 --lr_decay=0.999 --beta=0.1 --lr=0.1 --step=750;
python main.py --dataset='femnist' --nodes=540 --round=1000 --fraction=0.1 --FedDyn=1 --iid=0 --n_procs=4 --batch_size=50 --lr_decay=0.999 --beta=0.1 --lr=0.1;



