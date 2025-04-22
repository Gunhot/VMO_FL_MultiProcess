#! /bin/bash

# python main.py --dataset='tiny-imagenet' --nodes=100 --round=1000 --fraction=0.1 --FedDyn=0 --iid=1 --n_procs=4 --batch_size=50 --lr_decay=0.999 --lr=0.1;
# python main.py --dataset='tiny-imagenet' --nodes=100 --round=1000 --fraction=0.1 --FedDyn=1 --iid=1 --n_procs=4 --batch_size=50 --lr_decay=0.999 --lr=0.1 --step=750;
# python main.py --dataset='tiny-imagenet' --nodes=100 --round=1000 --fraction=0.1 --FedDyn=1 --iid=1 --n_procs=4 --batch_size=50 --lr_decay=0.999 --lr=0.1;

# python main.py --dataset='tiny-imagenet' --nodes=10 --round=300 --fraction=1.0 --FedDyn=0 --iid=1 --n_procs=4 --batch_size=50 --lr_decay=0.99  --lr=0.1 --local_epoch=1;
# python main.py --dataset='tiny-imagenet' --nodes=10 --round=300 --fraction=1.0 --FedDyn=1 --iid=1 --n_procs=4 --batch_size=50 --lr_decay=0.99  --lr=0.1 --step=225 --alpha=0.1 --local_epoch=1;
python main.py --dataset='tiny-imagenet' --nodes=10 --round=300 --fraction=1.0 --FedDyn=1 --iid=1 --n_procs=4 --batch_size=50 --lr_decay=0.99  --lr=0.1 --alpha=0.1 --local_epoch=1;

python main.py --dataset='tiny-imagenet' --nodes=10 --round=300 --fraction=1.0 --FedDyn=0 --iid=0 --n_procs=4 --batch_size=50 --lr_decay=0.99 --beta=0.1 --lr=0.1 --local_epoch=1;
python main.py --dataset='tiny-imagenet' --nodes=10 --round=300 --fraction=1.0 --FedDyn=1 --iid=0 --n_procs=4 --batch_size=50 --lr_decay=0.99 --beta=0.1 --lr=0.1 --step=225 --alpha=0.1 --local_epoch=1;
python main.py --dataset='tiny-imagenet' --nodes=10 --round=300 --fraction=1.0 --FedDyn=1 --iid=0 --n_procs=4 --batch_size=50 --lr_decay=0.99 --beta=0.1 --lr=0.1 --alpha=0.1 --local_epoch=1;

python main.py --dataset='tiny-imagenet' --nodes=100 --round=1000 --fraction=0.1 --FedDyn=0 --iid=0 --n_procs=3 --batch_size=50 --lr_decay=0.999 --beta=0.1 --lr=0.1;
python main.py --dataset='tiny-imagenet' --nodes=100 --round=1000 --fraction=0.1 --FedDyn=1 --iid=0 --n_procs=3 --batch_size=50 --lr_decay=0.999 --beta=0.1 --lr=0.1 --step=750;
python main.py --dataset='tiny-imagenet' --nodes=100 --round=1000 --fraction=0.1 --FedDyn=1 --iid=0 --n_procs=3 --batch_size=50 --lr_decay=0.999 --beta=0.1 --lr=0.1;


