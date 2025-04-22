import argparse

def parser():
    parser = argparse.ArgumentParser(description='Some hyperparameters')

    parser.add_argument('--nodes', type=int, default=100,
                        help='total number of nodes')
    parser.add_argument('--fraction', type=float, default=0.1,
                        help='ratio of participating node')
    parser.add_argument('--round', type=int, default=1000,
                        help='number of rounds')
    parser.add_argument('--local_epoch',  type=int, default=5,
                        help='number of local_epoch')
    parser.add_argument('--dataset',  type=str, default='cifar100',
                        help='type of dataset')
    parser.add_argument('--batch_size', type=int, default=50, 
                        help='size of batch')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=1e-5,
                        help='0.992, 0.998')
    parser.add_argument('--iid', type=int, default=1,
                        help='iid')
    parser.add_argument('--n_procs', type=int, default=2,
                        help='number of processes per GPU')
    parser.add_argument('--FedDyn', type=int, default=1,
                        help='whether FedDyn Algorithm')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='alpha for FedDyn')
    parser.add_argument('--beta', type=float, default= 0.1,
                        help='beta for non iid dirichlet dist')
    parser.add_argument('--norm', type=str, default='bn',
                        help='Default: Batch Normalization')
    parser.add_argument('--sequence_length', type=int, default=64,
                        help='sequence_length')
    parser.add_argument('--step', type=int, default=-1,
                        help = 'Step down round')
    parser.add_argument('--pretrained', type=int, default=0,
                        help = 'whether pretrained')
    parser.add_argument('--lamb', type=float, default=1.0,
                        help='hyperparameter lambda')
    parser.add_argument('--hidden', type=int, default=0,
                        help='hidden layer')
    parser.add_argument('--h_updated', type=int, default=1, #0: no update #1: push #2: pull
                        help='h updated')
    parser.add_argument('--client3', type=int, default=1,
                        help='client3')
    parser.add_argument('--h_updated_value', type=float, default=3.0,
                        help='h updated value')
    parser.add_argument('--max_length', type=int, default=64, 
                        help='Max token length for tokenizer input')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser()
    print(args)
