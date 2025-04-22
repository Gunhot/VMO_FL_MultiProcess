import numpy as np
from torch.utils.data import Subset

def iid(dataset, num_nodes):
    np.random.seed(41)
    num_sample = int(len(dataset)/(num_nodes))
    # num_sample = int(len(dataset)/(num_nodes)) // 10

    dict_nodes = {}
    index = [i for i in range(len(dataset))]
    for i in range(num_nodes):
        dict_nodes[i] = np.random.choice(index, num_sample,
                                         replace=False)
        index = list(set(index)-set(dict_nodes[i]))
    return dict_nodes


def iid_waterbirds(dataset_1, dataset_2, num_nodes):

    np.random.seed(41)
    num_sample_1 = int(len(dataset_1)/(num_nodes // 2))
    num_sample_2 = int(len(dataset_2)/(num_nodes // 2))

    dict_nodes = {}
    index_1 = [i for i in range(len(dataset_1))]
    for i in range(num_nodes // 2):
        dict_nodes[i] = np.random.choice(index_1, num_sample_1,
                                         replace=False)
        index_1 = list(set(index_1)-set(dict_nodes[i]))
    
    index_2 = [i for i in range(len(dataset_2))]
    for i in range(num_nodes // 2, num_nodes):
        dict_nodes[i] = np.random.choice(index_2, num_sample_2,
                                         replace=False)
        index_2 = list(set(index_2)-set(dict_nodes[i]))

    return dict_nodes

def noniid_num(dataset, num_nodes, beta):
    np.random.seed(41)
    # num_sample = int(len(dataset)/(num_nodes))
    proportions = len(dataset) * np.random.dirichlet(np.repeat(beta, num_nodes))
    dict_nodes = {}
    total_sum = 0

    index = [i for i in range(len(dataset))]
    for i in range(num_nodes):
        dict_nodes[i] = np.random.choice(index, int(proportions[i]),
                                         replace=False)
        index = list(set(index)-set(dict_nodes[i]))

        total_sum += len(dict_nodes[i])
        # print(len(dict_nodes[i]))

    #print(total_sum)
    #exit()

    return dict_nodes


def split_dataset(dataset):
    labels = np.array([label for _, label in dataset])
    K = np.max(labels) + 1
    np.random.seed(31)
    idx_1 = []
    idx_2 = []

    for k in range(K // 2):
        idx_k = np.where(labels == k)[0]
        idx_1.extend(idx_k)

    for k in range(K // 2, K):
        idx_k = np.where(labels == k)[0]
        idx_2.extend(idx_k)

    dataset_1 = Subset(dataset,idx_1)
    dataset_2 = Subset(dataset,idx_2)

    return dataset_1, dataset_2

def noniid_catastrophic(dataset, num_nodes):

    dataset_1, dataset_2 = split_dataset(dataset)
    
    dict_1 = iid(dataset_1, num_nodes // 2)
    dict_2 = iid(dataset_2, num_nodes // 2)

    return dict_1, dict_2

def noniid(dataset, num_nodes, beta):
    labels = np.array([label for _, label in dataset])
    min_size = 0
    K = np.max(labels) + 1
    N = labels.shape[0]
    net_dataidx_map = {}
    n_nets = num_nodes
    np.random.seed(31)

    while min_size < 10:

        idx_batch = [[] for _ in range(n_nets)]
        # for each class in the dataset
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_nets))
            ## Balance
            proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    data_min = 100000
    data_max = 0

    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = np.array(idx_batch[j])
        data_min = min(data_min, len(net_dataidx_map[j]))
        data_max = max(data_max, len(net_dataidx_map[j]))
        # print(len(net_dataidx_map[j]))

    # print(data_min)
    # print(data_max)
    # exit()

    return net_dataidx_map


def noniid_nlp(dataset, num_nodes, beta):

    min_size = 0
    N = len(dataset)
    net_dataidx_map = {}
    n_nets = num_nodes
    np.random.seed(31)

    while min_size < 640:
        idx_batch = [[] for _ in range(n_nets)]
        # for each class in the dataset
    
        idx_k = np.arange(N)
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(beta, n_nets))
        ## Balance
        proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
        proportions = proportions/proportions.sum()
        proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
        idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
        min_size = min([len(idx_j) for idx_j in idx_batch])
        # print(min_size)
        

    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = np.array(idx_batch[j])
    
    return net_dataidx_map