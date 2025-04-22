import torch
import numpy as np
import time
import arguments
import copy
import random
import os
import torch.multiprocessing as mp
import queue

from node import Client
from server import Server
from workers import *
from dataLoader.dataLoaders import getNodeIndicies
from nn_models.transformers.gpt2 import GPT2Medium

if __name__ == "__main__":

    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    mp.set_start_method('spawn')
    if torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
        devices = [torch.device("cuda:{}".format(i)) for i in range(n_devices)]
        cuda = True
        print(f"GPU 개수: {n_devices}")
    else:
        n_devices = 1
        devices = [torch.device('cpu')]
        cuda = False

    os.environ["OMP_NUM_THREADS"] = "1"

    args = arguments.parser()
    
    num_nodes = args.nodes
    num_round = args.round
    num_local_epoch = args.local_epoch
    
    print("> Setting:", args)

    nodeIDs = [i for i in range(num_nodes)]
    lambs = [[] for i in range(num_nodes)]

    #gunhot
    if args.dataset != "femnist" and args.dataset != "viggo":
        nodeindices = getNodeIndicies(nodeIDs, num_nodes, args)
    #gunhot
    n_train_processes = n_devices * args.n_procs
    n_test_processes = 1
    trainIDs = ["Train Worker : {}".format(i) for i in range(n_train_processes)]
    testIDs = ["Test Worker : {}".format(i) for i in range(n_test_processes)]
    
    trainQ = mp.Queue(maxsize=10)
    resultQ = mp.Queue()
    testQ = mp.Queue(maxsize=10)
    # preQ = queue.Queue()
    # if args.client3 == 1:
    #     preClient3Q_after_train = queue.Queue()
    #     preClient3Q_before_train = queue.Queue()

    # create test process
    processes = []
    for i, testID in enumerate(testIDs):
        p = mp.Process(target=gpu_test_worker, args=(testID, testQ, devices[i%n_devices], args))
        p.start()
        processes.append(p)
        time.sleep(0.1)

    # create train processes
    for i, trainID in enumerate(trainIDs):
        p = mp.Process(target=gpu_train_worker, args=(trainID, trainQ, resultQ, devices[i%n_devices], args))
        p.start()
        processes.append(p)
        time.sleep(0.1)

    # create pseudo server
    server = Server(args)
    server.set_initial_model()
    print("Gunhot VIGGO Initialized")
    # for FedDyn optimizer
    prev_grads = {}
    with torch.no_grad():
        prev_grads = {k: torch.zeros(v.numel()) for (k, v) in server.model.named_parameters()}

    # create pseudo clients
    nodes = []
    for i, nodeID in enumerate(nodeIDs):
        if args.dataset == 'viggo':
            nodes.append(Client(nodeID, copy.deepcopy(prev_grads), args))
            print(f"Client {nodeID} initialized")
        elif args.dataset == 'femnist':
            nodes.append(Client(nodeID, copy.deepcopy(prev_grads), args))
        else:
            nodes.append(Client(nodeID, copy.deepcopy(prev_grads), args, nodeindices[nodeID]))
    
    lr = args.lr
    prev_grads_list = [[prev_grads] for i in range(num_nodes)]
    
    # initialize h value for FedDyn optimizer
    server.initialize_h()
    
    # gunhot
    # Client 3의 모델을 서버 모델과 동일한 구조로 초기화
    if args.client3 == 1:
        last_client3_model_after_train = GPT2Medium()  # 서버 모델의 구조 복사
        last_client3_model_before_train = GPT2Medium()  # 서버 모델의 구조 복사
    # gunhot

    # Create a directory to save models if it doesn't exist
    model_dir = "checkpoints"
    os.makedirs(model_dir, exist_ok=True)

    # Initialize counters for saved models
    preQ_count = 0
    preClient3Q_after_train_count = 0
    preClient3Q_before_train_count = 0

    for roundIdx in range(args.round+1)[1:]:
        
        if roundIdx % 5 == 0:
            print(f"Round {roundIdx}", end=', ')
        
        if  args.step > 0:
            if roundIdx >= args.step:
                server.args.FedDyn = 0
                args.FedDyn = 0
              
        cur_time = time.time()
        
        lr *= args.lr_decay

        # Randomly selected clients
        n_trainees = int(num_nodes*args.fraction)
        trainees = [nodes[i] for i in np.random.choice(np.arange(num_nodes), n_trainees, replace=False)]
        count = 0
        
        for i, node in enumerate(trainees):
            #gunhot
            temp_server_model = GPT2Medium()
            temp_server_model.load_state_dict(server.model.state_dict())

            if args.h_updated != 0:
                client_id = node.nodeID 
                print(f"client_id: {client_id}")
                with torch.no_grad():
                    for k in temp_server_model.state_dict().keys():
                        if 'weight' not in k and 'bias' not in k:
                            continue
                        if args.h_updated == 1:
                            temp_server_model.state_dict()[k] += args.h_updated_value * server.h[client_id][k] / args.alpha
                        elif args.h_updated == 2:
                            temp_server_model.state_dict()[k] -= args.h_updated_value * server.h[client_id][k] / args.alpha
            #gunhot
            if args.client3 == 1:
                if node.nodeID == 3:
                    last_client3_model_before_train.load_state_dict(temp_server_model.state_dict())

            #gunhot
            # trainQ.put({'type': 'train', 'node': copy.deepcopy(node), 'lr':lr, \
            #     'model': copy.deepcopy(temp_server_model), 'round':roundIdx})
            trainQ.put({'type': 'train', 'node': node, 'lr':lr, \
                'model': temp_server_model, 'round':roundIdx})
            #gunhot
            count += 1

        for _ in range(count):
            msg = resultQ.get()
            weight = msg['weight']
            node_id = msg['id']
            node = nodes[node_id]
            
            # gunhot
            # Client 3의 모델 저장
            if args.client3 == 1:
                if node_id == 3:
                    with torch.no_grad():
                        last_client3_model_after_train.load_state_dict(weight)
                        print("Client 3의 모델 저장")
            # gunhot
            
            # set prev_grads for FedDyn optimizer
            if args.FedDyn == 1:
                prev_grads = msg['prev_grads']
                node.set_prev_grads(prev_grads)

            # upload weights to server
            server.update_node_info(weight, node_id)
            del msg
        server_model_before = copy.deepcopy(server.model.state_dict())
        # aggregate uploaded weights
        server.avg_parameters()
        with torch.no_grad():
            total_change = 0
            for name, param in server.model.named_parameters():
                if 'weight' not in name and 'bias' not in name:
                    continue
                diff = torch.norm(param - server_model_before[name])
                total_change += diff.item()
            print(f"[Round {roundIdx}] Avg Weight Δ Norm: {total_change:.4f}")
        server.active_clients = []

        # server.h_client_0.append(copy.deepcopy(server.h[0]))

        # if roundIdx % 5 == 0:
        if roundIdx % 5 == 0:
            # Save server model to file
            model_path = os.path.join(model_dir, f"server_model_round_{roundIdx}.pt")
            torch.save(server.model.state_dict(), model_path)
            preQ_count += 1

            # gunhot
            # Save Client 3's model to file if applicable
            if roundIdx % 10 == 0 and args.client3 == 1:
                model_after_train_path = os.path.join(model_dir, f"client3_after_train_round_{roundIdx}.pt")
                torch.save(last_client3_model_after_train.state_dict(), model_after_train_path)
                preClient3Q_after_train_count += 1

                model_before_train_path = os.path.join(model_dir, f"client3_before_train_round_{roundIdx}.pt")
                torch.save(last_client3_model_before_train.state_dict(), model_before_train_path)
                preClient3Q_before_train_count += 1
            # gunhot
            print(f"Elapsed Time : {time.time()-cur_time:.1f}")

    for _ in range(n_train_processes):
        trainQ.put('kill')

    # Train finished
    time.sleep(15)
    # Test start
    preQ_count = 20
    preClient3Q_after_train_count = 10
    preClient3Q_before_train_count = 10
    # Load models from files into testQ
    
    for i in range(preQ_count):
        model_path = os.path.join(model_dir, f"server_model_round_{(i+1)*5}.pt")
        model_state_dict = torch.load(model_path)

        model = GPT2Medium()
        model.load_state_dict(model_state_dict)
        testQ.put({'round': i+1, 'model': model})

    # gunhot
    if args.client3 == 1:
        testQ.put("client3AfterTrain")
        for i in range(preClient3Q_after_train_count):
            path = os.path.join(model_dir, f"client3_after_train_round_{(i+1)*10}.pt")
            state_dict = torch.load(path)
            model = GPT2Medium()
            model.load_state_dict(state_dict)
            testQ.put({'round': i+1, 'model': model})

        testQ.put("client3BeforeTrain")
        for i in range(preClient3Q_before_train_count):
            path = os.path.join(model_dir, f"client3_before_train_round_{(i+1)*10}.pt")
            state_dict = torch.load(path)
            model = GPT2Medium()
            model.load_state_dict(state_dict)
            testQ.put({'round': i+1, 'model': model})
    # gunhot

    testQ.put('kill')
    testQ.put(server.h_size)
    testQ.put(server.h_size_avg)
    testQ.put(server.pseudo_gradients_size)
    testQ.put(server.pseudo_gradients_size_avg)

    # cos = torch.nn.CosineSimilarity(dim=0)

    # cos_sim = []
    # for i in range(args.round):
    #     h_tmp = []

    #     for k in server.model.state_dict().keys():
    #         if 'weight' not in k and 'bias' not in k:
    #             continue

    #         h_tmp.append(server.h_client_0[i][k].flatten().clone())
        
    #     server.h_client_0[i] = torch.cat(h_tmp)
    
    # for i in range(args.round):
    #     cos_sim.append(cos(server.h_client_0[-1], server.h_client_0[i]))

    # testQ.put(cos_sim)
    for p in processes:
        p.join()
