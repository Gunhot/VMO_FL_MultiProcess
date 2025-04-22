import copy
import pickle
import os
import torch
from models import evaluate, evaluate_simple
from dataLoader.dataset import get_dataset, BatchDataset
from nn_models.resnet_sbn import resnet101
from torch.utils.data import DataLoader
import numpy as np
import random
from dataLoader.sampling import split_dataset
# from dataLoader.waterbirds import DRODataset

def gpu_train_worker(trainID, trainQ, resultQ, device, args):
	print("HI")
	if args.dataset == 'femnist':
		datasets, _ = get_dataset(args)
	
	else:
		print("Gunhot dataset")
		dataset, _ = get_dataset(args)

	while True:
		print(f"[DEBUG] train_worker got msg: ") 
		msg = trainQ.get()
		print(f"[DEBUG] train_worker got msg: {msg}") 
		if msg == 'kill':
			break

		elif msg['type'] == 'train':
			print("Gunhot train")
   
			processing_node = msg['node']

			model = msg['model']
			round = msg['round']
			
			if args.dataset == 'femnist':
				model_weight = processing_node.train(device, msg['lr'], model, datasets[processing_node.nodeID], round)
			else:
				print("Gunhot processing node")
				model_weight = processing_node.train(device, msg['lr'], model, dataset, round)

			# result = {'weight':copy.deepcopy(model_weight), 'id':processing_node.nodeID}
			result = {'weight': model_weight, 'id':processing_node.nodeID}
		
			# result['prev_grads'] = copy.deepcopy(processing_node.get_prev_grads())
			result['prev_grads'] = processing_node.get_prev_grads()
				
			resultQ.put(result)
			
		del processing_node
		del model
		del model_weight
		del msg

	#print("train end")

def gpu_test_worker(testID, testQ, device, args):

	# global_acc = {'data_1':list(), 'data_2':list()}
	global_acc = list()
	train_losses = list()
	test_losses = list()
 
	# gunhot
	client3After_acc = list()
	client3After_train_losses = list()
	client3After_test_losses = list()
	client3Before_acc = list()
	client3Before_train_losses = list()
	client3Before_test_losses = list()
	# gunhot
	if args.dataset == 'waterbirds':
		train_dataset, _, test_dataset = get_dataset(args)
		train_dataset = DRODataset(train_dataset, 2, 4, 256)
		test_dataset = DRODataset(test_dataset, 2, 4, 256)
	else:
		train_dataset, test_dataset = get_dataset(args)
	
	#test_loader_1 = DataLoader(test_dataset_1, 256, shuffle=False)
	#test_loader_2 = DataLoader(test_dataset_2, 256, shuffle=False)

	train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False)
	test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

	# gunhot
	save_acc = global_acc
	save_train_losses = train_losses
	save_test_losses = test_losses
	# gunhot
	while True:
		msg = testQ.get()
		
		if msg == 'kill':
			break

		elif msg == "client3AfterTrain":
			save_acc = client3After_acc
			save_train_losses = client3After_train_losses
			save_test_losses = client3After_test_losses
			continue
		elif msg == "client3BeforeTrain":
			save_acc = client3Before_acc
			save_train_losses = client3Before_train_losses
			save_test_losses = client3Before_test_losses
			continue
		else:		
			model = msg['model']
			round = msg['round']
			
			acc, train_loss, test_loss = evaluate_simple(model, train_loader, test_loader, args, device)
			save_acc.append(acc)
			save_train_losses.append(train_loss)
			save_test_losses.append(test_loss)
		
			print("Round: {} / Acc : {:.4f}".format(round, acc))
		
		del msg
	
	#  torch.save(model.state_dict(), f'../save/models/{args.dataset}_F[{args.FedDyn}]_S[{args.step}].pt')

	h_size = testQ.get()
	h_size_avg = testQ.get()
	gradients_size = testQ.get()
	gradients_size_avg = testQ.get()
	# cos_sim = testQ.get()
	cos_sim = list()
	print("GUNHOT PLEASEc")
	file_name = '../save/{}/gunhot_round[{}]_updated_h[{}]_h_updated_value[{}]_H[{}]_N[{}]_F[{}]_lr[{}]_decay[{}]_A[{}]_E[{}]_IID[{}]_Frac[{}]_S[{}]_pre[{}]_main.pkl'.\
		format(args.dataset, args.round, args.h_updated, args.h_updated_value, args.hidden, args.nodes, args.FedDyn, args.lr, args.lr_decay, args.alpha, args.local_epoch, args.iid, args.fraction, args.step, args.pretrained)
	
	with open(file_name, 'wb') as f:
		pickle.dump([global_acc, train_losses, test_losses, h_size, h_size_avg, gradients_size, gradients_size_avg, cos_sim], f)
	print("GUNHOT PLEASEc")
	# gunhot
	if args.client3 == 1:
		file_name = '../save/{}/gunhot_client3After[{}]_round[{}]_updated_h[{}]_h_updated_value[{}]_H[{}]_N[{}]_F[{}]_lr[{}]_decay[{}]_A[{}]_E[{}]_IID[{}]_Frac[{}]_S[{}]_pre[{}]_main.pkl'.\
			format(args.dataset, args.client3, args.round, args.h_updated, args.h_updated_value, args.hidden, args.nodes, args.FedDyn, args.lr, args.lr_decay, args.alpha, args.local_epoch, args.iid, args.fraction, args.step, args.pretrained)
		if client3After_acc != []:
			with open(file_name, 'wb') as f:
				pickle.dump([client3After_acc, client3After_train_losses, client3After_test_losses, h_size, h_size_avg, gradients_size, gradients_size_avg, cos_sim], f)
		file_name = '../save/{}/gunhot_client3Before[{}]_round[{}]_updated_h[{}]_h_updated_value[{}]_H[{}]_N[{}]_F[{}]_lr[{}]_decay[{}]_A[{}]_E[{}]_IID[{}]_Frac[{}]_S[{}]_pre[{}]_main.pkl'.\
			format(args.dataset, args.client3, args.round, args.h_updated, args.h_updated_value, args.hidden, args.nodes, args.FedDyn, args.lr, args.lr_decay, args.alpha, args.local_epoch, args.iid, args.fraction, args.step, args.pretrained)
		if client3Before_acc != []:
			with open(file_name, 'wb') as f:
				pickle.dump([client3Before_acc, client3Before_train_losses, client3Before_test_losses, h_size, h_size_avg, gradients_size, gradients_size_avg, cos_sim], f)
	 # gunhot

