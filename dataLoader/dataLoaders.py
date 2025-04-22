from .sampling import iid, noniid, noniid_nlp, noniid_num, iid_waterbirds
from .dataset import get_dataset
from torch.utils.data import DataLoader

def getNodeIndicies(nodeIDs, n_normal,  args):

	train_dataset, test_dataset = get_dataset(args)

	node_indices = {}
	n_nodes = len(nodeIDs)
	
	if args.iid == 1:
		dict_nodes = iid(train_dataset, n_nodes)

	elif args.iid == 0:
		dict_nodes = noniid(train_dataset, n_nodes, args.beta)

	for i, nodeID in enumerate(nodeIDs):
		node_indices[nodeID] = dict_nodes[i]

	# else:
	# 	if args.dataset == 'wikitext-2':
	# 		dict_nodes = iid(train_dataset, n_nodes)
	
	# 		for i, nodeID in enumerate(nodeIDs):
	# 			node_indices[nodeID] = dict_nodes[i]

	# 	else:
	# 		dict_nodes = noniid(train_dataset, n_nodes, args.beta)
	# 		for i, nodeID in enumerate(nodeIDs):
	# 			node_indices[nodeID] = dict_nodes[i]

	#testLoader = DataLoader(test_dataset, 256, shuffle=False, num_workers=4)
	
	return node_indices