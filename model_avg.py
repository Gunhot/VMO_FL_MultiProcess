import torch
import copy

def model_sum(params1, params2):
	with torch.no_grad():
		for k in params1.keys():
			params1[k] += params2[k]

def model_avg(param_sum, param_count, args, h, origin):
	w_avg = copy.deepcopy(param_sum)

	with torch.no_grad():
		for k in w_avg.keys():
			if param_count[k] == 0:
				w_avg[k] = origin[k]
				continue
			
			w_avg[k] = torch.div(w_avg[k], param_count[k])

			if args.FedDyn == 1:
			
				if 'weight' not in k and 'bias' not in k:
					continue
				#gunhot
				if args.hidden==0:
					# print("not hidden")
					w_avg[k] = w_avg[k] - h[k] / args.alpha
				else:
					# print("hidden")
					w_avg[k] = w_avg[k] 
				#gunhot

	return w_avg


		